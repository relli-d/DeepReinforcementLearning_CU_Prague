#!/usr/bin/env python3
import argparse
import collections

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import npfl139
npfl139.require_version("2425.8")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="HalfCheetah-v5", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--buffer_size", default=500_000, type=int)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--tau", default=0.005, type=float)
parser.add_argument("--alpha", default=0.2, type=float)
parser.add_argument("--learning_rate", default=3e-4, type=float)
parser.add_argument("--model_path", default="cheetah_model.pth", type=str)

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.ptr = 0
        self.full = False
        self.states = np.zeros((capacity, state_dim), np.float32)
        self.actions = np.zeros((capacity, action_dim), np.float32)
        self.rewards = np.zeros(capacity, np.float32)
        self.next_states = np.zeros((capacity, state_dim), np.float32)
        self.dones = np.zeros(capacity, np.float32)

    def add(self, s, a, r, ns, d):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = ns
        self.dones[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.full = self.full or self.ptr == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.capacity if self.full else self.ptr, size=batch_size)
        return (torch.tensor(self.states[idxs]),
                torch.tensor(self.actions[idxs]),
                torch.tensor(self.rewards[idxs]),
                torch.tensor(self.next_states[idxs]),
                torch.tensor(self.dones[idxs]))

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU())
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, x):
        x = self.net(x)
        mean = self.mean(x)
        std = F.softplus(self.log_std(x)) + 1e-6
        return mean, std

    def sample(self, x):
        mean, std = self.forward(x)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1))
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1))

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.q1(x), self.q2(x)

def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    npfl139.startup(args.seed, args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = Actor(obs_dim, act_dim).to(device)
    critic = Critic(obs_dim, act_dim).to(device)
    target_critic = Critic(obs_dim, act_dim).to(device)
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=args.learning_rate)
    critic_opt = optim.Adam(critic.parameters(), lr=args.learning_rate)

    log_alpha = torch.tensor(np.log(args.alpha), requires_grad=True, device=device)
    alpha_opt = optim.Adam([log_alpha], lr=args.learning_rate)
    target_entropy = -act_dim

    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        state = env.reset(options={"start_evaluation": start_evaluation, "logging": logging})[0]
        rewards, done = 0, False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _ = actor.sample(state_tensor)
            action = action.cpu().numpy()[0]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    if args.recodex:
        actor.load_state_dict(torch.load(args.model_path, map_location=device))
        while True:
            evaluate_episode(True)
        return

    episode_returns = collections.deque(maxlen=100)
    total_steps = 0
    state, _ = env.reset()
    episode_return = 0

    while True:
        if total_steps < 10000:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action, _ = actor.sample(state_tensor)
                action = action.cpu().numpy()[0]

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(state, action, reward, next_state, float(done))

        episode_return += reward
        total_steps += 1

        if done:
            episode_returns.append(episode_return)
            avg_return = np.mean(episode_returns)
            if avg_return > 0:
                torch.save(actor.state_dict(), args.model_path)
            if avg_return >= 8000:
                torch.save(actor.state_dict(), args.model_path)
                break
            state, _ = env.reset()
            episode_return = 0
        else:
            state = next_state

        if total_steps >= 1000:
            for _ in range(1):
                s, a, r, ns, d = buffer.sample(args.batch_size)
                s = s.to(device)
                a = a.to(device)
                r = r.to(device).unsqueeze(1)
                ns = ns.to(device)
                d = d.to(device).unsqueeze(1)

                with torch.no_grad():
                    na, logp = actor.sample(ns)
                    q1, q2 = target_critic(ns, na)
                    min_q = torch.min(q1, q2)
                    alpha = log_alpha.exp()
                    target = r + args.gamma * (1 - d) * (min_q - alpha * logp)

                q1, q2 = critic(s, a)
                critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

                if total_steps % 2 == 0:
                    new_action, logp = actor.sample(s)
                    q1_pi, q2_pi = critic(s, new_action)
                    min_q_pi = torch.min(q1_pi, q2_pi)
                    actor_loss = (alpha * logp - min_q_pi).mean()

                    actor_opt.zero_grad()
                    actor_loss.backward()
                    actor_opt.step()

                    alpha_loss = -(log_alpha.exp() * (logp + target_entropy).detach()).mean()
                    alpha_opt.zero_grad()
                    alpha_loss.backward()
                    alpha_opt.step()

                    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)



if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    main_env = npfl139.EvaluationEnv(gym.make(main_args.env), main_args.seed, main_args.render_each)

    main(main_env, main_args)