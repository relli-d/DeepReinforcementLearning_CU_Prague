#!/usr/bin/env python3
import argparse
import collections
import copy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import npfl139
npfl139.require_version("2425.8")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="Pendulum-v1", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
parser.add_argument("--evaluate_each", default=50, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=50, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.98, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=64, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--noise_sigma", default=0.4, type=float, help="UB noise sigma.")
parser.add_argument("--noise_theta", default=0.15, type=float, help="UB noise theta.")
parser.add_argument("--replay_buffer_size", default=100_000, type=int, help="Replay buffer size")
parser.add_argument("--target_tau", default=0.005, type=float, help="Target network update weight.")
parser.add_argument("--policy_delay", default=2, type=int, help="Delay for policy update.")
parser.add_argument("--warmup_steps", default=10_000, type=int, help="Steps of pure random exploration.")



class Agent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        self.args = args

        class Actor(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden = nn.Linear(np.prod(env.observation_space.shape), args.hidden_layer_size)
                self.output = nn.Linear(args.hidden_layer_size, np.prod(env.action_space.shape))
                self.register_buffer('low', torch.tensor(env.action_space.low, dtype=torch.float32))
                self.register_buffer('high', torch.tensor(env.action_space.high, dtype=torch.float32))

            def forward(self, x):
                x = F.relu(self.hidden(x))
                x = torch.tanh(self.output(x))
                return self.low + (x + 1) * 0.5 * (self.high - self.low)

        class Critic(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden1 = nn.Linear(np.prod(env.observation_space.shape) + np.prod(env.action_space.shape), args.hidden_layer_size)
                self.hidden2 = nn.Linear(args.hidden_layer_size, args.hidden_layer_size)
                self.output = nn.Linear(args.hidden_layer_size, 1)

            def forward(self, state, action):
                x = torch.cat([state, action], dim=-1)
                x = F.relu(self.hidden1(x))
                x = F.relu(self.hidden2(x))
                return self.output(x)

        self.actor = Actor().to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.learning_rate)
        self.target_actor = copy.deepcopy(self.actor)

        self.critic1 = Critic().to(self.device)
        self.critic2 = Critic().to(self.device)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=args.learning_rate)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=args.learning_rate)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.train_step = 0

    @npfl139.typed_torch_function(device, torch.float32, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        predicted_q1 = self.critic1(states, actions)
        predicted_q2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(predicted_q1, returns)
        critic2_loss = F.mse_loss(predicted_q2, returns)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        if self.train_step % self.args.policy_delay == 0:
            predicted_actions = self.actor(states)
            actor_loss = -self.critic1(states, predicted_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            npfl139.update_params_by_ema(self.target_actor, self.actor, self.args.target_tau)

        npfl139.update_params_by_ema(self.target_critic1, self.critic1, self.args.target_tau)
        npfl139.update_params_by_ema(self.target_critic2, self.critic2, self.args.target_tau)

        self.train_step += 1

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_actions(self, states: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return self.actor(states).cpu().numpy()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_target(self, states: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            actions = self.target_actor(states)
            noise = torch.clamp(0.2 * torch.randn_like(actions), -0.5, 0.5)
            actions = actions + noise
            actions = torch.min(torch.max(actions, self.actor.low), self.actor.high)
            q1 = self.target_critic1(states, actions)
            q2 = self.target_critic2(states, actions)
            return torch.min(q1, q2).cpu().numpy()


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, shape, mu, theta, sigma):
        self.mu = mu * np.ones(shape)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        self.state += self.theta * (self.mu - self.state) + np.random.normal(scale=self.sigma, size=self.state.shape)
        return self.state


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()

    agent = Agent(env, args)

    replay_buffer = npfl139.MonolithicReplayBuffer(args.replay_buffer_size, args.seed)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        state = env.reset(options={"start_evaluation": start_evaluation, "logging": logging})[0]
        rewards, done = 0, False
        while not done:
            action = agent.predict_actions(np.array([state], np.float32))[0]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    noise = OrnsteinUhlenbeckNoise(env.action_space.shape[0], 0, args.noise_theta, args.noise_sigma)
    total_timesteps = 0

    if args.env == "Pendulum-v1":
        target_return = -200
    elif args.env == "InvertedDoublePendulum-v5":
        target_return = 9000
    else:
        target_return = None  # No automatic stopping for other environments


    training = True
    while training:
        for _ in range(args.evaluate_each):
            state, done = env.reset()[0], False
            noise.reset()
            while not done:

                if total_timesteps < args.warmup_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.predict_actions(np.array([state], np.float32))[0]
                    action = np.clip(action + noise.sample(), env.action_space.low, env.action_space.high)

                next_state, reward, terminated, truncated, _ = env.step(action)
                total_timesteps += 1
                done = terminated or truncated
                replay_buffer.append(Transition(state, action, reward, done, next_state))
                state = next_state

                if len(replay_buffer) < 4 * args.batch_size:
                    continue

                states, actions, rewards, dones, next_states = replay_buffer.sample(args.batch_size)

                rewards = rewards.reshape(-1, 1)
                dones = dones.reshape(-1, 1)

                next_q = agent.predict_target(next_states)
                targets = rewards + args.gamma * (1 - dones) * next_q

                agent.train(states, actions, targets)

        returns = [evaluate_episode(logging=False) for _ in range(args.evaluate_for)]
        mean_return = np.mean(returns)
        print(f"Evaluation after episode {env.episode}: {mean_return:.2f}")

        # Stop if the performance condition is satisfied
        if target_return is not None:
            if (args.env == "Pendulum-v1" and mean_return >= target_return) or \
               (args.env == "InvertedDoublePendulum-v5" and mean_return >= target_return):
                print(f"Target return reached ({mean_return:.2f} >= {target_return}). Stopping training.")
                training = False
                break

    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make(main_args.env), main_args.seed, main_args.render_each)

    main(main_env, main_args)