#!/usr/bin/env python3
import argparse
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import npfl139
npfl139.require_version("2425.10")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="SingleCollect-v0", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--clip_epsilon", default=0.2, type=float, help="Clipping epsilon.")
parser.add_argument("--entropy_regularization", default=0.01, type=float, help="Entropy regularization weight.")
parser.add_argument("--envs", default=8, type=int, help="Workers during experience collection.")
parser.add_argument("--epochs", default=4, type=int, help="Epochs to train each iteration.")
parser.add_argument("--evaluate_each", default=10, type=int, help="Evaluate each given number of iterations.")
parser.add_argument("--evaluate_for", default=5, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=64, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=5e-4, type=float, help="Learning rate.")
parser.add_argument("--critic_learning_rate", default=3e-3, type=float, help="Critic learning rate.")
parser.add_argument("--trace_lambda", default=0.9, type=float, help="Traces factor lambda.")
parser.add_argument("--worker_steps", default=128, type=int, help="Steps for each worker to perform.")


class Agent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        self._args = args
        obs_shape = env.observation_space.shape
        obs_dim = int(np.prod(obs_shape))
        action_dim = env.action_space.n

        self._actor = nn.Sequential(
            nn.Linear(obs_dim, args.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(args.hidden_layer_size, action_dim),
            nn.Softmax(dim=-1)
        ).to(self.device)

        self._critic = nn.Sequential(
            nn.Linear(obs_dim, args.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(args.hidden_layer_size, 1)
        ).to(self.device)

        self._actor_opt = optim.Adam(self._actor.parameters(), lr=args.learning_rate)
        self._critic_opt = optim.Adam(self._critic.parameters(), lr=args.critic_learning_rate)

    @npfl139.typed_torch_function(device, torch.float32, torch.int64, torch.float32, torch.float32, torch.float32)
    def train(self, states, actions, old_log_probs, advantages, returns):
        for _ in range(self._args.epochs):
            dataset = DataLoader(TensorDataset(states, actions, old_log_probs, advantages, returns),
                                 batch_size=self._args.batch_size, shuffle=True)
            for s, a, logp_old, adv, ret in dataset:
                probs = self._actor(s)
                dist = torch.distributions.Categorical(probs)
                logp = dist.log_prob(a)
                ratio = torch.exp(logp - logp_old)

                clipped = torch.clamp(ratio, 1 - self._args.clip_epsilon, 1 + self._args.clip_epsilon)
                policy_loss = -torch.min(ratio * adv, clipped * adv).mean()

                entropy = dist.entropy().mean()
                entropy_loss = -self._args.entropy_regularization * entropy

                values = self._critic(s).squeeze(-1)
                value_loss = F.mse_loss(values, ret)

                loss = policy_loss + entropy_loss + value_loss

                self._actor_opt.zero_grad()
                self._critic_opt.zero_grad()
                loss.backward()
                self._actor_opt.step()
                self._critic_opt.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_actions(self, states):
        with torch.no_grad():
            return self._actor(states)

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_values(self, states):
        with torch.no_grad():
            return self._critic(states).squeeze(-1)

    def save(self, path: str) -> None:
        torch.save({
            "actor_state_dict": self._actor.state_dict(),
            "critic_state_dict": self._critic.state_dict(),
            "actor_optimizer": self._actor_opt.state_dict(),
            "critic_optimizer": self._critic_opt.state_dict()
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self._actor.load_state_dict(checkpoint["actor_state_dict"])
        self._critic.load_state_dict(checkpoint["critic_state_dict"])
        self._actor_opt.load_state_dict(checkpoint["actor_optimizer"])
        self._critic_opt.load_state_dict(checkpoint["critic_optimizer"])


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()
    agent = Agent(env, args)

    def evaluate_episode(start_evaluation=False, logging=True):
        state = env.reset(options={"start_evaluation": start_evaluation, "logging": logging})[0]
        rewards, done = 0, False
        while not done:
            probs = agent.predict_actions(np.array([state]))[0]
            action = np.argmax(probs)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards
    
    if args.recodex:
        if os.path.exists("ppo.pth"):
            print("Loading saved model from ppo.pth")
            agent.load("ppo.pth")
            while True:
                evaluate_episode(True)
        else:
            print("No saved model found!")


    vector_env = gym.make_vec(args.env, args.envs, gym.VectorizeMode.ASYNC,
                              vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})
    state = vector_env.reset(seed=args.seed)[0]

    iteration = 0
    while True:
        states, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []

        for _ in range(args.worker_steps):
            probs = agent.predict_actions(state)
            dist = torch.distributions.Categorical(torch.tensor(probs, device=Agent.device))
            action_tensor = dist.sample()
            logp = dist.log_prob(action_tensor)

            action = action_tensor.cpu().numpy()
            logp = logp.cpu().numpy()

            value = agent.predict_values(state)
            next_state, reward, terminated, truncated, _ = vector_env.step(action)
            done = np.logical_or(terminated, truncated)

            states.append(state)
            actions.append(action)
            log_probs_old.append(logp)
            rewards.append(reward)
            dones.append(done)
            values.append(value)

            state = next_state

        states = np.array(states)
        actions = np.array(actions)
        log_probs_old = np.array(log_probs_old)
        rewards = np.array(rewards)
        dones = np.array(dones)
        values = np.array(values)

        last_values = agent.predict_values(state)
        next_values = np.vstack([values[1:], last_values[None, :]])

        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        gae = np.zeros(args.envs)

        for t in reversed(range(args.worker_steps)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + args.gamma * next_values[t] * mask - values[t]
            gae = delta + args.gamma * args.trace_lambda * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        advantages = advantages.flatten()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        agent.train(
            states.reshape(-1, states.shape[2]),
            actions.flatten(),
            log_probs_old.flatten(),
            advantages,
            returns.flatten()
        )

        if iteration % args.evaluate_each == 0:
            eval_returns = [evaluate_episode() for _ in range(args.evaluate_for)]
            avg_return = np.mean(eval_returns)
            #print(f"Iteration {iteration}: avg return = {avg_return:.2f}")
            if avg_return >= 550:
                print("Target performance reached. Saving full model.")
                agent.save("ppo.pth")
                break

    while True:
        evaluate_episode(start_evaluation=True)



if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make(main_args.env), main_args.seed, main_args.render_each)

    main(main_env, main_args)