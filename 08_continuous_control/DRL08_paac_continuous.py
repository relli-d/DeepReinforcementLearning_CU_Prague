#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import npfl139
npfl139.require_version("2425.8")

parser = argparse.ArgumentParser()
parser.add_argument("--recodex", default=False, action="store_true")
parser.add_argument("--render_each", default=0, type=int)
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--threads", default=1, type=int)
parser.add_argument("--entropy_regularization", default=0.01, type=float)
parser.add_argument("--envs", default=16, type=int)
parser.add_argument("--evaluate_each", default=100, type=int)
parser.add_argument("--evaluate_for", default=10, type=int)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--hidden_layer_size", default=128, type=int)
parser.add_argument("--learning_rate", default=1e-3, type=float)
parser.add_argument("--tiles", default=32, type=int)

class SoftplusWithEps(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.softplus(x) + self.eps

class Agent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        self.args = args
        self.actions = env.action_space.shape[0]
        self.tiles = args.tiles
        self.index_dim = int(np.max(env.observation_space.nvec))

        # Shared embedding for input
        self.embedding = nn.EmbeddingBag(self.index_dim, args.hidden_layer_size, mode="sum").to(self.device)

        # Actor network: two heads for mean and std
        self.actor_mu = nn.Sequential(
            nn.ReLU(),
            nn.Linear(args.hidden_layer_size, self.actions),
            nn.Tanh()
        ).to(self.device)

        self.actor_sd = nn.Sequential(
            nn.ReLU(),
            nn.Linear(args.hidden_layer_size, self.actions),
            SoftplusWithEps()
        ).to(self.device)

        # Critic network
        self.critic = nn.Sequential(
            nn.ReLU(),
            nn.Linear(args.hidden_layer_size, 1)
        ).to(self.device)

        # Optimizer
        self._optimizer = optim.Adam(
            list(self.embedding.parameters()) +
            list(self.actor_mu.parameters()) +
            list(self.actor_sd.parameters()) +
            list(self.critic.parameters()),
            lr=args.learning_rate
        )

    def embed(self, states):
        return self.embedding(states)
    
    @npfl139.typed_torch_function(device, torch.int64, torch.float32, torch.float32)
    def train(self, states, actions, returns):
        features = self.embed(states)
        values = self.critic(features).squeeze(-1)

        mus = self.actor_mu(features)
        sds = self.actor_sd(features)
        dist = torch.distributions.Normal(mus, sds)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)

        advantage = returns - values.detach()
        actor_loss = -(log_probs * advantage).mean() - self.args.entropy_regularization * entropy.mean()
        critic_loss = F.mse_loss(values, returns)
        loss = actor_loss + 0.5 * critic_loss

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    @npfl139.typed_torch_function(device, torch.int64)
    def predict_actions(self, states):
        with torch.no_grad():
            features = self.embed(states)
            mus = self.actor_mu(features)
            sds = self.actor_sd(features)
            return mus, sds

    @npfl139.typed_torch_function(device, torch.int64)
    def predict_values(self, states):
        with torch.no_grad():
            features = self.embed(states)
            return self.critic(features).squeeze(-1).cpu().numpy()


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()
    agent = Agent(env, args)

    def evaluate_episode(start_evaluation=False, logging=True) -> float:
        state = env.reset(options={"start_evaluation": start_evaluation, "logging": logging})[0]
        rewards, done = 0, False
        while not done:
            mus, _ = agent.predict_actions(np.array([state]))
            action = np.clip(mus[0], env.action_space.low, env.action_space.high)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    vector_env = gym.make_vec("MountainCarContinuous-v0", args.envs, gym.VectorizeMode.ASYNC,
                              wrappers=[lambda e: npfl139.DiscreteMountainCarWrapper(e, tiles=args.tiles)],
                              vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})
    states = vector_env.reset(seed=args.seed)[0]
    training = True

    while training:
        for _ in range(args.evaluate_each):
            mus, sds = agent.predict_actions(states)
            actions = np.random.normal(mus, sds)
            actions = np.clip(actions, vector_env.single_action_space.low, vector_env.single_action_space.high)
            next_states, rewards, terminated, truncated, _ = vector_env.step(actions)
            dones = terminated | truncated

            values = agent.predict_values(states)
            next_values = agent.predict_values(next_states)
            returns = rewards + args.gamma * next_values * (1.0 - dones.astype(np.float32))

            agent.train(states, actions, returns)
            states = next_states

        eval_returns = [evaluate_episode() for _ in range(args.evaluate_for)]
        avg_return = np.mean(eval_returns)
        print(f"Average return: {avg_return}")

        if avg_return >= 90:
            print("Solved!")
            training = False

    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main_env = npfl139.EvaluationEnv(
        npfl139.DiscreteMountainCarWrapper(gym.make("MountainCarContinuous-v0"), tiles=main_args.tiles),
        main_args.seed, main_args.render_each)
    main(main_env, main_args)