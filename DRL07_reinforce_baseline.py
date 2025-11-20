#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2425.7")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=False, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
parser.add_argument("--episodes", default=200, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=32, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.05, type=float, help="Learning rate.")


class Agent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        # Policy network
        self._policy = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, n_actions),
            torch.nn.Softmax(dim=1),
        ).to(self.device)

        # Baseline (value) network
        self._baseline = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, 1)
        ).to(self.device)

        self._policy_optimizer = torch.optim.Adam(self._policy.parameters(), lr=args.learning_rate)
        self._policy_scheduler = torch.optim.lr_scheduler.StepLR(self._policy_optimizer, step_size=50, gamma=0.9)
        self._baseline_optimizer = torch.optim.Adam(self._baseline.parameters(), lr=args.learning_rate * 0.1, weight_decay=1e-4)
        self._baseline_loss = torch.nn.MSELoss()
        self._args = args

    def _initialize_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)

        self._policy.apply(self._initialize_weights)
        self._baseline.apply(self._initialize_weights)

    @npfl139.typed_torch_function(device, torch.float32, torch.int64, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        # Baseline prediction
        predicted_baseline = self._baseline(states).squeeze()

        # Compute advantage
        advantage = returns - predicted_baseline.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)  # Normalize advantage

        # Policy loss (negative log prob * advantage + entropy regularization)
        log_probs = torch.log(self._policy(states)[range(len(actions)), actions])
        entropy = -(self._policy(states) * torch.log(self._policy(states) + 1e-8)).sum(dim=1).mean()
        policy_loss = -(log_probs * advantage).mean() - 0.01 * entropy

        # Update policy network
        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        # Train baseline to predict returns
        baseline_loss = self._baseline_loss(predicted_baseline, returns)
        self._baseline_optimizer.zero_grad()
        baseline_loss.backward()
        self._baseline_optimizer.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return self._policy(states).cpu().numpy()


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()

    agent = Agent(env, args)

    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            states, actions, rewards = [], [], []
            state, done = env.reset()[0], False
            while not done:
                probs = agent.predict([state])[0]
                action = np.random.choice(len(probs), p=probs)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # Compute discounted returns
            G = 0
            returns = []
            for r in reversed(rewards):
                G = r + args.gamma * G
                returns.insert(0, G)

            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_returns.extend(returns)

        # Do not normalize returns
        returns = torch.tensor(batch_returns, dtype=torch.float32, device=agent.device)

        agent.train(batch_states, batch_actions, returns)

    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            probs = agent.predict([state])[0]
            action = np.argmax(probs)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make("CartPole-v1"), main_args.seed, main_args.render_each)

    main(main_env, main_args)