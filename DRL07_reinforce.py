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
parser.add_argument("--batch_size", default=40, type=int, help="Batch size.")
parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.985, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=128, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")


class Agent:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        self._policy = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, env.action_space.n),
            torch.nn.Softmax(dim=1),
        ).to(self.device)

        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=args.learning_rate)
        self._loss = torch.nn.CrossEntropyLoss(reduction='none')  # Not directly used, we compute loss manually
        self._args = args

    @npfl139.typed_torch_function(device, torch.float32, torch.int64, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        probs = self._policy(states)
        log_probs = torch.log(probs[range(len(actions)), actions])
        loss = -(log_probs * returns).mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

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

            G = 0
            returns = []
            for r in reversed(rewards):
                G = r + args.gamma * G
                returns.insert(0, G)

            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_returns.extend(returns)

        agent.train(batch_states, batch_actions, batch_returns)

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