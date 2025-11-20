#!/usr/bin/env python3
import argparse
import collections
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import npfl139
npfl139.require_version("2425.6")

parser = argparse.ArgumentParser()
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--verify", default=False, action="store_true", help="Verify the loss computation")

parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epsilon", default=1.0, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.05, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=500, type=int, help="Episodes until reaching final epsilon.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=128, type=int, help="Size of hidden layer.")
parser.add_argument("--kappa", default=1.0, type=float, help="The quantile Huber loss threshold.")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--quantiles", default=51, type=int, help="Number of quantiles.")
parser.add_argument("--target_update_freq", default=100, type=int, help="Target update frequency.")


class Network:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        self.env = env
        input_size = int(np.prod(env.observation_space.shape))
        output_size = env.action_space.n * args.quantiles

        self.quantiles = args.quantiles
        self.gamma = args.gamma
        self.kappa = args.kappa

        self._model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, args.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(args.hidden_layer_size, output_size),
            nn.Unflatten(-1, (env.action_space.n, args.quantiles))
        ).to(self.device)

        self._optimizer = optim.Adam(self._model.parameters(), lr=args.learning_rate)

    @staticmethod
    def compute_loss(
        states_quantiles: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor,
        next_states_quantiles: torch.Tensor, gamma: float, kappa: float
    ) -> torch.Tensor:
        batch_size, n_actions, n_quantiles = states_quantiles.shape
        device = states_quantiles.device

        # Predicted quantiles of selected actions
        chosen_quantiles = states_quantiles[torch.arange(batch_size), actions]  # [B, Q]

        with torch.no_grad():
            # Double DQN target quantiles
            next_q_means = next_states_quantiles.mean(dim=2)
            best_actions = next_q_means.argmax(dim=1)
            next_chosen_quantiles = next_states_quantiles[torch.arange(batch_size), best_actions]
            target_quantiles = rewards.unsqueeze(1) + gamma * (1 - dones.unsqueeze(1)) * next_chosen_quantiles

        # Pairwise TD errors
        pred = chosen_quantiles.unsqueeze(2)  # [B, Q, 1]
        target = target_quantiles.unsqueeze(1)  # [B, 1, Q]
        td_errors = target - pred  # [B, Q, Q]

        # Quantile levels τ
        tau = (torch.arange(n_quantiles, device=device) + 0.5) / n_quantiles
        tau = tau.view(1, n_quantiles, 1)

        # Huber loss
        if kappa == 0.0:
            loss = torch.abs(td_errors)
        else:
            abs_td = torch.abs(td_errors)
            huber = torch.where(abs_td <= kappa, 0.5 * td_errors.pow(2), kappa * (abs_td - 0.5 * kappa))
            loss = huber

        quantile_weights = torch.abs(tau - (td_errors < 0).float())
        loss = quantile_weights * loss

        return loss.mean()

    @npfl139.typed_torch_function(device, torch.float32, torch.int64, torch.float32, torch.float32, torch.float32)
    def train(self, states, actions, rewards, dones, next_states) -> None:
        self._model.train()
        loss = self.compute_loss(
            self._model(states), actions, rewards, dones, self._model(next_states).detach(),
            self.gamma, self.kappa)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            quantiles = self._model(states)
            return quantiles.mean(dim=2).cpu().numpy()

    def copy_weights_from(self, other: "Network") -> None:
        self._model.load_state_dict(other._model.state_dict())


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> Callable | None:
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()

    if args.verify:
        return Network.compute_loss

    network = Network(env, args)
    target_network = Network(env, args)
    target_network.copy_weights_from(network)

    buffer = npfl139.ReplayBuffer()
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    epsilon = args.epsilon
    while True:
        state, done = env.reset()[0], False
        while not done:
            if np.random.uniform() < epsilon:
                action = np.random.randint(env.action_space.n)
            else:
                q_values = network.predict(state[np.newaxis])[0]
                action = int(np.argmax(q_values))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.append(Transition(state, action, reward, done, next_state))

            if len(buffer) >= args.batch_size:
                batch = buffer.sample(args.batch_size)
                network.train(batch.state, batch.action, batch.reward, batch.done, batch.next_state)

            state = next_state

        if args.epsilon_final_at:
            epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

        if env.episode % args.target_update_freq == 0:
            target_network.copy_weights_from(network)

        if env.should_stop():
            break

    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            q_values = network.predict(state[np.newaxis])[0]
            action = int(np.argmax(q_values))
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main_env = npfl139.EvaluationEnv(gym.make("CartPole-v1"), main_args.seed, main_args.render_each)
    result = main(main_env, main_args)

    if main_args.verify:
        # Run ReCodEx loss verification
        import numpy as np
        import torch
        np.testing.assert_allclose(result(
            states_quantiles=torch.tensor([[[-1.4, 0.1, 0.8], [-1.2, 0.1, 1.1]]]),
            actions=torch.tensor([1]), rewards=torch.tensor([-1.5]), dones=torch.tensor([0.]),
            next_states_quantiles=torch.tensor([[[-0.4, 0.1, 0.4], [-0.5, 1.0, 1.6]]]),
            gamma=0.2, kappa=1.5).numpy(force=True), 0.3294963, atol=1e-5)
