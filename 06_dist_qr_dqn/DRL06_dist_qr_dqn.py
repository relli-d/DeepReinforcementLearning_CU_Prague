#!/usr/bin/env python3
import argparse
import collections
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2425.6")

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=False, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--verify", default=False, action="store_true", help="Verify the loss computation")
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epsilon", default=1.0, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=100, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")
parser.add_argument("--kappa", default=1.0, type=float, help="The quantile Huber loss threshold.")
parser.add_argument("--learning_rate", default=0.02, type=float, help="Learning rate.")
parser.add_argument("--quantiles", default=100, type=int, help="Number of quantiles.")
parser.add_argument("--target_update_freq", default=5, type=int, help="Target update frequency.")
parser.add_argument("--episodes", default=200, type=int, help="Number of episodes.")

class Network:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        obs_dim = int(np.prod(env.observation_space.shape))
        num_actions = env.action_space.n

        self._model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(obs_dim, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, num_actions * args.quantiles),
            # Reshapes the output to [batch_size, num_actions, quantiles].
            torch.nn.Unflatten(1, (int(num_actions), int(args.quantiles)))
            # In QR-DQN, we don’t predict one Q-value per action
            # instead, we predict a distribution of Q-values, represented as a set of quantiles,
            # these quantiles approximate the distribution of future returns for each action.
        )
        self._model.to(self.device)

        self.gamma = args.gamma
        self.kappa = args.kappa
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args.learning_rate)

    @staticmethod
    def compute_loss(states_quantiles: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
                     dones: torch.Tensor, next_states_quantiles: torch.Tensor,
                     gamma: float, kappa: float) -> torch.Tensor:

        batch_size, num_actions, num_quant = states_quantiles.shape
        pred_quant = states_quantiles[torch.arange(batch_size), actions]

        next_q_values = next_states_quantiles.mean(dim=2)
        best_actions = next_q_values.argmax(dim=1)
        next_quant = next_states_quantiles[torch.arange(batch_size), best_actions].detach()

        target_quant = rewards.unsqueeze(1) + gamma * (1 - dones).unsqueeze(1) * next_quant
        td_errors = target_quant.unsqueeze(1) - pred_quant.unsqueeze(2)

        tau = ((torch.arange(num_quant, dtype=pred_quant.dtype, device=pred_quant.device) + 0.5)
               / num_quant).view(1, num_quant, 1)

        abs_errors = td_errors.abs()
        if kappa > 0:
            huber = torch.where(abs_errors <= kappa,
                                0.5 * td_errors.pow(2),
                                kappa * (abs_errors - 0.5 * kappa))
        else:
            huber = abs_errors

        indicator = (td_errors < 0).float()
        quantile_loss = (torch.abs(tau - indicator) * huber).sum(dim=2).mean() / num_quant

        return quantile_loss

    def train(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
              dones: torch.Tensor, next_states: torch.Tensor, target_network: "Network") -> None:
        self._model.train()
        with torch.no_grad():
            next_quantiles = target_network._model(next_states)
        current_quantiles = self._model(states)

        loss = self.compute_loss(current_quantiles, actions, rewards, dones, next_quantiles,
                                 self.gamma, self.kappa)

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 10.0)
        self._optimizer.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            return self._model(states).mean(dim=2).cpu().numpy()

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

    replay_buffer = npfl139.ReplayBuffer()
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    epsilon = args.epsilon
    episode_count = 0

    for episode_count in range(args.episodes):
        state, done = env.reset()[0], False
        while not done:
            q_values = network.predict(state[np.newaxis])[0]
            action = np.random.randint(env.action_space.n) if np.random.rand() < epsilon else np.argmax(q_values)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.append(Transition(state, action, reward, done, next_state))

            if len(replay_buffer) >= args.batch_size:
                batch = replay_buffer.sample(args.batch_size)
                states, actions, rewards, dones, next_states = zip(*batch)
                network.train(
                    torch.tensor(np.array(states), dtype=torch.float32),
                    torch.tensor(np.array(actions), dtype=torch.int64),
                    torch.tensor(np.array(rewards), dtype=torch.float32),
                    torch.tensor(np.array(dones), dtype=torch.float32),
                    torch.tensor(np.array(next_states), dtype=torch.float32),
                    target_network
                )

            state = next_state

        if args.epsilon_final_at:
            epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

        if env.episode % args.target_update_freq == 0:
            target_network.copy_weights_from(network)

        if args.render_each and env.episode % args.render_each == 0:
            env.render()

        episode_count += 1

    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            q_values = network.predict(state[np.newaxis])[0]
            action = np.argmax(q_values)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main_env = npfl139.EvaluationEnv(gym.make("CartPole-v1"), main_args.seed, main_args.render_each)

    result = main(main_env, main_args)

    if main_args.verify:
        import numpy as np
        import torch
        np.testing.assert_allclose(result(
            states_quantiles=torch.tensor([[[-1.4, 0.1, 0.8], [-1.2, 0.1, 1.1]]]),
            actions=torch.tensor([1]), rewards=torch.tensor([-1.5]), dones=torch.tensor([0.]),
            next_states_quantiles=torch.tensor([[[-0.4, 0.1, 0.4], [-0.5, 1.0, 1.6]]]),
            gamma=0.2, kappa=1.5).numpy(force=True), 0.3294963, atol=1e-5)

        np.testing.assert_allclose(result(
            states_quantiles=torch.tensor([[[-0.0, 0.1, 1.2], [-1.8, -0.2, -0.1]],
                                           [[-0.3, 0.5, 1.3], [-1.4, -0.7, -0.1]],
                                           [[-0.3, -0.0, 1.9], [-1.1, -0.2, -0.1]]]),
            actions=torch.tensor([1, 0, 1]), rewards=torch.tensor([0.5, 1.4, 0.1]), dones=torch.tensor([0., 0., 1.]),
            next_states_quantiles=torch.tensor([[[-1.1, 0.2, 0.3], [-0.4, 1.1, 1.3]],
                                                [[-0.6, -0.5, 2.0], [-0.3, 0.2, 0.4]],
                                                [[-0.9, 0.7, 2.3], [-0.3, 0.7, 0.7]]]),
            gamma=0.8, kappa=0.0).numpy(force=True), 0.4392593, atol=1e-5)

        np.testing.assert_allclose(result(
            states_quantiles=torch.tensor([[[-0.8, -0.5, -0.0, 0.3], [-0.7, -0.2, -0.2, 1.6]],
                                           [[-1.5, -1.4, -0.6, 0.1], [-2.1, -1.5, -0.3, 0.3]]]),
            actions=torch.tensor([1, 0]), rewards=torch.tensor([-0.0, 0.7]), dones=torch.tensor([1., 0.]),
            next_states_quantiles=torch.tensor([[[-1.2, 0.3, 0.4, 0.7], [-1.2, -0.1, 0.4, 2.2]],
                                                [[-1.5, 0.2, 0.2, 0.5], [-0.9, 0.4, 0.5, 1.3]]]),
            gamma=0.3, kappa=3.5).numpy(force=True), 0.2906375, atol=1e-5)
