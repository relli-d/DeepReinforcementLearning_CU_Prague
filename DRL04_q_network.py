#!/usr/bin/env python3
import argparse
import collections

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2425.4")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epsilon", default=1.0, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.05, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=1250, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.985, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=128, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.005, type=float, help="Learning rate.")
parser.add_argument("--target_update_freq", default=10, type=int, help="Target update frequency.")
parser.add_argument("--episodes", default=1700, type=int, help="Number of training steps.")


class Network:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        obs_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        self._model = torch.nn.Sequential(
            torch.nn.Linear(obs_size, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, action_size)
        ).to(self.device)

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args.learning_rate)
        self._loss = torch.nn.HuberLoss()

    @npfl139.typed_torch_function(device, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, q_values: torch.Tensor) -> None:
        self._model.train()
        predictions = self._model(states)
        loss = self._loss(predictions, q_values)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            return self._model(states)

    def copy_weights_from(self, other: "Network") -> None:
        self._model.load_state_dict(other._model.state_dict())


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()

    network, target_network = Network(env, args), Network(env, args)
    replay_buffer = npfl139.ReplayBuffer(max_length=1000000)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    epsilon = args.epsilon
    #training = True
    for _ in range(args.episodes):
        state, done = env.reset()[0], False
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = network.predict(state[np.newaxis])[0]
                action = np.argmax(q_values)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.append(Transition(state, action, reward, done, next_state))
            state = next_state

            if len(replay_buffer) > args.batch_size:
                batch = replay_buffer.sample(args.batch_size)
                states = np.array([t.state for t in batch])
                actions = np.array([t.action for t in batch])
                rewards = np.array([t.reward for t in batch])
                next_states = np.array([t.next_state for t in batch])
                dones = np.array([t.done for t in batch])
                
                target_qs = target_network.predict(next_states)
                targets = rewards + args.gamma * np.max(target_qs, axis=1) * ~dones
                q_values = network.predict(states)
                q_values[np.arange(args.batch_size), actions] = targets
                
                network.train(states, q_values)

        if env.episode % args.target_update_freq == 0:
            target_network.copy_weights_from(network)
        if args.epsilon_final_at:
            epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])
    
    print("Finished Training")

    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            action = np.argmax(network.predict(state[np.newaxis])[0])
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make("CartPole-v1"), main_args.seed, main_args.render_each)

    main(main_env, main_args)