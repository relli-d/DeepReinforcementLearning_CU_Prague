#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random

import npfl139
npfl139.require_version("2425.7")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--epsilon", default=1.0, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.05, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=400, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.985, type=float, help="Discounting factor.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--target_update_freq", default=1000, type=int, help="Target update frequency.")
parser.add_argument("--episodes", default=1000, type=int, help="Number of episodes.")
parser.add_argument("--model_path", default="cart_pole_dqn_model.pth", type=str, help="Path to save/load model.")


Transition = namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

class DQN(nn.Module):
    def __init__(self, action_space: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),  # (3, 80, 80) -> (16, 19, 19)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # (16, 19, 19) -> (32, 8, 8)
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, action_space),
        )

    def forward(self, x):
        x = x.float() / 255.0
        return self.fc(self.conv(x))

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def append(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return Transition(*map(np.stack, zip(*batch)))

    def __len__(self):
        return len(self.buffer)

def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    action_space = env.action_space.n
    online_net = DQN(action_space).to(device)
    target_net = DQN(action_space).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=1e-4)
    buffer = ReplayBuffer(100_000)
    batch_size = args.batch_size
    gamma = args.gamma
    epsilon_start = args.epsilon
    epsilon_final = args.epsilon_final
    epsilon_decay = args.epsilon_final_at
    target_update_freq = args.target_update_freq
    train_freq = 4
    warmup_steps = 10000
    max_steps = args.episodes

    def select_action(state, step):
        epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * step / epsilon_decay)
        if random.random() < epsilon:
            return np.random.randint(action_space)
        else:
            with torch.no_grad():
                state_t = torch.tensor(state[None], device=device, dtype=torch.uint8).permute(0, 3, 1, 2)
                q_values = online_net(state_t)
                return int(torch.argmax(q_values).item())

    if args.recodex:
        # Load trained model
        online_net.load_state_dict(torch.load(args.model_path, map_location=device))
        online_net.eval()

        # Final evaluation
        while True:
            state, done = env.reset(options={"start_evaluation": True})[0], False
            while not done:
                state_t = torch.tensor(state[None], device=device, dtype=torch.uint8).permute(0, 3, 1, 2)
                with torch.no_grad():
                    action = int(torch.argmax(online_net(state_t)).item())
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
        return

    # Training
    state, _ = env.reset()
    total_steps = 0
    for episode in range(args.episodes):  # Iterate over episodes
        done = False
        while not done:
            action = select_action(state, total_steps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.append(state, action, reward, done, next_state)

            if done:
                state, _ = env.reset()
            else:
                state = next_state

            total_steps += 1

            if total_steps >= warmup_steps and total_steps % train_freq == 0:
                batch = buffer.sample(batch_size)

                states = torch.tensor(batch.state, device=device, dtype=torch.uint8).permute(0, 3, 1, 2)
                actions = torch.tensor(batch.action, device=device, dtype=torch.int64)  # Ensure actions are int64
                rewards = torch.tensor(batch.reward, device=device, dtype=torch.float32)
                dones = torch.tensor(batch.done, device=device, dtype=torch.float32)
                next_states = torch.tensor(batch.next_state, device=device, dtype=torch.uint8).permute(0, 3, 1, 2)

                q_values = online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1).values
                    target = rewards + (1 - dones) * gamma * next_q_values

                loss = nn.functional.smooth_l1_loss(q_values, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if total_steps % target_update_freq == 0:
                target_net.load_state_dict(online_net.state_dict())

    # Save model for ReCodEx
    torch.save(online_net.state_dict(), args.model_path)

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make("CartPolePixels-v1"), main_args.seed, main_args.render_each)

    main(main_env, main_args)
