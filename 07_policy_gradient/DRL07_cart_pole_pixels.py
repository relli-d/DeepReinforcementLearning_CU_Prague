#!/usr/bin/env python3
import argparse
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import npfl139
npfl139.require_version("2425.7")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.

# Path to save/load the model
MODEL_PATH = "reinforce_cartpole_pixels.pth"

class SharedCNN(nn.Module):
    """Feature extractor shared by both policy and value network."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=2),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 80, 80)
            dummy_output = self.conv(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).shape[1]

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # HWC -> CHW
        x = x.float() / 255.0
        x = self.conv(x)
        x = self.flatten(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, shared_cnn: SharedCNN, action_space: int):
        super().__init__()
        self.shared_cnn = shared_cnn
        self.fc = nn.Sequential(
            nn.Linear(self.shared_cnn.flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )

    def forward(self, x):
        features = self.shared_cnn(x)
        logits = self.fc(features)
        return torch.softmax(logits, dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, shared_cnn: SharedCNN):
        super().__init__()
        self.shared_cnn = shared_cnn
        self.fc = nn.Sequential(
            nn.Linear(self.shared_cnn.flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.shared_cnn(x)
        value = self.fc(features)
        return value.squeeze(-1)  # output shape [batch]


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_actions = env.action_space.n

    shared_cnn = SharedCNN().to(device)
    policy = PolicyNetwork(shared_cnn, n_actions).to(device)
    value_net = ValueNetwork(shared_cnn).to(device)

    if args.recodex:
        policy.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        policy.eval()

        while True:
            state, done = env.reset(options={"start_evaluation": True})[0], False
            while not done:
                state_tensor = torch.tensor(state[np.newaxis], device=device)
                with torch.no_grad():
                    action_probs = policy(state_tensor)
                action = torch.argmax(action_probs, dim=-1).item()
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
        return

    # TRAINING
    policy_optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)
    gamma = 0.99
    episodes = 2000

    for episode in range(episodes):
        states, actions, rewards = [], [], []
        state, done = env.reset()[0], False

        while not done:
            state_tensor = torch.tensor(state[np.newaxis], device=device)
            action_probs = policy(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        states_tensor = torch.tensor(np.array(states), device=device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)

        # Value predictions
        predicted_values = value_net(states_tensor).detach()

        # Advantage = (return - value)
        advantages = returns_tensor - predicted_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss
        action_probs = policy(states_tensor)
        log_probs = torch.log(action_probs.gather(1, actions_tensor.view(-1, 1)).squeeze())
        policy_loss = -(log_probs * advantages).mean()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        # Value loss (MSE)
        predicted_values_for_update = value_net(states_tensor)
        value_loss = (predicted_values_for_update - returns_tensor).pow(2).mean()

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        # if (episode + 1) % 10 == 0:
        #     print(f"Episode {episode+1}/{episodes}: Total Reward = {sum(rewards)}")

    # Save the model
    print("Saving the model...")
    torch.save(policy.state_dict(), MODEL_PATH)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make("CartPolePixels-v1"), main_args.seed, main_args.render_each)

    main(main_env, main_args)
