#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> np.ndarray:
    # Create the environment.
    env = gym.make("FrozenLake-v1")
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    # Set the random seed.
    np.random.seed(args.seed)

    # Behaviour policy is uniformly random.
    # Target policy uniformly chooses either action 1 or 2.
    V = np.zeros(env.observation_space.n)
    C = np.zeros(env.observation_space.n)

    for _ in range(args.episodes):
        state, done = env.reset()[0], False

        # Generate episode
        episode = []
        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode.append((state, action, reward))
            state = next_state

        # Compute return and importance sampling ratio
        G = 0
        W = 1.0  # Importance weight starts at 1
        
        for state, action, reward in reversed(episode):
            G = reward + G  # Update return without DISCOUNT on G

                        # Compute importance weight W
            if action in {1, 2}:  # Target policy picks actions 1 or 2 randomly (p=0.5)
                target_prob = 0.5
            else:
                target_prob = 0.0 # Target policy never picks 0 or 3
                break

            behavior_prob = 1 / env.action_space.n  # Random uniform policy (1/4)
            # if behavior_prob > 0:
            W *= target_prob / behavior_prob
            C[state] += W  # Update cumulative weight
            V[state] += (W / C[state]) * (G - V[state])  # Weighted importance sampling update
    return V


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    value_function = main(main_args)

    # Print the final value function V
    for row in value_function.reshape(4, 4):
        print(" ".join(["{:5.2f}".format(x) for x in row]))
