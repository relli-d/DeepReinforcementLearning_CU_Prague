#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np

import npfl139
npfl139.require_version("2425.2")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=0.00001, type=float, help="Exploration factor.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")

parser.add_argument("--episodes", default=10000, type=int, help="Training episodes.")
# parser.add_argument("--epsilon_decay", default=0.995, type=float, help="Epsilon decay factor.")
# parser.add_argument("--min_epsilon", default=0.01, type=float, help="Minimum epsilon value.")


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed.
    npfl139.startup(args.seed)

    # TODO: Variable creation and initialization
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

        #TRAINING PHASE
    training = True
    for _ in range(args.episodes):
        # Perform episode
        state, done = env.reset()[0], False
        while not done:
            # TODO: Perform an action.
                #Epsilon-greedy action selection
            if np.random.uniform(0, 1) < args.epsilon:
                action = env.action_space.sample()  #Explore
            else:
                action = np.argmax(q_table[state])  #Exploit

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # TODO: Update the action-value estimates
                #Q-learning update rule
            q_table[state, action] += args.alpha * (reward + args.gamma * np.max(q_table[next_state]) - q_table[state, action])
            state = next_state
        # #Decay epsilon
        # args.epsilon = max(args.min_epsilon, args.epsilon * args.epsilon_decay)

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO: Choose a greedy action
            action = np.argmax(q_table[state])  # Always take the best action
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        npfl139.DiscreteMountainCarWrapper(gym.make("MountainCar1000-v0")), main_args.seed, main_args.render_each)

    main(main_env, main_args)
