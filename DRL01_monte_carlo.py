#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np

import npfl139

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--episodes", default=10000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor.")


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed.
    npfl139.startup(args.seed)
    
    # Create Q and C tables
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    
    Q = np.zeros((num_states, num_actions))  # Estimated Q-values
    C = np.zeros((num_states, num_actions))  # Count of state-action pairs

    for _ in range(args.episodes):
        # Perform an episode, collecting states, actions, and rewards.
        state, done = env.reset()[0], False
        episode = []  # Store (state, action, reward) tuples
        
        while not done:
            # Compute `action` using epsilon-greedy policy
            if np.random.uniform(0, 1) <= args.epsilon:
                action = np.random.choice(num_actions)  # Explore
            else:
                action = np.argmax(Q[state])  # Exploit

            # Perform the action.
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode.append((state, action, reward))  # Store the step
            state = next_state
        
        # Compute returns from the received rewards and update Q and C.
        G = 0  # Initialize return        
        for state, action, reward in reversed(episode):
            G = reward + G  # Monte Carlo return (discount factor = 1)
            C[state, action] += 1
            Q[state, action] += (G - Q[state, action]) / C[state, action]  # Incremental mean update

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # Choose a greedy action
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    
    # Create the environment
    main_env = npfl139.EvaluationEnv(
        npfl139.DiscreteCartPoleWrapper(gym.make("CartPole-v1")), main_args.seed, main_args.render_each)
    
    main(main_env, main_args)
