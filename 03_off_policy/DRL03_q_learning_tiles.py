#!/usr/bin/env python3
import argparse
import gymnasium as gym
import numpy as np
import npfl139
npfl139.require_version("2425.3")

parser = argparse.ArgumentParser()
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=0.55, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=1500, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=1, type=float, help="Discounting factor.")
parser.add_argument("--tiles", default=8, type=int, help="Number of tiles.")
parser.add_argument("--episodes", default=3000, type=int, help="Number of training episodes.")

def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed.
    npfl139.startup(args.seed)
    # Initialize weights (for linear approximation of Q-values).
    W = np.zeros([env.observation_space.nvec[-1], env.action_space.n])
    epsilon = args.epsilon
    alpha = args.alpha

    def get_q_values(state_indices):
        return W[state_indices].sum(axis=0)
    
    def choose_action(state_indices, epsilon):
        #ε-greedy strategy
        if np.random.rand() < epsilon:
            return env.action_space.sample()
        return np.argmax(get_q_values(state_indices))

    for episode in range(args.episodes):
        # Perform episode
        state_indx, done = env.reset()[0], False
        while not done:
            # Choose an action using epsilon-greedy policy
            action = choose_action(state_indx, epsilon)
            # Take the action, observe next state and reward
            next_state_indx, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Compute the Q-value for the next state (greedy policy)
            Q_next = np.max(get_q_values(next_state_indx)) if not done else 0
            td_target = reward + args.gamma * Q_next
            td_error = td_target - get_q_values(state_indx)[action]

            # Q-learning update rule for linear approximation
            W[state_indx, action] += alpha * td_error

            # Move to the next state
            state_indx = next_state_indx
        alpha *= 0.99
        # Decay epsilon if necessary
        if args.epsilon_final_at:
            epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # Choose the greedy action (max Q-value)
            action = np.argmax(get_q_values(state))  # Linear approximation of Q-values
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        npfl139.DiscreteMountainCarWrapper(gym.make("MountainCar1000-v0"), tiles=main_args.tiles),
        main_args.seed, main_args.render_each)

    main(main_env, main_args)
