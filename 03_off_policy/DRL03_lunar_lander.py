#!/usr/bin/env python3
import argparse
import gymnasium as gym
import numpy as np
import npfl139
npfl139.require_version("2425.3")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--episodes", default=4000, type=int, help="Number of training episodes.")

def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed.
    npfl139.startup(args.seed)
    state_size, action_size = env.observation_space.n, env.action_space.n
    Q = np.random.uniform(low=-1, high=1, size=(state_size, action_size))


    # Assuming you have pre-trained your agent locally, perform only evaluation in ReCodEx
    if args.recodex:
        # TODO: Load the agent
        Q = np.load("Q.npy")
        while True:
            state, done = env.reset(start_evaluation=True)[0], False
            while not done:
                action = np.argmax(Q[state])
                state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

    # TODO: Implement a suitable RL algorithm and train the agent.
    for episode in range(args.episodes):
        state, done = env.reset()[0], False
        while not done:
            action = env.action_space.sample() if np.random.rand() < args.epsilon else np.argmax(Q[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            Q[state][action] = (1 - args.alpha) * Q[state][action] + args.alpha * (reward + args.gamma * np.max(Q[next_state]))
            state = next_state
    
    np.save("Q.npy", Q)
    print("Finished training: Agent saved.")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main_env = npfl139.EvaluationEnv(
        npfl139.DiscreteLunarLanderWrapper(gym.make("LunarLander-v3")), main_args.seed, main_args.render_each)
    main(main_env, main_args)
