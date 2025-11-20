#!/usr/bin/env python3
import argparse
import gymnasium as gym
import numpy as np
import npfl139

npfl139.require_version("2425.3")

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate alpha.")
parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration epsilon factor.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor gamma.")
parser.add_argument("--mode", default="sarsa", type=str, help="Mode (sarsa/expected_sarsa/tree_backup).")
parser.add_argument("--n", default=1, type=int, help="Use n-step method.")
parser.add_argument("--off_policy", default=False, action="store_true", help="Off-policy; use greedy as target")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=47, type=int, help="Random seed.")

def argmax_with_tolerance(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Argmax with small tolerance, choosing the value with smallest index on ties"""
    x = np.asarray(x)
    return np.argmax(x + 1e-6 >= np.max(x, axis=axis, keepdims=True), axis=axis)

def main(args: argparse.Namespace) -> np.ndarray:
    # Create a random generator with a fixed seed.
    generator = np.random.RandomState(args.seed)

    # Create the environment.
    env = npfl139.EvaluationEnv(gym.make("Taxi-v3"), seed=args.seed, report_each=min(200, args.episodes))

    Q = np.zeros((env.observation_space.n, env.action_space.n))

    def choose_next_action(Q: np.ndarray) -> tuple[int, float]:
        greedy_action = argmax_with_tolerance(Q[next_state])
        next_action = greedy_action if generator.uniform() >= args.epsilon else env.action_space.sample()
        return next_action, args.epsilon / env.action_space.n + (1 - args.epsilon) * (greedy_action == next_action)

    def compute_target_policy(Q: np.ndarray) -> np.ndarray:
        target_policy = np.eye(env.action_space.n)[argmax_with_tolerance(Q, axis=-1)]
        if not args.off_policy:
            target_policy = (1 - args.epsilon) * target_policy + args.epsilon / env.action_space.n
        return target_policy

    for _ in range(args.episodes):
        next_state, done = env.reset()[0], False
        S = [next_state]
        next_action, next_action_prob = choose_next_action(Q)
        A = [next_action]
        A_prob = [next_action_prob]
        R = []
        T = np.inf
        t = 0

        while True:
            if t < T:
                next_state, reward, terminated, truncated, _ = env.step(A[t])
                R.append(reward)
                S.append(next_state)
                done = terminated or truncated
                if done:
                    T = t + 1
                else:
                    next_action, next_action_prob = choose_next_action(Q)
                    A.append(next_action)
                    A_prob.append(next_action_prob)
            
            tau = t - args.n + 1
            
            if tau >= 0:
                rho = 1
                if args.mode == "sarsa":
                    imp_sampling_end = min(tau + args.n, T - 1) + 1
                    for i in range(tau + 1, imp_sampling_end):
                        rho *= compute_target_policy(Q)[S[i]][A[i]] / A_prob[i]
                    G = 0
                    for i in range(tau + 1, min(tau + args.n, T) + 1):
                        G += (args.gamma ** (i - tau - 1)) * R[i - 1]
                    if tau + args.n < T:
                        G += (args.gamma ** args.n) * Q[S[tau+args.n]][A[tau+args.n]]
                    if not args.off_policy:
                        rho = 1 #Remove importance sampling if on policy
                    Q[S[tau]][A[tau]] += args.alpha * rho * (G - Q[S[tau]][A[tau]])

                elif args.mode == "expected_sarsa":
                    #rho = 1
                    imp_sampling_end = min(tau + args.n - 1, T - 1) + 1
                    for i in range(tau + 1, imp_sampling_end):
                        rho *= compute_target_policy(Q)[S[i]][A[i]] / A_prob[i]
                    G = 0
                    for i in range(tau + 1, min(tau + args.n, T) + 1):
                        G += (args.gamma ** (i - tau - 1)) * R[i - 1]
                    if tau + args.n < T:
                        G += (args.gamma ** args.n) * np.dot(compute_target_policy(Q)[S[tau+args.n]], Q[S[tau+args.n]])
                    if not args.off_policy:
                        rho = 1
                    Q[S[tau]][A[tau]] += args.alpha * rho * (G - Q[S[tau]][A[tau]])

                elif args.mode == "tree_backup":
                    if t + 1 >= T:
                        G = R[T-1]
                    else:
                        G = R[t] + args.gamma * np.sum(compute_target_policy(Q)[S[t+1]] * Q[S[t+1]])
                    for k in range(min(t, T - 1), tau, -1):
                        G = R[k-1] + args.gamma * (np.sum(compute_target_policy(Q)[S[k]] * Q[S[k]]) - compute_target_policy(Q)[S[k]][A[k]] * Q[S[k]][A[k]]) + args.gamma * compute_target_policy(Q)[S[k]][A[k]] * G
                    Q[S[tau]][A[tau]] += args.alpha * (G - Q[S[tau]][A[tau]])

            t += 1
            if tau == T - 1:
                break
    
    return Q

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
