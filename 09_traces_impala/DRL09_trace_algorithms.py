#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np

import npfl139
npfl139.require_version("2425.8")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate alpha.")
parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration epsilon factor.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor gamma.")
parser.add_argument("--n", default=1, type=int, help="Use n-step method.")
parser.add_argument("--off_policy", default=False, action="store_true", help="Off-policy (less exploratory target)")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--trace_lambda", default=None, type=float, help="Trace factor lambda, if any.")
parser.add_argument("--vtrace_clip", default=None, type=float, help="V-Trace clip rho and c, if any.")
# If you add more arguments, ReCodEx will keep them with your default values.


def argmax_with_tolerance(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Argmax with small tolerance, choosing the value with smallest index on ties"""
    x = np.asarray(x)
    return np.argmax(x + 1e-6 >= np.max(x, axis=axis, keepdims=True), axis=axis)


def create_env(args: argparse.Namespace, report_each: int = 100, **kwargs) \
        -> tuple[npfl139.EvaluationEnv, np.ndarray, np.ndarray, np.ndarray]:
    # Create the environment
    env = npfl139.EvaluationEnv(gym.make("Taxi-v3"), seed=args.seed, report_each=report_each, **kwargs)

    # Extract a deterministic MDP into three NumPy arrays
    # - R[state][action] is the reward
    # - D[state][action] is the True/False value indicating end of episode
    # - N[state][action] is the next state
    R, D, N = [
        np.array([
            [env.unwrapped.P[s][a][0][i] for a in range(env.action_space.n)] for s in range(env.observation_space.n)])
        for i in [2, 3, 1]
    ]

    return env, R, D, N


def main(args: argparse.Namespace) -> np.ndarray:
    env, R, D, N = create_env(args)
    generator = np.random.RandomState(args.seed)
    V = np.zeros(env.observation_space.n)

    def compute_target_policy(V: np.ndarray) -> np.ndarray:
        epsilon = args.epsilon / 3 if args.off_policy else args.epsilon
        greedy_policy = np.eye(env.action_space.n)[argmax_with_tolerance(R + (1 - D) * args.gamma * V[N])]
        return (1 - epsilon) * greedy_policy + epsilon / env.action_space.n * np.ones_like(greedy_policy)

    for _ in range(args.episodes):
        state, done = env.reset()[0], False
        trajectory = []

        while not done:
            best_action = argmax_with_tolerance(R[state] + (1 - D[state]) * args.gamma * V[N[state]])
            action = best_action if generator.uniform() >= args.epsilon else env.action_space.sample()
            action_prob = args.epsilon / env.action_space.n + (1 - args.epsilon) * (action == best_action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            trajectory.append((state, action, reward, done, next_state, action_prob))

            while len(trajectory) >= args.n or (done and trajectory):
                steps = min(args.n, len(trajectory))
                states, actions, rewards, dones, next_states, behavior_probs = zip(*trajectory[:steps])
                s0, a0, _, _, _, b_prob = trajectory[0]

                if args.vtrace_clip is not None and args.trace_lambda is not None:
                    # --- Off-policy V-trace with λ ---
                    target_policy = compute_target_policy(V)
                    deltas, cs = [], []

                    for i in range(steps):
                        pi_prob = target_policy[states[i]][actions[i]]
                        rho = pi_prob / behavior_probs[i]
                        rho_bar = min(rho, args.vtrace_clip)
                        c = min(1.0, rho) * args.trace_lambda

                        v_next = 0 if dones[i] else V[next_states[i]]
                        delta = rho_bar * (rewards[i] + args.gamma * v_next - V[states[i]])
                        deltas.append(delta)
                        cs.append(c)

                    g = 0
                    for i in reversed(range(steps)):
                        g = deltas[i] + args.gamma * cs[i] * g

                    V[s0] += args.alpha * g

                elif args.trace_lambda is not None:
                    # --- On/off-policy λ-return ---
                    G_lambda = 0.0 if dones[-1] else V[next_states[-1]]
                    for i in reversed(range(steps)):
                        G_lambda = rewards[i] + args.gamma * (
                            (1 - args.trace_lambda) * V[next_states[i]] + args.trace_lambda * G_lambda
                        )
                        if dones[i]:
                            G_lambda = rewards[i]

                    if args.off_policy:
                        target_policy = compute_target_policy(V)
                        pi_prob = target_policy[s0][a0]
                        rho = pi_prob / b_prob
                        V[s0] += args.alpha * rho * (G_lambda - V[s0])
                    else:
                        V[s0] += args.alpha * (G_lambda - V[s0])

                elif args.off_policy and args.vtrace_clip is not None:
                    # --- Off-policy V-trace without λ ---
                    target_policy = compute_target_policy(V)
                    deltas, cs, rhos = [], [], []

                    for i in range(steps):
                        pi_prob = target_policy[states[i]][actions[i]]
                        rho = pi_prob / behavior_probs[i]
                        rho_bar = min(rho, args.vtrace_clip)
                        c = min(1.0, rho)
                        rhos.append(rho_bar)
                        cs.append(c)

                        v_next = 0 if dones[i] else V[next_states[i]]
                        delta = rewards[i] + args.gamma * v_next - V[states[i]]
                        deltas.append(delta)

                    g = 0
                    for i in reversed(range(steps)):
                        g = rhos[i] * deltas[i] + args.gamma * cs[i] * g

                    V[s0] += args.alpha * g

                elif args.off_policy:
                    # --- Off-policy n-step return using TD errors with control variates ---
                    target_policy = compute_target_policy(V)
                    G_cv = 0.0 if dones[-1] else V[next_states[-1]]
                    
                    for i in reversed(range(steps)):
                        s_i = states[i]
                        a_i = actions[i]
                        s_next = next_states[i]
                        done_i = dones[i]
                        
                        pi_prob = target_policy[s_i][a_i]
                        rho = pi_prob / behavior_probs[i]
                        
                        v_next = V[s_next] if not done_i else 0.0
                        G_cv = rho * (rewards[i] + args.gamma * G_cv) + (1 - rho) * V[s_i]
                    
                    V[s0] += args.alpha * (G_cv - V[s0])

                else:
                    # --- On-policy n-step return ---
                    G = 0.0
                    if not dones[-1]:
                        G = V[next_states[-1]]
                    for i in reversed(range(steps)):
                        G = rewards[i] + args.gamma * G

                    V[s0] += args.alpha * (G - V[s0])

                trajectory.pop(0)
            state = next_state

    return V




if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    V = main(main_args)

    env, R, D, N = create_env(main_args, report_each=0, evaluate_for=1000)
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            action = argmax_with_tolerance(R[state] + (1 - D[state]) * main_args.gamma * V[N[state]])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated