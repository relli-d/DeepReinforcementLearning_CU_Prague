#!/usr/bin/env python3
import argparse

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--gamma", default=1.0, type=float, help="Discount factor.")
parser.add_argument("--iterations", default=1, type=int, help="Number of iterations in policy evaluation step.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--steps", default=10, type=int, help="Number of policy evaluation/improvements to perform.")
# If you add more arguments, ReCodEx will keep them with your default values.


class GridWorld:
    # States in the gridworld are the following:
    # 0 1 2 3
    # 4 x 5 6
    # 7 8 9 10

    # The rewards are +1 in state 10 and -100 in state 6

    # Actions are ↑ → ↓ ←; with probability 80% they are performed as requested,
    # with 10% move 90° CCW is performed, with 10% move 90° CW is performed.
    states: int = 11
    actions: int = 4
    action_labels: list[str] = ["↑", "→", "↓", "←"]

    @staticmethod
    def step(state: int, action: int) -> list[tuple[float, float, int]]:
        return [GridWorld._step(0.8, state, action),
                GridWorld._step(0.1, state, (action + 1) % 4),
                GridWorld._step(0.1, state, (action + 3) % 4)]

    @staticmethod
    def _step(probability: float, state: int, action: int) -> tuple[float, float, int]:
        state += (state >= 5)
        x, y = state % 4, state // 4
        offset_x = -1 if action == 3 else action == 1
        offset_y = -1 if action == 0 else action == 2
        new_x, new_y = x + offset_x, y + offset_y
        if not (new_x >= 4 or new_x < 0 or new_y >= 3 or new_y < 0 or (new_x == 1 and new_y == 1)):
            state = new_x + 4 * new_y
        state -= (state >= 5)
        return (probability, +1 if state == 10 else -100 if state == 6 else 0, state)


def argmax_with_tolerance(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Argmax with small tolerance, choosing the value with smallest index on ties"""
    x = np.asarray(x)
    return np.argmax(x + 1e-6 >= np.max(x, axis=axis, keepdims=True), axis=axis)


def main(args: argparse.Namespace) -> tuple[list[float] | np.ndarray, list[int] | np.ndarray]:
    # TODO: Implement policy iteration algorithm, with `args.steps` steps of
    # policy evaluation/policy improvement. During policy evaluation, use the
    # current value function and perform `args.iterations` applications of the
    # Bellman equation. Perform the policy evaluation asynchronously (i.e., update
    # the value function in-place for states 0, 1, ...). During the policy
    # improvement, use the `argmax_with_tolerance` to choose the best action.

    # TODO: The final value function should be in `value_function` and final greedy policy in `policy`.
    # Start with zero value function and "go North" policy
    # counter = 0
    value_function = [0.0] * GridWorld.states
    policy = [0] * GridWorld.states #By initializing the policy with all zeros we have our default direction being NORTH
    
    for i in range(args.steps): #We do a number of evaluation+improvement equal to our STEPS
        # counter+=1
        for i in range(args.iterations):
            for s in range(GridWorld.states):
                PRS = GridWorld.step(s, policy[s])  # Get transition probabilities, rewards, and next states
                value_function[s] = sum(prob * (reward + args.gamma * value_function[next_state]) for prob, reward, next_state in PRS)
        for s in range(GridWorld.states):
            old_a = policy[s]
            term = []
            for i in range(GridWorld.actions):
                PRS = GridWorld.step(s, i)
                term.append(sum(prob * (reward + args.gamma * value_function[next_state]) for prob, reward, next_state in PRS)) 
            policy[s] = argmax_with_tolerance(term)
    return value_function, policy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    value_function, policy = main(args)

    # Print results
    for r in range(3):
        for c in range(4):
            state = 4 * r + c
            state -= (state >= 5)
            print("        " if r == 1 and c == 1 else "{:-8.2f}".format(value_function[state]), end="")
            print(" " if r == 1 and c == 1 else GridWorld.action_labels[policy[state]], end="")
        print()
