#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import npfl139
npfl139.require_version("2425.7")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="LunarLander-v3", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=45, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--entropy_regularization", default=0.003, type=float, help="Entropy regularization weight.")
parser.add_argument("--envs", default=32, type=int, help="Number of parallel environments.")
parser.add_argument("--evaluate_each", default=1000, type=int, help="Evaluate each number of batches.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=128, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="paac_actor.pt", type=str, help="Path to the actor model.")
parser.add_argument("--episodes", default=1000, type=int, help="Number of training episodes.")
parser.add_argument("--batch_steps", default=32, type=int, help="Number of steps to collect before update.")  # Important!



class Agent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        self.args = args

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        
        self._actor = nn.Sequential(
            nn.Linear(obs_dim, args.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(args.hidden_layer_size, args.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(args.hidden_layer_size, action_dim),
            nn.LogSoftmax(dim=-1)
        ).to(self.device)

        self._critic = nn.Sequential(
            nn.Linear(obs_dim, args.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(args.hidden_layer_size, args.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(args.hidden_layer_size, 1)
        ).to(self.device)

        self._optimizer = optim.Adam(
            list(self._actor.parameters()) + list(self._critic.parameters()),
            lr=args.learning_rate
        )

    @npfl139.typed_torch_function(device, torch.float32, torch.int64, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor, advantages: torch.Tensor) -> None:
        # Calculate both losses together
        log_probs = self._actor(states)
        dist = torch.distributions.Categorical(logits=log_probs)
        
        # Actor loss
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        actor_loss = -(action_log_probs * advantages).mean() - self.args.entropy_regularization * entropy
        
        # Critic loss
        values = self._critic(states).squeeze(-1)
        critic_loss = torch.nn.functional.mse_loss(values, returns)
        
        # Combined loss
        loss = actor_loss + 0.5 * critic_loss
        
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._actor.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self._critic.parameters(), max_norm=0.5)
        self._optimizer.step()  

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_actions(self, states: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            log_probs = self._actor(states)
            dist = torch.distributions.Categorical(logits=log_probs)
            return dist.sample().cpu().numpy()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_values(self, states: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            values = self._critic(states)
            return values.squeeze(-1).cpu().numpy()

    def save_actor(self, path: str) -> None:
        torch.save(self._actor.state_dict(), path)

    def load_actor(self, path: str) -> None:
        self._actor.load_state_dict(torch.load(path, map_location=self.device))


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()

    agent = Agent(env, args)
    initial_entropy = args.entropy_regularization
    episodes_played = 0

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        nonlocal episodes_played
        state = env.reset(options={"start_evaluation": start_evaluation, "logging": logging})[0]
        rewards, done = 0, False
        while not done:
            action = agent.predict_actions(np.array([state]))[0]
            state, reward, terminated, truncated, _ = env.step(action)
            args.entropy_regularization = initial_entropy * 0.99
            done = terminated or truncated
            rewards += reward
        episodes_played += 1
        return rewards

    if args.recodex:
        agent.load_actor(args.model_path)
        while True:
            evaluate_episode(start_evaluation=True)

    vector_env = gym.make_vec(args.env, args.envs, gym.VectorizeMode.ASYNC,
                              vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})
    states = vector_env.reset(seed=args.seed)[0]
    
    # Adjust these hyperparameters
    args.learning_rate = 3e-4
    args.batch_steps = 64  # Collect more steps before update
    args.entropy_regularization = 0.01  # Start with higher entropy

    best_avg_return = -np.inf

    while True:
        for _ in range(args.evaluate_each):
            # Collect a batch
            batch_states, batch_actions, batch_rewards, batch_dones, batch_values = [], [], [], [], []

            for _ in range(args.batch_steps):
                actions = agent.predict_actions(states)
                values = agent.predict_values(states)

                next_states, rewards, terminated, truncated, _ = vector_env.step(actions)
                dones = terminated | truncated

                batch_states.append(states)
                batch_actions.append(actions)
                batch_rewards.append(rewards)
                batch_dones.append(dones)
                batch_values.append(values)

                states = next_states

            # Compute bootstrap value
            next_values = agent.predict_values(states)

            # Compute returns
            returns = []
            discounted_return = next_values
            for rewards, dones in zip(reversed(batch_rewards), reversed(batch_dones)):
                discounted_return = rewards + args.gamma * discounted_return * (1.0 - dones)
                returns.insert(0, discounted_return)

            batch_states = np.concatenate(batch_states)
            batch_actions = np.concatenate(batch_actions)
            returns = np.concatenate(returns)
            batch_values = np.concatenate(batch_values)

            advantages = returns - batch_values
            advantages_std = advantages.std()
            if advantages_std > 1e-6:
                advantages = (advantages - advantages.mean()) / (advantages_std + 1e-8)

            agent.train(batch_states, batch_actions, returns, advantages)

        # Evaluation
        eval_returns = [evaluate_episode() for _ in range(args.evaluate_for)]
        avg_return = np.mean(eval_returns)
        print(f"Evaluation average return: {avg_return}")

        # Gradually reduce entropy
        args.entropy_regularization *= 0.995
        args.entropy_regularization = max(args.entropy_regularization, 0.001)

        # Save if this is the best avg_return so far
        if avg_return > best_avg_return:
            best_avg_return = avg_return
            agent.save_actor(args.model_path)
            print(f"New best model saved with average return {avg_return}")

        if avg_return >= 290:
            print("Target achieved. Ending training early.")
            break

        if episodes_played >= args.episodes:
            print(f"Reached {episodes_played} episodes. Ending training.")
            break

    # After training finishes, evaluate forever (ReCodEx or for visualization)
    while True:
        evaluate_episode(start_evaluation=True)

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main_env = npfl139.EvaluationEnv(gym.make(main_args.env), main_args.seed, main_args.render_each)
    main(main_env, main_args)