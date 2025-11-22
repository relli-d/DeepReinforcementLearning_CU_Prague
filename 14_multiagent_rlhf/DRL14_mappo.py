#!/usr/bin/env python3
import argparse
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import npfl139
npfl139.require_version("2425.13")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--agents", default=2, type=int, help="Agents to use.")
parser.add_argument("--recodex", default=True, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=512, type=int, help="Batch size.")
parser.add_argument("--clip_epsilon", default=0.2, type=float, help="Clipping epsilon.")
parser.add_argument("--entropy_regularization", default=0.02, type=float, help="Entropy regularization weight.")
parser.add_argument("--envs", default=16, type=int, help="Workers during experience collection.")
parser.add_argument("--epochs", default=8, type=int, help="Epochs to train each iteration.")
parser.add_argument("--evaluate_each", default=10, type=int, help="Evaluate each given number of iterations.")
parser.add_argument("--evaluate_for", default=5, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning rate.")
parser.add_argument("--trace_lambda", default=0.95, type=float, help="Traces factor lambda.")
parser.add_argument("--worker_steps", default=512, type=int, help="Steps for each worker to perform.")
parser.add_argument("--shared_network", default=False, action="store_true", help="Use shared network for both agents.")
parser.add_argument("--max_iterations", default=100, type=int, help="Maximum training iterations.")
parser.add_argument("--model_path", default="mappo.pth", type=str, help="Path to save the model.")



class Agent:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, args: argparse.Namespace, agent_id: int = 0) -> None:
        self._args = args
        self._agent_id = agent_id
        
        # Input size - if shared network, add agent ID as one-hot
        input_size = observation_space.shape[0]
        if args.shared_network:
            input_size += args.agents  # Add agent ID as one-hot
        
        # Create an actor using two hidden layers for better capacity
        self._actor = nn.Sequential(
            nn.Linear(input_size, args.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(args.hidden_layer_size, args.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(args.hidden_layer_size, action_space.n)
        ).to(self.device)

        # Create a critic with two hidden layers
        self._critic = nn.Sequential(
            nn.Linear(input_size, args.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(args.hidden_layer_size, args.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(args.hidden_layer_size, 1)
        ).to(self.device)

        # Initialize weights using Xavier initialization
        self._initialize_weights()

        # Optimizers with weight decay for regularization
        self._actor_optimizer = optim.Adam(self._actor.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        self._critic_optimizer = optim.Adam(self._critic.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        
        # Learning rate schedulers
        self._actor_scheduler = optim.lr_scheduler.StepLR(self._actor_optimizer, step_size=20, gamma=0.9)
        self._critic_scheduler = optim.lr_scheduler.StepLR(self._critic_optimizer, step_size=20, gamma=0.9)

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in [self._actor, self._critic]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)

    def _prepare_input(self, states):
        """Prepare input for shared network by adding agent ID"""
        if not self._args.shared_network:
            return states
        
        batch_size = states.shape[0]
        agent_onehot = torch.zeros(batch_size, self._args.agents, device=self.device)
        agent_onehot[:, self._agent_id] = 1.0
        return torch.cat([states, agent_onehot], dim=1)

    @npfl139.typed_torch_function(device, torch.float32, torch.int64, torch.float32, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, action_probs: torch.Tensor,
              advantages: torch.Tensor, returns: torch.Tensor) -> None:
        
        # Prepare input
        inputs = self._prepare_input(states)
        
        # Train actor (policy)
        self._actor_optimizer.zero_grad()
        
        # Get current policy logits and probabilities
        logits = self._actor(inputs)
        current_policy = F.softmax(logits, dim=-1)
        current_action_probs = current_policy.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Compute probability ratio
        ratio = current_action_probs / (action_probs + 1e-8)
        
        # PPO clipped objective
        clipped_ratio = torch.clamp(ratio, 1 - self._args.clip_epsilon, 1 + self._args.clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # Entropy regularization
        entropy = -(current_policy * torch.log(current_policy + 1e-8)).sum(dim=-1).mean()
        entropy_loss = -self._args.entropy_regularization * entropy
        
        # Total actor loss
        actor_loss = policy_loss + entropy_loss
        actor_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self._actor.parameters(), max_norm=0.5)
        self._actor_optimizer.step()

        # Train critic (value function)
        self._critic_optimizer.zero_grad()
        
        values = self._critic(inputs).squeeze(-1)
        critic_loss = F.mse_loss(values, returns)
        
        critic_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self._critic.parameters(), max_norm=0.5)
        self._critic_optimizer.step()

    def step_schedulers(self):
        """Step learning rate schedulers"""
        self._actor_scheduler.step()
        self._critic_scheduler.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_actions(self, states: torch.Tensor) -> np.ndarray:
        # Return predicted action probabilities.
        with torch.no_grad():
            inputs = self._prepare_input(states)
            logits = self._actor(inputs)
            action_probs = F.softmax(logits, dim=-1)
        return action_probs

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_values(self, states: torch.Tensor) -> np.ndarray:
        # Return estimates of value function.
        with torch.no_grad():
            inputs = self._prepare_input(states)
            values = self._critic(inputs).squeeze(-1)
        return values

    # Serialization methods.
    def save_actor(self, path: str) -> None:
        torch.save(self._actor.state_dict(), path)

    def load_actor(self, path: str) -> None:
        self._actor.load_state_dict(torch.load(path, map_location=self.device))


def compute_gae(rewards, values, dones, gamma, trace_lambda, next_value=0):
    """Compute Generalized Advantage Estimation (GAE)"""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = delta + gamma * trace_lambda * (1 - dones[t]) * gae
        advantages[t] = gae
    
    returns = advantages + values
    return advantages, returns


def create_agents(env, args):
    """Create agents - either shared or independent networks"""
    if args.shared_network:
        # Create one shared agent that handles both agent positions
        shared_agent = Agent(env.observation_space, env.action_space[0], args, agent_id=0)
        agents = [shared_agent, shared_agent]  # Same agent used for both positions
    else:
        # Create independent agents
        agents = [Agent(env.observation_space, env.action_space[i], args, agent_id=i) 
                 for i in range(args.agents)]
    return agents


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Construct the agents
    agents = create_agents(env, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        state = env.reset(options={"start_evaluation": start_evaluation, "logging": logging})[0]
        rewards, done = 0, False
        while not done:
            # Predict a vector of actions using the greedy policy.
            actions = []
            for a in range(args.agents):
                action_probs = agents[a].predict_actions(np.expand_dims(state, 0))
                action = np.argmax(action_probs[0])
                actions.append(action)
            action = np.array(actions)
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards
    
        # If running in ReCodEx evaluation mode, load and evaluate only
    if args.recodex:
        if args.shared_network:
            agents[0].load_actor(args.model_path)
        else:
            saved = torch.load(args.model_path, map_location=Agent.device)
            for i, agent in enumerate(agents):
                agent._actor.load_state_dict(saved[f"agent_{i}"])
        while True:
            evaluate_episode(start_evaluation=True)
        return

    # Create an asynchronous vector environment for training.
    vector_env = gym.make_vec(env.spec.id, args.envs, gym.VectorizeMode.ASYNC, agents=args.agents,
                              vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})

    # Training
    state = vector_env.reset(seed=args.seed)[0]
    training, iteration = True, 0
    best_return = -float('inf')
    
    while training:
        # Collect experience
        states, actions, action_probs, rewards, dones, values = [], [], [], [], [], []
        
        for _ in range(args.worker_steps):
            # Choose actions for all agents
            action = np.zeros((args.envs, args.agents), dtype=np.int32)
            step_action_probs = np.zeros((args.envs, args.agents), dtype=np.float32)
            step_values = np.zeros((args.envs, args.agents), dtype=np.float32)
            
            for a in range(args.agents):
                # Get action probabilities and values for all environments
                agent_action_probs = agents[a].predict_actions(state)
                agent_values = agents[a].predict_values(state)
                
                # Sample actions from the policy
                for env_idx in range(args.envs):
                    action[env_idx, a] = np.random.choice(
                        len(agent_action_probs[env_idx]), 
                        p=agent_action_probs[env_idx]
                    )
                    step_action_probs[env_idx, a] = agent_action_probs[env_idx, action[env_idx, a]]
                
                step_values[:, a] = agent_values

            # Perform the step
            next_state, _, terminated, truncated, info = vector_env.step(action)
            reward = np.array([*info["agent_rewards"]], dtype=np.float32)
            done = terminated | truncated

            # Collect the required quantities
            states.append(state.copy())
            actions.append(action.copy())
            action_probs.append(step_action_probs.copy())
            rewards.append(reward.copy())
            dones.append(done.copy())
            values.append(step_values.copy())

            state = next_state

        # Convert to numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions)
        action_probs = np.array(action_probs, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones)
        values = np.array(values, dtype=np.float32)

        # Get next values for bootstrap
        next_values = np.zeros((args.envs, args.agents), dtype=np.float32)
        for a in range(args.agents):
            next_values[:, a] = agents[a].predict_values(state)

        # Train each agent (or shared agent once)
        agents_to_train = [0] if args.shared_network else list(range(args.agents))
        
        for a in agents_to_train:
            # Collect all experience for this agent
            all_advantages = []
            all_returns = []
            all_states = []
            all_actions = []
            all_action_probs = []
            
            # Process experience from both agent positions if using shared network
            agent_positions = range(args.agents) if args.shared_network else [a]
            
            for agent_pos in agent_positions:
                for env_idx in range(args.envs):
                    env_rewards = rewards[:, env_idx, agent_pos]
                    env_values = values[:, env_idx, agent_pos]
                    env_dones = dones[:, env_idx]
                    
                    # Bootstrap value
                    bootstrap_value = 0.0 if dones[-1, env_idx] else next_values[env_idx, agent_pos]
                    
                    advantages, returns = compute_gae(
                        env_rewards, env_values, env_dones, 
                        args.gamma, args.trace_lambda, bootstrap_value
                    )
                    
                    all_advantages.extend(advantages)
                    all_returns.extend(returns)
                    all_states.extend(states[:, env_idx])
                    all_actions.extend(actions[:, env_idx, agent_pos])
                    all_action_probs.extend(action_probs[:, env_idx, agent_pos])
            
            # Convert to numpy arrays
            all_advantages = np.array(all_advantages, dtype=np.float32)
            all_returns = np.array(all_returns, dtype=np.float32)
            all_states = np.array(all_states, dtype=np.float32)
            all_actions = np.array(all_actions)
            all_action_probs = np.array(all_action_probs, dtype=np.float32)
            
            # Normalize advantages
            all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

            # Create dataset and train
            dataset = TensorDataset(
                torch.FloatTensor(all_states),
                torch.LongTensor(all_actions),
                torch.FloatTensor(all_action_probs),
                torch.FloatTensor(all_advantages),
                torch.FloatTensor(all_returns)
            )
            
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            
            for epoch in range(args.epochs):
                for batch in dataloader:
                    batch_states, batch_actions, batch_action_probs, batch_advantages, batch_returns = batch
                    agents[a].train(batch_states, batch_actions, batch_action_probs, 
                                   batch_advantages, batch_returns)

        # Step learning rate schedulers
        for agent in set(agents):  # Use set to avoid stepping shared agent twice
            agent.step_schedulers()

        # Periodic evaluation
        iteration += 1
        if iteration % args.evaluate_each == 0:
            returns = [evaluate_episode() for _ in range(args.evaluate_for)]
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            print(f"Iteration {iteration}: Average return = {mean_return:.2f} ± {std_return:.2f}")
            
            # Track best performance
            if mean_return > best_return:
                best_return = mean_return
                # Save best model
                if args.shared_network:
                    agents[0].save_actor(args.model_path)
                else:
                    torch.save({f"agent_{i}": agents[i]._actor.state_dict() for i in range(args.agents)}, args.model_path)

            # Early stopping if we reach target performance
            if mean_return >= 450:
                print(f"Target return of 450 reached! Best: {best_return:.2f}")
                break  # Stop training

    #print(f"Training completed after {iteration} iterations. Best return: {best_return:.2f}")

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make("MultiCollect-v0", agents=main_args.agents), main_args.seed, main_args.render_each)

    main(main_env, main_args)