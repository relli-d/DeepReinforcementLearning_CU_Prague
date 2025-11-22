#!/usr/bin/env python3
import argparse
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

import npfl139
npfl139.require_version("2425.13")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--cards", default=6, type=int, help="Number of cards in the memory game.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--batch_size", default=32, type=int, help="Number of episodes to train on.")
parser.add_argument("--gradient_clipping", default=1.0, type=float, help="Gradient clipping.")
parser.add_argument("--entropy_regularization", default=0.1, type=float, help="Entropy regularization weight.")
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate for number of episodes.")
parser.add_argument("--hidden_layer", default=None, type=int, help="Hidden layer size; default 8*`cards`")
parser.add_argument("--memory_cells", default=None, type=int, help="Number of memory cells; default 2*`cards`")
parser.add_argument("--memory_cell_size", default=None, type=int, help="Memory cell size; default 3/2*`cards`")
parser.add_argument("--replay_buffer", default=None, type=int, help="Max replay buffer size; default batch_size")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--expert_episodes", default=0.2, type=float, help="Fraction of expert episodes for pretraining")

class Agent:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        self.args = args
        self.env = env
        
        # For baseline tracking
        self.baseline = 0.0
        self.baseline_momentum = 0.01
        self.episode_count = 0

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                
                # Input encoding size
                self.input_size = sum(env.observation_space.nvec)
                
                # Read key generation layers
                self.read_key_hidden = torch.nn.Linear(self.input_size, args.hidden_layer)
                self.read_key_output = torch.nn.Linear(args.hidden_layer, args.memory_cell_size)
                
                # Policy layers
                self.policy_hidden = torch.nn.Linear(self.input_size + args.memory_cell_size, args.hidden_layer)
                self.policy_output = torch.nn.Linear(args.hidden_layer, env.action_space.n)

            def forward(self, memory, state):
                # Improved one-hot encoding with proper shape handling
                batch_size = state.shape[0]
                encoded_input = torch.cat([torch.nn.functional.one_hot(torch.clamp(state[:, i], 0, dim-1), dim).float()
                                           for i, dim in enumerate(env.observation_space.nvec)], dim=-1)

                # Generate read key with better activation
                read_key = torch.relu(self.read_key_hidden(encoded_input))
                read_key = torch.tanh(self.read_key_output(read_key))

                # Improved memory read with better batch handling
                # Ensure memory has batch dimension
                if len(memory.shape) == 2:
                    memory_batch = memory.unsqueeze(0).expand(batch_size, -1, -1)
                else:
                    memory_batch = memory
                
                # More stable cosine similarity computation
                eps = 1e-8
                read_key_norm = F.normalize(read_key.unsqueeze(1) + eps, p=2, dim=2)
                memory_norm = F.normalize(memory_batch + eps, p=2, dim=2)
                
                # Compute similarities more efficiently
                similarities = torch.sum(read_key_norm * memory_norm, dim=2)
                
                # Apply softmax with temperature for better exploration
                temperature = 1.0
                weights = F.softmax(similarities / temperature, dim=1)
                
                # Weighted average
                read_value = torch.sum(weights.unsqueeze(2) * memory_batch, dim=1)

                # Policy computation
                policy_input = torch.cat([encoded_input, read_value], dim=-1)
                policy_hidden = torch.relu(self.policy_hidden(policy_input))
                policy_logits = self.policy_output(policy_hidden)

                # Improved memory write with better handling
                new_row = encoded_input.unsqueeze(1)
                if len(memory.shape) == 2:
                    # Single memory case
                    new_memory = torch.cat([new_row.squeeze(0), memory[:-1]], dim=0)
                else:
                    # Batch memory case
                    new_memory = torch.cat([new_row, memory_batch[:, :-1]], dim=1)

                return new_memory, policy_logits

        # Create the agent
        self._model = Model().to(self.device)

        # Create optimizer
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args.learning_rate)
        
        # For supervised learning from expert episodes
        self._supervised_loss_fn = torch.nn.CrossEntropyLoss()

    def zero_memory(self):
        return torch.zeros(self.args.memory_cells, self.args.memory_cell_size, device=self.device)

    @npfl139.typed_torch_function(device, torch.int64, torch.int64, torch.float32)
    def _train(self, states, actions, returns):
        # Improved REINFORCE training with better batching
        self._model.train()
        batch_size, max_length = states.shape[:2]
        
        # Initialize memories for the batch
        memory = self.zero_memory().unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        
        total_loss = 0.0
        valid_steps = 0
        
        for t in range(max_length):
            current_states = states[:, t]
            current_actions = actions[:, t]
            current_returns = returns[:, t]
            
            # Active mask for valid episodes
            active_mask = (current_states[:, 0] >= 0)
            
            if not active_mask.any():
                break
            
            # Forward pass for all episodes (including inactive for memory consistency)
            memory, logits = self._model(memory, current_states)
            
            # Only compute loss for active episodes
            if active_mask.any():
                active_logits = logits[active_mask]
                active_actions = current_actions[active_mask]
                active_returns = current_returns[active_mask]
                
                # Compute policy probabilities
                log_probs = F.log_softmax(active_logits, dim=-1)
                action_log_probs = log_probs.gather(1, active_actions.unsqueeze(1)).squeeze(1)
                
                # Update baseline more efficiently
                if len(active_returns) > 0:
                    current_return_mean = active_returns.mean().item()
                    self.baseline = (1 - self.baseline_momentum) * self.baseline + self.baseline_momentum * current_return_mean
                
                # Compute advantages
                advantages = active_returns - self.baseline
                
                # REINFORCE loss
                policy_loss = -(action_log_probs * advantages).mean()
                
                # Entropy regularization
                probs = F.softmax(active_logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1)
                entropy_loss = -self.args.entropy_regularization * entropy.mean()
                
                step_loss = policy_loss + entropy_loss
                total_loss += step_loss
                valid_steps += 1
        
        if valid_steps > 0:
            avg_loss = total_loss / valid_steps
            
            # Backpropagation
            self._optimizer.zero_grad()
            avg_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.args.gradient_clipping)
            
            self._optimizer.step()

    @npfl139.typed_torch_function(device, torch.int64, torch.int64, torch.int64)
    def _train_supervised(self, states, targets, lengths):
        # Supervised training from expert episodes (borrowed from second version)
        self._model.train()
        batch_size = states.shape[0]
        max_length = states.shape[1]
        
        # Initialize batch of empty memories
        memories = self.zero_memory().unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        
        total_loss = 0.0
        num_predictions = 0
        
        # Process sequences step by step
        for t in range(max_length):
            current_states = states[:, t]
            current_targets = targets[:, t]
            
            # Create mask for valid entries
            mask = t < lengths
            
            if mask.any():
                # Forward pass
                memories, logits = self._model(memories, current_states)
                
                # Compute loss only for valid entries
                valid_logits = logits[mask]
                valid_targets = current_targets[mask]
                
                if valid_logits.shape[0] > 0:
                    loss = self._supervised_loss_fn(valid_logits, valid_targets)
                    total_loss += loss * valid_logits.shape[0]
                    num_predictions += valid_logits.shape[0]
        
        # Compute average loss and backpropagate
        if num_predictions > 0:
            avg_loss = total_loss / num_predictions
            self._optimizer.zero_grad()
            avg_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.args.gradient_clipping)
            
            self._optimizer.step()

    def train(self, episodes):
        # Original REINFORCE training
        if not episodes:
            return
            
        # Find maximum episode length
        max_length = max(len(episode) for episode in episodes)
        batch_size = len(episodes)
        
        # Prepare tensors
        states = torch.full((batch_size, max_length, 2), -1, dtype=torch.int64, device=self.device)
        actions = torch.full((batch_size, max_length), -1, dtype=torch.int64, device=self.device)
        returns = torch.zeros((batch_size, max_length), dtype=torch.float32, device=self.device)
        
        # Fill tensors with episode data
        for i, episode in enumerate(episodes):
            episode_length = len(episode)
            for t, (state, action, reward) in enumerate(episode):
                states[i, t] = torch.tensor(np.array(state), dtype=torch.int64, device=self.device)
                actions[i, t] = action
                returns[i, t] = reward
        
        self._train(states, actions, returns)

    def train_supervised(self, episodes):
        # Supervised training from expert episodes
        if not episodes:
            return
            
        # Find maximum episode length
        max_length = max(len(episode) - 1 for episode in episodes)  # -1 because last action is None
        batch_size = len(episodes)
        
        # Prepare padded states and targets
        states = torch.full((batch_size, max_length, 2), -1, dtype=torch.int64, device=self.device)
        targets = torch.full((batch_size, max_length), -1, dtype=torch.int64, device=self.device)
        lengths = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
        
        for i, episode in enumerate(episodes):
            episode_length = len(episode) - 1  # Exclude last state with None action
            lengths[i] = episode_length
            
            for j in range(episode_length):
                state, action = episode[j]
                states[i, j] = torch.tensor(np.array(state), dtype=torch.int64, device=self.device)
                targets[i, j] = action
        
        self._train_supervised(states, targets, lengths)

    def predict(self, memory, state):
        self._model.eval()
        with torch.no_grad():
            memory, logits = self._model(memory, state)
            return memory, torch.softmax(logits, dim=-1)


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()

    # Post-process arguments to default values if not overridden on the command line.
    if args.hidden_layer is None:
        args.hidden_layer = 8 * args.cards
    if args.memory_cells is None:
        args.memory_cells = 2 * args.cards
    if args.memory_cell_size is None:
        args.memory_cell_size = 3 * args.cards // 2
    if args.replay_buffer is None:
        args.replay_buffer = args.batch_size
    assert sum(env.observation_space.nvec) == args.memory_cell_size

    # Construct the agent
    agent = Agent(env, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        state, memory = env.reset(start_evaluation=start_evaluation, logging=logging)[0], agent.zero_memory()
        rewards, done = 0, False
        while not done:
            # Find out which action to use
            state_tensor = torch.tensor(np.array([state]), dtype=torch.int64, device=agent.device)
            memory, action_probs = agent.predict(memory, state_tensor)
            action = torch.multinomial(action_probs, 1).item()
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Pretraining with expert episodes
    expert_episodes_count = int(args.evaluate_each * args.expert_episodes)
    if expert_episodes_count > 0:
        print(f"Pretraining with {expert_episodes_count} expert episodes...")
        for _ in range(expert_episodes_count // args.batch_size + 1):
            expert_episodes = []
            for _ in range(min(args.batch_size, expert_episodes_count)):
                expert_episodes.append(env.expert_episode())
            if expert_episodes:
                agent.train_supervised(expert_episodes)

    # Training with improved experience replay
    replay_buffer = npfl139.ReplayBuffer(max_length=args.replay_buffer)
    training = True
    
    while training:
        # Generate required number of episodes
        for _ in range(args.evaluate_each):
            state, memory, episode, done = env.reset()[0], agent.zero_memory(), [], False
            while not done:
                # Choose an action according to the generated distribution
                state_tensor = torch.tensor(np.array([state]), dtype=torch.int64, device=agent.device)
                memory, action_probs = agent.predict(memory, state_tensor)
                action = torch.multinomial(action_probs, 1).item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode.append([state, action, reward])
                state = next_state

            # Compute returns from rewards
            returns = []
            cumulative_return = 0
            for state, action, reward in reversed(episode):
                cumulative_return += reward
                returns.append(cumulative_return)
            returns.reverse()
            
            # Update episode with returns
            for i, (state, action, reward) in enumerate(episode):
                episode[i] = [state, action, returns[i]]

            replay_buffer.append(episode)
            agent.episode_count += 1

            # Train the network if enough data is available
            if len(replay_buffer) >= args.batch_size:
                agent.train(replay_buffer.sample(args.batch_size, np.random, replace=False))

        # Periodic evaluation
        returns = [evaluate_episode() for _ in range(args.evaluate_for)]
        
        # Check if we should stop training
        mean_return = np.mean(returns)
        print(f"Episode {agent.episode_count}: Mean return = {mean_return:.3f}")
        
        if mean_return >= 2.0:
            training = False
            print(f"Training stopped - achieved target return of {mean_return:.2f}")
            break

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        gym.make("MemoryGame-v0", cards=main_args.cards), main_args.seed, main_args.render_each)

    main(main_env, main_args)