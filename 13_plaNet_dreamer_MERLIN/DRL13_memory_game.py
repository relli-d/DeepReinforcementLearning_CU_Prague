#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2425.13")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--cards", default=8, type=int, help="Number of cards in the memory game.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--batch_size", default=32, type=int, help="Number of episodes to train on.")
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate for number of episodes.")
parser.add_argument("--hidden_layer", default=None, type=int, help="Hidden layer size; default 8*`cards`")
parser.add_argument("--memory_cells", default=None, type=int, help="Number of memory cells; default 2*`cards`")
parser.add_argument("--memory_cell_size", default=None, type=int, help="Memory cell size; default 3/2*`cards`")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")


class Agent:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        self.args = args
        self.env = env

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Create suitable layers.
                input_size = sum(env.observation_space.nvec)
                
                # Layers for generating read key
                self.read_key_hidden = torch.nn.Linear(input_size, args.hidden_layer)
                self.read_key_output = torch.nn.Linear(args.hidden_layer, args.memory_cell_size)
                
                # Layers for policy generation
                policy_input_size = input_size + args.memory_cell_size
                self.policy_hidden = torch.nn.Linear(policy_input_size, args.hidden_layer)
                self.policy_output = torch.nn.Linear(args.hidden_layer, env.action_space.n)

            def forward(self, memory, state):
                # Encode the input state, which is a (card, observation) pair,
                # by representing each element as one-hot and concatenating them, resulting
                # in a vector of length `sum(env.observation_space.nvec)`.
                encoded_input = torch.cat([torch.nn.functional.one_hot(torch.relu(state[:, i]), dim).float()
                                           for i, dim in enumerate(env.observation_space.nvec)], dim=-1)

                # Generate a read key for memory read from the encoded input, by using
                # a ReLU hidden layer of size `args.hidden_layer` followed by a dense layer
                # with `args.memory_cell_size` units and `tanh` activation (to keep the memory
                # content in limited range).
                read_key = torch.relu(self.read_key_hidden(encoded_input))
                read_key = torch.tanh(self.read_key_output(read_key))

                # Read the memory using the generated read key. Notably, compute cosine
                # similarity of the key and every memory row, apply softmax to generate
                # a weight distribution over the rows, and finally take a weighted average of
                # the memory rows.
                
                # Ensure memory has batch dimension
                if len(memory.shape) == 2:
                    # Single memory case: add batch dimension
                    memory_batch = memory.unsqueeze(0)
                    batch_size = 1
                else:
                    # Already has batch dimension
                    memory_batch = memory
                    batch_size = memory.shape[0]
                
                # Ensure read_key has proper shape
                if len(read_key.shape) == 1:
                    read_key = read_key.unsqueeze(0)
                
                # Normalize for cosine similarity
                read_key_norm = torch.nn.functional.normalize(read_key.unsqueeze(1), p=2, dim=2)
                memory_norm = torch.nn.functional.normalize(memory_batch, p=2, dim=2)
                
                # Compute cosine similarity: [batch_size, memory_cells]
                similarities = torch.sum(read_key_norm * memory_norm, dim=2)
                
                # Apply softmax to get weights: [batch_size, memory_cells]
                weights = torch.softmax(similarities, dim=1)
                
                # Weighted average of memory rows: [batch_size, memory_cell_size]
                read_value = torch.sum(weights.unsqueeze(2) * memory_batch, dim=1)

                # Using concatenated encoded input and the read value, use a ReLU hidden
                # layer of size `args.hidden_layer` followed by a dense layer with
                # `env.action_space.n` units to produce policy logits.
                policy_input = torch.cat([encoded_input, read_value], dim=-1)
                policy_hidden = torch.relu(self.policy_hidden(policy_input))
                policy_logits = self.policy_output(policy_hidden)

                # Perform memory write. For faster convergence, append directly
                # the `encoded_input` to the memory, i.e., add it as a first memory row, and drop
                # the last memory row to keep memory size constant.
                new_row = encoded_input.unsqueeze(1)  # [batch_size, 1, memory_cell_size]
                new_memory_batch = torch.cat([new_row, memory_batch[:, :-1]], dim=1)
                
                # Return memory in same format as input
                if len(memory.shape) == 2:
                    new_memory = new_memory_batch.squeeze(0)
                else:
                    new_memory = new_memory_batch

                # Return the updated memory and the policy
                return new_memory, policy_logits

        # Create the agent
        self._model = Model().to(self.device)

        # Create an optimizer and a loss function.
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args.learning_rate)
        self._loss_fn = torch.nn.CrossEntropyLoss()

    def zero_memory(self):
        # Return an empty memory. It should be a tensor
        # with shape `[self.args.memory_cells, self.args.memory_cell_size]` on `self.device`.
        return torch.zeros(self.args.memory_cells, self.args.memory_cell_size, device=self.device)

    @npfl139.typed_torch_function(device, torch.int64, torch.int64, torch.int64)
    def _train(self, states, targets, lengths):
        # Given a batch of sequences of `states` (each being a (card, symbol) pair),
        # train the network to predict the required `targets`.
        #
        # Specifically, start with a batch of empty memories, and run the agent
        # sequentially as many times as necessary, using `targets` as gold labels.
        
        self._model.train()
        batch_size = states.shape[0]
        max_length = states.shape[1]
        
        # Initialize batch of empty memories
        memories = self.zero_memory().unsqueeze(0).repeat(batch_size, 1, 1)
        
        total_loss = 0.0
        num_predictions = 0
        
        # Process sequences step by step
        for t in range(max_length):
            # Get states and targets for this timestep
            current_states = states[:, t]
            current_targets = targets[:, t]
            
            # Create mask for valid entries (within episode length)
            mask = t < lengths
            
            if mask.any():
                # Forward pass
                memories, logits = self._model(memories, current_states)
                
                # Compute loss only for valid entries
                valid_logits = logits[mask]
                valid_targets = current_targets[mask]
                
                if valid_logits.shape[0] > 0:
                    loss = self._loss_fn(valid_logits, valid_targets)
                    total_loss += loss * valid_logits.shape[0]
                    num_predictions += valid_logits.shape[0]
        
        # Compute average loss and backpropagate
        if num_predictions > 0:
            avg_loss = total_loss / num_predictions
            self._optimizer.zero_grad()
            avg_loss.backward()
            self._optimizer.step()
            
            return avg_loss.item()
        return 0.0

    def train(self, episodes):
        # Given a list of episodes, prepare the arguments
        # of the self._train method, and execute it.
        
        # Find maximum episode length
        max_length = max(len(episode) - 1 for episode in episodes)  # -1 because last action is None
        batch_size = len(episodes)
        
        # Prepare padded states and targets
        states = np.zeros((batch_size, max_length, 2), dtype=np.int64)
        targets = np.zeros((batch_size, max_length), dtype=np.int64)
        lengths = np.zeros(batch_size, dtype=np.int64)
        
        for i, episode in enumerate(episodes):
            episode_length = len(episode) - 1  # Exclude last state with None action
            lengths[i] = episode_length
            
            for j in range(episode_length):
                state, action = episode[j]
                states[i, j] = state
                targets[i, j] = action
        
        # Train the model
        return self._train(states, targets, lengths)

    @npfl139.typed_torch_function(device, torch.float32, torch.int64)
    def predict(self, memory, state):
        self._model.eval()
        with torch.no_grad():
            memory, logits = self._model(memory, state)
            return memory, torch.softmax(logits, dim=-1)


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Post-process arguments to default values if not overridden on the command line.
    if args.hidden_layer is None:
        args.hidden_layer = 8 * args.cards
    if args.memory_cells is None:
        args.memory_cells = 2 * args.cards
    if args.memory_cell_size is None:
        args.memory_cell_size = 3 * args.cards // 2
    assert sum(env.observation_space.nvec) == args.memory_cell_size

    # Construct the agent.
    agent = Agent(env, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        state, memory = env.reset(start_evaluation=start_evaluation, logging=logging)[0], agent.zero_memory()
        rewards, done = 0, False
        while not done:
            # Find out which action to use.
            memory, action_probs = agent.predict(memory, np.array([state], dtype=np.int64))
            action = np.argmax(action_probs[0])
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Training
    training = True
    while training:
        # Generate required number of episodes
        for _ in range(args.evaluate_each // args.batch_size):
            episodes = []
            for _ in range(args.batch_size):
                episodes.append(env.expert_episode())

            # Train the agent
            agent.train(episodes)

        # Periodic evaluation
        returns = [evaluate_episode() for _ in range(args.evaluate_for)]
        
        # Check if we should continue training
        avg_return = np.mean(returns)
        if avg_return >= 3:
            print(f"Achieved average return: {avg_return:.2f}")
            training = False

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        gym.make("MemoryGame-v0", cards=main_args.cards), main_args.seed, main_args.render_each)

    main(main_env, main_args)