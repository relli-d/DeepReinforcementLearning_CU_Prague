#!/usr/bin/env python3
import argparse
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import npfl139
npfl139.require_version("2425.11")
from npfl139.board_games import AZQuiz

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.3, type=float, help="MCTS root Dirichlet alpha")
parser.add_argument("--batch_size", default=128, type=int, help="Number of game positions to train on.")
parser.add_argument("--epsilon", default=0.25, type=float, help="MCTS exploration epsilon in root")
parser.add_argument("--evaluate_each", default=1, type=int, help="Evaluate each number of iterations.")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="az_quiz1.pt", type=str, help="Model path")
parser.add_argument("--num_simulations", default=64, type=int, help="Number of simulations in one MCTS.")
parser.add_argument("--replay_buffer_length", default=100000, type=int, help="Replay buffer max length.")
parser.add_argument("--sampling_moves", default=20, type=int, help="Sampling moves.")
parser.add_argument("--show_sim_games", default=False, action="store_true", help="Show simulated games.")
parser.add_argument("--sim_games", default=10, type=int, help="Simulated games to generate in every iteration.")
parser.add_argument("--train_for", default=5, type=int, help="Update steps in every iteration.")


#########
# Agent #
#########
class Agent:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, args: argparse.Namespace):
        # Define an agent network in `self._model`.
        # Architecture: 5 conv layers + policy head + value head
        class AZQuizNet(nn.Module):
            def __init__(self):
                super().__init__()

                # Reduced channel counts
                self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

                # Policy head
                self.policy_conv = nn.Conv2d(32, 8, kernel_size=3, padding=1)  # Reduced from 16
                self.policy_fc1 = nn.Linear(8 * 7 * 7, 64)  # Reduced from 128
                self.policy_fc2 = nn.Linear(64, 28)  # Output remains the same

                # Value head
                self.value_conv = nn.Conv2d(32, 16, kernel_size=3, padding=1)
                self.value_fc1 = nn.Linear(16 * 7 * 7, 128)
                self.value_fc2 = nn.Linear(128, 1)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))

                p = F.relu(self.policy_conv(x))
                p = p.flatten(start_dim=1)
                p = F.relu(self.policy_fc1(p))
                p = self.policy_fc2(p)

                v = F.relu(self.value_conv(x))
                v = v.flatten(start_dim=1)
                v = F.relu(self.value_fc1(v))
                v = torch.tanh(self.value_fc2(v))

                return p, v.squeeze(-1)
        
        
        self._model = AZQuizNet().to(self.device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size=100, gamma=0.1)

    @classmethod
    def load(cls, path: str, args: argparse.Namespace) -> "Agent":
        # A static method returning a new Agent loaded from the given path.
        agent = Agent(args)
        agent._model.load_state_dict(torch.load(path, map_location=agent.device))
        return agent

    def save(self, path: str) -> None:
        torch.save(self._model.state_dict(), path)

    @npfl139.typed_torch_function(device, torch.float32, torch.float32, torch.float32)
    def train(self, boards: torch.Tensor, target_policies: torch.Tensor, target_values: torch.Tensor) -> None:
        # Train the model based on given boards, target policies and target values.
        self._model.train()
        self._optimizer.zero_grad()
        
        policy_logits, predicted_values = self._model(boards)
        
        # Policy loss - use target_policies as probabilities, not class indices
        policy_loss = -torch.sum(target_policies * F.log_softmax(policy_logits, dim=-1), dim=-1).mean()
        
        # Value loss (MSE)
        value_loss = F.mse_loss(predicted_values, target_values)
        
        # Total loss
        total_loss = policy_loss + value_loss
        total_loss.backward()
        self._optimizer.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, boards: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        # Disable gradients and set model to eval mode
        with torch.no_grad():
            self._model.eval()
            policy_logits, values = self._model(boards)
            policies = F.softmax(policy_logits, dim=-1)
        
        # Return numpy arrays directly (no .cpu() if already on CPU)
        return policies.numpy(), values.numpy()

    def board_features(self, game: AZQuiz) -> np.ndarray:
        # 4 channels: player 0, player 1, current player, move count
        features = np.zeros((4, 7, 7), dtype=np.float32)
        
        # Player positions
        if game.board.ndim == 3:
            features[0] = game.board[:, :, 0]
            features[1] = game.board[:, :, 1]
        else:
            features[0] = (game.board == 1).astype(np.float32)
            features[1] = (game.board == 2).astype(np.float32)
        
        # Current player indicator
        features[2] = float(game.to_play)
        
        # Move count (normalized)
        features[3] = game.board.sum() / (7*7)  # Normalized count of moves
        
        return features


########
# MCTS #
########
class MCTNode:
    def __init__(self, prior: float | None):
        self.prior = prior  # Prior probability from the agent.
        self.game = None    # If the node is evaluated, the corresponding game instance.
        self.children = {}  # If the node is evaluated, mapping of valid actions to the child `MCTNode`s.
        self.visit_count = 0
        self.total_value = 0

    def value(self) -> float:
        # Return the value of the current node, handling the case when `self.visit_count` is 0.
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def is_evaluated(self) -> bool:
        # A node is evaluated if it has non-zero `self.visit_count`.
        # In such case `self.game` is not None.
        return self.visit_count > 0

    def evaluate(self, game: AZQuiz, agent: Agent) -> None:
        # Each node can be evaluated at most once
        assert self.game is None
        self.game = game

        # Compute the value of the current game.
        if game.outcome(game.to_play) is not None:
            # Game has ended, compute value directly
            outcome = game.outcome(game.to_play)
            if outcome == 0:  # Draw
                value = 0.0
            elif outcome == game.to_play + 1:  # Current player wins
                value = 1.0
            else:  # Current player loses
                value = -1.0
        else:
            # Use the agent to evaluate the current game
            board_features = agent.board_features(game)
            board_tensor = torch.from_numpy(board_features).unsqueeze(0).to(agent.device)
            policy, values = agent.predict(board_tensor)
            value = values[0]
            
            # Populate children with priors from the policy
            for action in game.valid_actions():
                self.children[action] = MCTNode(policy[0][action])

        self.total_value = value
        self.visit_count = 1

    def add_exploration_noise(self, epsilon: float, alpha: float) -> None:
        # Update the children priors by exploration noise Dirichlet(alpha)
        if not self.children:
            return
            
        actions = list(self.children.keys())
        noise = np.random.dirichlet([alpha] * len(actions))
        
        for i, action in enumerate(actions):
            old_prior = self.children[action].prior
            self.children[action].prior = epsilon * noise[i] + (1 - epsilon) * old_prior

    def select_child(self) -> tuple[int, "MCTNode"]:
        # Select a child according to the PUCT formula.
        def ucb_score(action_child_pair):
            action, child = action_child_pair
            # Q(s, a) - value from child's perspective (need to invert if different player)
            q_value = -child.value()  # Invert because it's from opponent's perspective
            
            # C(s) as defined in AlphaZero
            # c_puct = np.log((1 + self.visit_count + 19652) / 19652) + 1.25
            c_puct = 1.0  # Reduced from 1.25 for more exploration

            
            # P(s, a) - prior probability
            prior = child.prior
            
            # UCB score
            exploration = c_puct * prior * (np.sqrt(self.visit_count) / (child.visit_count + 1))
            
            return q_value + exploration

        # Return the (action, child) pair with the highest `ucb_score`.
        return max(self.children.items(), key=ucb_score)


def mcts(game: AZQuiz, agent: Agent, args: argparse.Namespace, explore: bool) -> np.ndarray:
    root = MCTNode(None)
    root.evaluate(game, agent)
    if explore:
        root.add_exploration_noise(args.epsilon, args.alpha)

    for _ in range(args.num_simulations):
        node = root
        path = [node]
        game_path = [game.clone()]
        
        # Selection phase - traverse down the tree
        while node.children and node.is_evaluated():
            action, node = node.select_child()
            path.append(node)
            # Apply the action to get the next game state
            current_game = game_path[-1].clone()
            current_game.move(action)
            game_path.append(current_game)

        # Expansion and evaluation phase
        if not node.is_evaluated():
            node.evaluate(game_path[-1], agent)

        # Backup phase - propagate values up the tree
        value = node.value()
        for i, path_node in enumerate(reversed(path)):
            # Alternate value sign for different players
            current_value = value if i % 2 == 0 else -value
            path_node.visit_count += 1
            path_node.total_value += current_value

    # Return policy based on visit counts
    policy = np.zeros(28, dtype=np.float32)
    if root.children:
        total_visits = sum(child.visit_count for child in root.children.values())
        for action, child in root.children.items():
            policy[action] = child.visit_count / total_visits if total_visits > 0 else 0

    return policy


############
# Training #
############
ReplayBufferEntry = collections.namedtuple("ReplayBufferEntry", ["board", "policy", "outcome"])


def sim_game(agent: Agent, args: argparse.Namespace) -> list[ReplayBufferEntry]:
    # Simulate a game, return a list of `ReplayBufferEntry`s.
    game = AZQuiz()
    replay_buffer_entries = []
    
    # Keep track of boards and policies during the game
    boards = []
    policies = []
    
    while not game.outcome(game.to_play):
        # Run the MCTS with exploration
        policy = mcts(game, agent, args, explore=True)
        
        # Save current state for the replay buffer
        boards.append(agent.board_features(game))
        policies.append(policy)
        
        # Select an action, either by sampling or greedily
        valid_actions = game.valid_actions()
        if len(valid_actions) == 0:
            break
            
        if game.board.sum() < args.sampling_moves:
            # Sample an action from the policy
            # Create a probability distribution over valid actions
            valid_policy = np.array([policy[a] for a in valid_actions])
            valid_policy /= valid_policy.sum() + 1e-8  # Normalize
            action_idx = np.random.choice(len(valid_actions), p=valid_policy)
            action = valid_actions[action_idx]
        else:
            # Choose the action with highest probability
            action = valid_actions[np.argmax([policy[a] for a in valid_actions])]
        
        # Make the move
        game.move(action)
    
    # Determine the final outcome for each saved state
    outcome = game.outcome(0)  # Outcome for player 0
    if outcome == AZQuiz.Outcome.WIN:
        final_value = 1.0
    elif outcome == AZQuiz.Outcome.LOSS:
        final_value = -1.0
    else:
        final_value = 0.0  # Draw or unfinished (shouldn't happen)
    
    # Create replay buffer entries with correct outcome values for each player
    for i, (board, policy) in enumerate(zip(boards, policies)):
        # The value depends on which player's turn it was
        # If i is even, it was player 0's turn, otherwise player 1's turn
        player_value = final_value if i % 2 == 0 else -final_value
        replay_buffer_entries.append(ReplayBufferEntry(board, policy, player_value))
    
    return replay_buffer_entries


def train(args: argparse.Namespace) -> Agent:
    agent = Agent(args)
    replay_buffer = npfl139.MonolithicReplayBuffer(max_length=args.replay_buffer_length)
    
    # Warmup phase - fill buffer before training
    while len(replay_buffer) < args.batch_size * 10:
        game = sim_game(agent, args)
        replay_buffer.extend(game)
    
    iteration = 0
    best_score = 0
    while True:
        iteration += 1
        
        # Generate more games as training progresses
        games_to_generate = min(args.sim_games + iteration // 10, 50)
        for _ in range(games_to_generate):
            replay_buffer.extend(sim_game(agent, args))
        
        # Train in larger batches as we progress
        train_steps = min(args.train_for + iteration // 20, 20)
        for _ in range(train_steps):
            if len(replay_buffer) < args.batch_size:
                continue
                
            batch = replay_buffer.sample(args.batch_size)
            boards = torch.tensor(batch.board, dtype=torch.float32)
            policies = torch.tensor(batch.policy, dtype=torch.float32)
            values = torch.tensor(batch.outcome, dtype=torch.float32)
            
            agent.train(boards, policies, values)
        
        # Evaluation
        if iteration % args.evaluate_each == 0:
            score = npfl139.board_games.evaluate(
                AZQuiz, 
                [Player(agent, argparse.Namespace(num_simulations=args.num_simulations)),
                 AZQuiz.player_from_name("simple_heuristic")(seed=args.seed)],
                games=56, first_chosen=False, render=False, verbose=False,
            )
            
            if score > best_score:
                best_score = score
                agent.save(args.model_path)
                print("New best score: {:.1f}%".format(100 * score))
            
            if score >= 0.95:
                return agent


#############################
# BoardGamePlayer interface #
#############################
class Player(npfl139.board_games.BoardGamePlayer[AZQuiz]):
    def __init__(self, agent: Agent, args: argparse.Namespace):
        self.agent = agent
        self.args = args

    def play(self, game: AZQuiz) -> int:
        # Skip MCTS entirely if num_simulations is 0
        board_features = self.agent.board_features(game)
        board_tensor = torch.from_numpy(board_features).unsqueeze(0).to(self.agent.device)
        policy, _ = self.agent.predict(board_tensor)
        policy = policy[0]  # Remove batch dimension
        
        # Select the best valid action
        valid_actions = game.valid_actions()
        return max(valid_actions, key=lambda a: policy[a])


########
# Main #
########
def main(args: argparse.Namespace) -> Player:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    if args.recodex:
        # Load the trained agent
        agent = Agent.load(args.model_path, args)
    else:
        # Perform training
        agent = train(args)

    return Player(agent, args)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    player = main(main_args)

    # Run an evaluation versus the simple heuristic with the same parameters as in ReCodEx.
    npfl139.board_games.evaluate(
        AZQuiz, [player, AZQuiz.player_from_name("simple_heuristic")(seed=main_args.seed)],
        games=56, first_chosen=False, render=False, verbose=True,
    )