#!/usr/bin/env python3
import argparse
import collections
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import npfl139
npfl139.require_version("2425.11")
from npfl139.board_games import AZQuizRandomized

parser = argparse.ArgumentParser()
parser.add_argument("--recodex", default=False, action="store_true")
parser.add_argument("--render_each", default=0, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--threads", default=1, type=int)
parser.add_argument("--alpha", default=0.3, type=float)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--epsilon", default=0.25, type=float)
parser.add_argument("--evaluate_each", default=1, type=int)
parser.add_argument("--learning_rate", default=1e-3, type=float)
parser.add_argument("--model_path", default="az_quiz_randomized.pt", type=str)
parser.add_argument("--num_simulations", default=400, type=int)  # Increased for stochasticity
parser.add_argument("--replay_buffer_length", default=100000, type=int)
parser.add_argument("--sampling_moves", default=5, type=int)  # Reduced since environment is stochastic
parser.add_argument("--show_sim_games", default=False, action="store_true")
parser.add_argument("--sim_games", default=1, type=int)  # Reduced
parser.add_argument("--train_for", default=1, type=int)  # Reduced
# Debug arguments
parser.add_argument("--debug_mcts", default=False, action="store_true", help="Run MCTS debugging with blank agent")
parser.add_argument("--debug_sim_counts", default="0,10,20,50,100", type=str, help="Comma-separated list of simulation counts for debugging")


class Agent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, args: argparse.Namespace):
        class AZQuizNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(5, 64, kernel_size=3, padding=1)  # Changed from 4 to 5 channels
                self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
                self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

                self.policy_conv = nn.Conv2d(32, 16, kernel_size=3, padding=1)
                self.policy_fc1 = nn.Linear(16 * 7 * 7, 128)
                self.policy_fc2 = nn.Linear(128, 64)
                self.policy_fc3 = nn.Linear(64, 28)

                self.value_conv = nn.Conv2d(32, 16, kernel_size=3, padding=1)
                self.value_fc1 = nn.Linear(16 * 7 * 7, 128)
                self.value_fc2 = nn.Linear(128, 1)

                self.dropout = nn.Dropout(0.1)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
                x = F.relu(self.conv4(x))

                # Policy head
                p = F.relu(self.policy_conv(x))
                p = p.flatten(start_dim=1)
                p = F.relu(self.policy_fc1(p))
                p = self.dropout(p)
                p = F.relu(self.policy_fc2(p))
                p = self.policy_fc3(p)

                # Value head
                v = F.relu(self.value_conv(x))
                v = v.flatten(start_dim=1)
                v = F.relu(self.value_fc1(v))
                v = torch.tanh(self.value_fc2(v))

                return p, v.squeeze(-1)
        
        self._model = AZQuizNet().to(self.device)
        # Constant learning rate - no weight decay or scheduler
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args.learning_rate)

    @classmethod
    def load(cls, path: str, args: argparse.Namespace) -> "Agent":
        agent = Agent(args)
        agent._model.load_state_dict(torch.load(path, map_location=agent.device))
        return agent

    def save(self, path: str) -> None:
        torch.save(self._model.state_dict(), path)

    @npfl139.typed_torch_function(device, torch.float32, torch.float32, torch.float32)
    def train(self, boards: torch.Tensor, target_policies: torch.Tensor, target_values: torch.Tensor) -> None:
        self._model.train()
        self._optimizer.zero_grad()
        
        policy_logits, predicted_values = self._model(boards)
        
        policy_loss = -torch.sum(target_policies * F.log_softmax(policy_logits, dim=-1), dim=-1).mean()
        value_loss = F.smooth_l1_loss(predicted_values, target_values)
        total_loss = 1.5 * policy_loss + value_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
        self._optimizer.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, boards: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            self._model.eval()
            policy_logits, values = self._model(boards)
            policies = F.softmax(policy_logits, dim=-1)
        return policies.numpy(), values.numpy()

    def board_features(self, game: AZQuizRandomized) -> np.ndarray:
        # Use the game's board_features as the foundation
        base_features = game.board_features  # Shape: (7, 7, 3) - it's a property, not a method
        
        # Create 5-channel feature tensor
        features = np.zeros((5, 7, 7), dtype=np.float32)
        
        # First two channels: current player and opponent from game.board_features
        features[0] = base_features[:, :, 0]  # Current player (to_play)
        features[1] = base_features[:, :, 1]  # Opponent player
        
        # Third channel: failed attempts (from game.board_features)
        features[2] = base_features[:, :, 2]  # Places where someone played but failed
        
        # Fourth channel: current player indicator
        features[3].fill(float(game.to_play))
        
        # Fifth channel: game progress (normalized move count)
        features[4].fill(game.board.sum() / 49.0)
        
        return features


class BlankAgent(Agent):
    """Blank agent that returns uniform policy and draw value for debugging"""
    
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        # Override the model to return uniform policy and draw value
        
    @npfl139.typed_torch_function(Agent.device, torch.float32)
    def predict(self, boards: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        batch_size = boards.shape[0]
        # Uniform policy over all 28 actions
        policies = np.ones((batch_size, 28), dtype=np.float32) / 28.0
        # Draw value (0.0) for all boards
        values = np.zeros(batch_size, dtype=np.float32)
        return policies, values


class MCTNode:
    def __init__(self, prior: float | None, player: int):
        self.prior = prior
        self.player = player  # This should be the player who will evaluate this node
        self.game = None
        self.children = {}
        self.visit_count = 0
        self.total_value = 0

    def value(self) -> float:
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0

    def is_evaluated(self) -> bool:
        return self.visit_count > 0

    def evaluate(self, game: AZQuizRandomized, agent: Agent) -> None:
        assert self.game is None
        self.game = game

        # Evaluate from the perspective of self.player
        outcome = game.outcome(self.player)
        if outcome is not None:
            # Use proper outcome constants
            if outcome == AZQuizRandomized.Outcome.WIN:
                value = 1.0
            elif outcome == AZQuizRandomized.Outcome.LOSS:
                value = -1.0
            else:  # DRAW
                value = 0.0
        else:
            board_features = agent.board_features(game)
            board_tensor = torch.from_numpy(board_features).unsqueeze(0).to(agent.device)
            policy, values = agent.predict(board_tensor)
            
            # The network evaluates from the perspective of game.to_play
            # We need to convert to self.player's perspective
            if game.to_play == self.player:
                value = values[0]
            else:
                value = -values[0]
            
            self._policy = policy[0]

        self.total_value = value
        self.visit_count = 1

    def add_exploration_noise(self, epsilon: float, alpha: float) -> None:
        if not hasattr(self, '_policy'):
            return
            
        valid_actions = self.game.valid_actions()
        if len(valid_actions) == 0:
            return
            
        noise = np.random.dirichlet([alpha] * len(valid_actions))
        
        for i, action in enumerate(valid_actions):
            old_prior = self._policy[action]
            self._policy[action] = epsilon * noise[i] + (1 - epsilon) * old_prior


class RandomNode:
    """Explicit random node to handle stochastic outcomes"""
    def __init__(self, action: int, parent_player: int):
        self.action = action
        self.parent_player = parent_player
        self.children = {}  # outcome_key -> MCTNode
        self.visit_count = 0
        self.total_value = 0
        self.outcomes = None  # Will store all possible outcomes

    def value(self) -> float:
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0

    def initialize_outcomes(self, game: AZQuizRandomized):
        """Initialize all possible outcomes for this action"""
        if self.outcomes is None:
            self.outcomes = game.all_moves(self.action)
            for probability, outcome_game in self.outcomes:
                outcome_key = f"{outcome_game.board.tobytes()}_{outcome_game.to_play}"
                if outcome_key not in self.children:
                    # The child node should be evaluated from the perspective of the parent player
                    self.children[outcome_key] = MCTNode(None, self.parent_player)

    def select_outcome(self, game: AZQuizRandomized) -> tuple[MCTNode, AZQuizRandomized]:
        """Select one of the possible outcomes based on the game's randomness"""
        self.initialize_outcomes(game)
        
        # Make the move and get the actual outcome
        game_copy = game.clone()
        game_copy.move(self.action)
        outcome_key = f"{game_copy.board.tobytes()}_{game_copy.to_play}"
        
        return self.children[outcome_key], game_copy


def mcts(game: AZQuizRandomized, agent: Agent, args: argparse.Namespace, explore: bool) -> np.ndarray:
    root = MCTNode(None, game.to_play)
    root.evaluate(game, agent)
    if explore:
        root.add_exploration_noise(args.epsilon, args.alpha)

    for _ in range(args.num_simulations):
        current_game = game.clone()
        node = root
        path = [(node, current_game.to_play)]  # Store (node, player) pairs
        random_nodes = []  # Track random nodes in path
        
        # Selection and expansion
        while node.is_evaluated() and current_game.outcome(current_game.to_play) is None:
            valid_actions = current_game.valid_actions()
            if len(valid_actions) == 0:
                break
                
            def ucb_score(action):
                if action not in node.children:
                    # Create random node if it doesn't exist
                    node.children[action] = RandomNode(action, node.player)
                
                random_node = node.children[action]
                random_node.initialize_outcomes(current_game)
                
                # Calculate visits for this action across all outcomes
                action_visits = random_node.visit_count
                
                prior = node._policy[action] if hasattr(node, '_policy') else 1.0 / len(valid_actions)
                c_puct = 1.0
                exploration = c_puct * prior * (np.sqrt(node.visit_count) / (action_visits + 1))
                
                q_value = -random_node.value() if action_visits > 0 else 0.0
                
                return q_value + exploration
            
            selected_action = max(valid_actions, key=ucb_score)
            random_node = node.children[selected_action]
            random_nodes.append(random_node)
            
            # Select outcome from random node
            node, current_game = random_node.select_outcome(current_game)
            path.append((node, node.player))

        # Evaluation
        if not node.is_evaluated():
            node.evaluate(current_game, agent)

        # Backup - use consistent player perspective
        leaf_value = node.value()
        for i, (path_node, player) in enumerate(reversed(path)):
            # The leaf value is already from the perspective of the leaf node's player
            # Convert to current path node's player perspective
            if player == node.player:
                backup_value = leaf_value
            else:
                backup_value = -leaf_value
                
            path_node.visit_count += 1
            path_node.total_value += backup_value
        
        # Update random nodes
        for random_node in reversed(random_nodes):
            random_node.visit_count += 1
            # Random node value is negative of its children's average
            if random_node.visit_count > 0:
                total_child_value = sum(child.visit_count * child.value() 
                                      for child in random_node.children.values())
                total_child_visits = sum(child.visit_count for child in random_node.children.values())
                if total_child_visits > 0:
                    random_node.total_value = -total_child_value

    # Generate policy
    policy = np.zeros(28, dtype=np.float32)
    valid_actions = game.valid_actions()
    
    if len(valid_actions) > 0:
        total_visits = 0
        action_visits = {}
        
        for action in valid_actions:
            visits = 0
            if action in root.children:
                visits = root.children[action].visit_count
            action_visits[action] = visits
            total_visits += visits
        
        if total_visits > 0:
            for action in valid_actions:
                policy[action] = action_visits[action] / total_visits
        else:
            for action in valid_actions:
                policy[action] = 1.0 / len(valid_actions)

    return policy


ReplayBufferEntry = collections.namedtuple("ReplayBufferEntry", ["board", "policy", "outcome"])


def sim_game(agent: Agent, args: argparse.Namespace) -> list[ReplayBufferEntry]:
    game = AZQuizRandomized()
    boards = []
    policies = []
    players = []  # Track which player made each move
    
    while not game.outcome(game.to_play):
        policy = mcts(game, agent, args, explore=True)
        boards.append(agent.board_features(game))
        policies.append(policy)
        players.append(game.to_play)  # Store current player
        
        valid_actions = game.valid_actions()
        if len(valid_actions) == 0:
            break
            
        if game.board.sum() < args.sampling_moves:
            valid_policy = np.array([policy[a] for a in valid_actions])
            valid_policy /= valid_policy.sum() + 1e-8
            action_idx = np.random.choice(len(valid_actions), p=valid_policy)
            action = valid_actions[action_idx]
        else:
            action = valid_actions[np.argmax([policy[a] for a in valid_actions])]
        
        game.move(action)
    
    # Determine final outcome for each player
    final_outcomes = {}
    for player in [0, 1]:
        outcome = game.outcome(player)
        if outcome == AZQuizRandomized.Outcome.WIN:
            final_outcomes[player] = 1.0
        elif outcome == AZQuizRandomized.Outcome.LOSS:
            final_outcomes[player] = -1.0
        else:
            final_outcomes[player] = 0.0
    
    replay_buffer_entries = []
    for board, policy, player in zip(boards, policies, players):
        player_value = final_outcomes[player]
        replay_buffer_entries.append(ReplayBufferEntry(board, policy, player_value))
    
    if args.show_sim_games:
        print(f"Game finished with outcomes: Player 0: {final_outcomes[0]}, Player 1: {final_outcomes[1]}")
    
    return replay_buffer_entries


def debug_mcts(args: argparse.Namespace) -> None:
    """Debug MCTS by testing with blank agent and different simulation counts"""
    print("=== DEBUGGING MCTS ===")
    print("Testing MCTS with blank agent (uniform policy, draw value)")
    
    # Parse simulation counts
    sim_counts = [int(x.strip()) for x in args.debug_sim_counts.split(',')]
    
    # Create blank agent
    blank_agent = BlankAgent(args)
    
    # Test each simulation count
    results = []
    for sim_count in sim_counts:
        print(f"\nTesting with {sim_count} simulations...")
        
        # Create test arguments with specific simulation count
        test_args = argparse.Namespace(**vars(args))
        test_args.num_simulations = sim_count
        
        # Create player with blank agent
        player = Player(blank_agent, test_args)
        
        # Evaluate against simple heuristic
        score = npfl139.board_games.evaluate(
            AZQuizRandomized,
            [player, AZQuizRandomized.player_from_name("simple_heuristic")(seed=args.seed)],
            games=20,  # Use fewer games for debugging
            first_chosen=False,
            render=False,
            verbose=False,
        )
        
        results.append((sim_count, score))
        print(f"  Score with {sim_count} simulations: {score:.3f} ({score*100:.1f}%)")
    
    # Print summary
    print("\n=== MCTS DEBUG SUMMARY ===")
    print("Simulation Count | Score")
    print("-" * 25)
    for sim_count, score in results:
        print(f"{sim_count:12d} | {score:.3f}")
    
    # Check if performance is increasing
    print("\nPerformance trend:")
    for i in range(1, len(results)):
        prev_score = results[i-1][1]
        curr_score = results[i][1]
        trend = "↑" if curr_score > prev_score else "↓" if curr_score < prev_score else "→"
        print(f"  {results[i-1][0]} → {results[i][0]}: {trend} ({curr_score - prev_score:+.3f})")
    
    print("\n=== TESTING SIMULATED GAMES ===")
    print("Running simulated games with show_sim_games=True to verify outcome computation")
    
    # Test simulated games with show output
    test_args = argparse.Namespace(**vars(args))
    test_args.show_sim_games = True
    test_args.num_simulations = 50  # Use moderate simulation count
    
    print("\nRunning 3 test games:")
    for i in range(3):
        print(f"\n--- Game {i+1} ---")
        game_entries = sim_game(blank_agent, test_args)
        print(f"Generated {len(game_entries)} training entries")
        
        # Show some stats about the game
        outcomes = [entry.outcome for entry in game_entries]
        print(f"Outcome distribution: {np.bincount(np.array(outcomes, dtype=int) + 1)} (loss/draw/win)")


def train(args: argparse.Namespace) -> Agent:
    agent = Agent(args)
    replay_buffer = npfl139.MonolithicReplayBuffer(max_length=args.replay_buffer_length)
    
    # Warmup
    while len(replay_buffer) < args.batch_size * 15:
        game = sim_game(agent, args)
        replay_buffer.extend(game)
    
    iteration = 0
    best_score = 0
    
    while True:
        iteration += 1
        
        # Generate games - using args.sim_games directly
        for i in range(args.sim_games):
            game = sim_game(agent, args)
            replay_buffer.extend(game)
        
        # Training - using args.train_for directly
        for _ in range(args.train_for):
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
                AZQuizRandomized, 
                [Player(agent, argparse.Namespace(num_simulations=args.num_simulations)),
                 AZQuizRandomized.player_from_name("simple_heuristic")(seed=args.seed)],
                games=56, first_chosen=False, render=False, verbose=False,
            )
            
            if score > best_score:
                best_score = score
                agent.save(args.model_path)
                print("New best score: {:.1f}%".format(100 * score))
            
            if score >= 0.95:
                return agent


class Player(npfl139.board_games.BoardGamePlayer[AZQuizRandomized]):
    def __init__(self, agent: Agent, args: argparse.Namespace):
        self.agent = agent
        self.args = args
        self._policy_only = False
        self._move_times = []
        self._time_threshold = 0.05
        self._game_cache = {}

    def play(self, game: AZQuizRandomized) -> int:
        start_time = time.time()
        valid_actions = game.valid_actions()
        if len(valid_actions) == 0:
            return 0

        # 🚨 Avoid MCTS if explicitly disabled
        if self.args.num_simulations == 0 or self._should_use_policy_only(game):
            action = self._play_policy_only(game)
        else:
            reduced_args = argparse.Namespace(**vars(self.args))
            reduced_args.num_simulations = max(16, self.args.num_simulations // 4)

            try:
                policy = mcts(game, self.agent, reduced_args, explore=False)
                action = max(valid_actions, key=lambda a: policy[a])

                move_time = time.time() - start_time
                self._move_times.append(move_time)

                if len(self._move_times) >= 2 and all(t > self._time_threshold for t in self._move_times[-2:]):
                    self._policy_only = True
            except Exception:
                action = self._play_policy_only(game)

        return action
    
    def _should_use_policy_only(self, game: AZQuizRandomized) -> bool:
        if self._policy_only:
            return True
            
        move_count = game.board.sum()
        if move_count < 15 or move_count > 30:
            return True
            
        game_key = self._get_game_key(game)
        if game_key in self._game_cache:
            return True
            
        return False
    
    def _play_policy_only(self, game: AZQuizRandomized) -> int:
        valid_actions = game.valid_actions()
        if len(valid_actions) == 0:
            return 0
        
        game_key = self._get_game_key(game)
        if game_key in self._game_cache:
            cached_action = self._game_cache[game_key]
            if cached_action in valid_actions:
                return cached_action
        
        board_features = self.agent.board_features(game)
        board_tensor = torch.from_numpy(board_features).unsqueeze(0).to(self.agent.device)
        
        with torch.no_grad():
            self.agent._model.eval()
            policy_logits, _ = self.agent._model(board_tensor)
            policy = F.softmax(policy_logits, dim=-1)[0]
        
        best_action = max(valid_actions, key=lambda a: policy[a])
        self._game_cache[game_key] = best_action
        
        return best_action
    
    def _get_game_key(self, game: AZQuizRandomized) -> str:
        return f"{game.board.tobytes()}_{game.to_play}"


def main(args: argparse.Namespace) -> Player:
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()

    # Run debugging if requested
    if args.debug_mcts:
        debug_mcts(args)
        return None  # Exit after debugging

    if args.recodex:
        agent = Agent.load(args.model_path, args)
    else:
        agent = train(args)

    return Player(agent, args)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    player = main(main_args)
    
    # Only run evaluation if not debugging
    if not main_args.debug_mcts and player is not None:
        npfl139.board_games.evaluate(
            AZQuizRandomized, [player, AZQuizRandomized.player_from_name("simple_heuristic")(seed=main_args.seed)],
            games=56, first_chosen=False, render=False, verbose=True,
        )