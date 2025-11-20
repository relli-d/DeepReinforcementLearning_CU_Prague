#!/usr/bin/env python3
import argparse
import collections
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

import npfl139
npfl139.require_version("2425.6")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--frame_skip", default=4, type=int, help="Frame skip.")
parser.add_argument("--game", default="Pong", type=str, help="Game to play.")
# Training hyperparameters
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--epsilon", default=1.0, type=float, help="Initial exploration.")
parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration.")
parser.add_argument("--epsilon_final_at", default=1000000, type=int, help="Final exploration frame.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor.")
parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate.")
parser.add_argument("--quantiles", default=51, type=int, help="Number of quantiles.")
parser.add_argument("--kappa", default=1.0, type=float, help="Huber loss threshold.")
parser.add_argument("--target_update_freq", default=10000, type=int, help="Target network update frequency.")
parser.add_argument("--buffer_size", default=100000, type=int, help="Replay buffer size.")
parser.add_argument("--training_steps", default=3000000, type=int, help="Total training steps.")
parser.add_argument("--learning_starts", default=50000, type=int, help="Steps before learning starts.")
parser.add_argument("--update_freq", default=4, type=int, help="Update frequency.")
parser.add_argument("--grad_clip", default=10.0, type=float, help="Gradient clipping threshold.")
# New arguments for resume training
parser.add_argument("--resume_training", default=False, action="store_true", help="Resume training from checkpoint")
parser.add_argument("--checkpoint_path", default="", type=str, help="Path to checkpoint to resume from")
parser.add_argument("--target_score", default=16.0, type=float, help="Target score to reach")


class QRDQNNetwork(nn.Module):
    """Enhanced QR-DQN network for Atari games."""
    
    def __init__(self, num_actions: int, num_quantiles: int = 51):
        super().__init__()
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Calculate conv output size: 84x84 -> 20x20 -> 9x9 -> 7x7
        conv_out_size = 64 * 7 * 7
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_actions * num_quantiles)
        
        self.dropout = nn.Dropout(0.1)
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning quantiles for each action."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        # Normalize input to [0, 1] if not already
        if x.max() > 1.0:
            x = x / 255.0
        
        # Convolutional layers with batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Reshape to [batch, num_actions, num_quantiles]
        return x.view(-1, self.num_actions, self.num_quantiles)


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer."""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0
        
        # Storage
        self.states = np.zeros((capacity, 4, 84, 84), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, 4, 84, 84), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        
        self.position = 0
        self.size = 0
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """Add a transition with maximum priority."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        self.priorities[self.position] = self.max_priority
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample batch with prioritized sampling."""
        if self.size == 0:
            raise ValueError("Buffer is empty")
            
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # Importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return (
            self.states[indices],
            self.actions[indices], 
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights
        )
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities of sampled transitions."""
        priorities = np.abs(priorities) + 1e-6
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())
    
    def __len__(self) -> int:
        return self.size


class QRDQNAgent:
    """QR-DQN agent with prioritized replay."""
    
    def __init__(self, num_actions: int, args: argparse.Namespace):
        self.num_actions = num_actions
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = QRDQNNetwork(num_actions, args.quantiles).to(self.device)
        self.target_network = QRDQNNetwork(num_actions, args.quantiles).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), 
            lr=args.learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=500000, gamma=0.5
        )
        
        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(args.buffer_size)
        
        # Training parameters
        self.epsilon = args.epsilon
        self.gamma = args.gamma
        self.kappa = args.kappa
        self.total_steps = 0
        
        # Quantile midpoints
        self.tau = ((torch.arange(args.quantiles, device=self.device).float() + 0.5) / 
                   args.quantiles).view(1, -1, 1)
        
        self.training_losses = []
        
    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """Select action using epsilon-greedy policy."""
        if epsilon is None:
            epsilon = self.epsilon
            
        if np.random.random() < epsilon:
            return np.random.randint(self.num_actions)
        
        with torch.no_grad():
            if state.shape == (84, 84, 4):
                state = state.transpose(2, 0, 1)
            
            state_tensor = torch.from_numpy(state).float().to(self.device)
            if state_tensor.dim() == 3:
                state_tensor = state_tensor.unsqueeze(0)
            
            quantiles = self.q_network(state_tensor)
            q_values = quantiles.mean(dim=2)
            return q_values.argmax(dim=1).item()
    
    def compute_loss(self, states: torch.Tensor, actions: torch.Tensor, 
                     rewards: torch.Tensor, next_states: torch.Tensor, 
                     dones: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        """Compute QR-DQN loss with importance sampling."""
        batch_size = states.size(0)
        
        # Current quantiles
        current_quantiles = self.q_network(states)
        current_quantiles = current_quantiles[torch.arange(batch_size), actions]
        
        # Target quantiles (Double DQN)
        with torch.no_grad():
            next_quantiles_main = self.q_network(next_states)
            next_q_values = next_quantiles_main.mean(dim=2)
            next_actions = next_q_values.argmax(dim=1)
            
            next_quantiles_target = self.target_network(next_states)
            next_quantiles = next_quantiles_target[torch.arange(batch_size), next_actions]
            
            target_quantiles = rewards.unsqueeze(1) + \
                              self.gamma * (1 - dones.unsqueeze(1)) * next_quantiles
        
        # TD errors and quantile loss
        td_errors = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)
        
        huber_loss = torch.where(
            td_errors.abs() <= self.kappa,
            0.5 * td_errors.pow(2),
            self.kappa * (td_errors.abs() - 0.5 * self.kappa)
        )
        
        quantile_loss = (self.tau - (td_errors < 0).float()).abs() * huber_loss
        elementwise_loss = quantile_loss.sum(dim=2).mean(dim=1)
        
        # Apply importance sampling weights
        loss = (weights * elementwise_loss).mean()
        
        # Return TD errors for priority update
        priorities = elementwise_loss.detach().cpu().numpy()
        
        return loss, priorities
    
    def update(self) -> Optional[float]:
        """Update with prioritized replay."""
        if len(self.replay_buffer) < self.args.batch_size:
            return None
        
        # Sample with priorities
        states, actions, rewards, next_states, dones, indices, weights = \
            self.replay_buffer.sample(self.args.batch_size)
        
        # Convert to tensors
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)
        weights = torch.from_numpy(weights).float().to(self.device)
        
        # Compute loss and priorities
        loss, priorities = self.compute_loss(states, actions, rewards, next_states, dones, weights)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.args.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        
        # Update priorities
        self.replay_buffer.update_priorities(indices, priorities)
        
        self.training_losses.append(loss.item())
        return loss.item()
    
    def update_target_network(self) -> None:
        """Update target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def update_epsilon(self) -> None:
        """Update epsilon with linear decay."""
        self.epsilon = np.interp(self.total_steps, 
                               [0, self.args.epsilon_final_at], 
                               [self.args.epsilon, self.args.epsilon_final])
    
    def save(self, path: str, lightweight: bool = False) -> None:
        """Save agent state."""
        if lightweight:
            # Save only the network weights for evaluation (much smaller file)
            torch.save(self.q_network.state_dict(), path)
        else:
            # Save full training state for resuming training
            torch.save({
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'total_steps': self.total_steps,
                'epsilon': self.epsilon,
                'training_losses': self.training_losses,
                'replay_buffer_size': len(self.replay_buffer),
                'replay_buffer_position': self.replay_buffer.position
            }, path)
    
    def save_for_evaluation(self, path: str) -> None:
        """Save lightweight model for evaluation only."""
        self.save(path, lightweight=True)
        
    def load(self, path: str, resume_training: bool = False) -> bool:
        """Load agent state. Returns True if successfully loaded full checkpoint."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'q_network' in checkpoint:
                # Full checkpoint with training state
                self.q_network.load_state_dict(checkpoint['q_network'])
                self.target_network.load_state_dict(checkpoint['target_network'])
                
                if resume_training:
                    # Restore full training state
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                    self.total_steps = checkpoint.get('total_steps', 0)
                    self.epsilon = checkpoint.get('epsilon', self.args.epsilon_final)
                    self.training_losses = checkpoint.get('training_losses', [])
                    
                    print(f"Successfully resumed training from {path}")
                    print(f"  Total steps: {self.total_steps}")
                    print(f"  Current epsilon: {self.epsilon:.4f}")
                    print(f"  Training losses: {len(self.training_losses)} recorded")
                    return True
                else:
                    print(f"Successfully loaded full checkpoint from {path} (evaluation mode)")
                    return True
            else:
                # Lightweight checkpoint - just network weights
                self.q_network.load_state_dict(checkpoint)
                self.target_network.load_state_dict(checkpoint)
                print(f"Successfully loaded lightweight model from {path}")
                return False
        except FileNotFoundError:
            print(f"No saved model found at {path}, starting fresh training")
            return False
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
            return False


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()
    
    # Apply wrappers to the environment
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.FrameStackObservation(env, 4)

    # Create agent
    agent = QRDQNAgent(env.action_space.n, args)
    
    # For ReCodEx evaluation
    if args.recodex:
        # Load the agent for evaluation
        agent.load(f"qrdqn_{args.game}_model.pth")
        
        # Final evaluation
        while True:
            state, done = env.reset(options={"start_evaluation": True})[0], False
            if state.shape == (84, 84, 4):
                state = state.transpose(2, 0, 1)
            
            while not done:
                # Choose a greedy action
                action = agent.select_action(state, epsilon=0.0)
                state, reward, terminated, truncated, _ = env.step(action)
                if state.shape == (84, 84, 4):
                    state = state.transpose(2, 0, 1)
                done = terminated or truncated
        return
    
    # Determine which checkpoint to load
    checkpoint_loaded = False
    if args.resume_training and args.checkpoint_path:
        checkpoint_loaded = agent.load(args.checkpoint_path, resume_training=True)
    else:
        # Try to load existing checkpoints automatically
        for checkpoint_name in [f"qrdqn_{args.game}_best_full.pth", f"qrdqn_{args.game}_final_full.pth"]:
            try:
                checkpoint_loaded = agent.load(checkpoint_name, resume_training=True)
                if checkpoint_loaded:
                    print(f"Automatically loaded checkpoint: {checkpoint_name}")
                    break
            except:
                continue
    
    # Training mode
    print(f"Enhanced QR-DQN Training on {args.game}")
    print(f"Device: {agent.device}")
    print(f"Target score: {args.target_score}")
    if checkpoint_loaded:
        print(f"Resuming training from step {agent.total_steps}")
    else:
        print("Starting fresh training")
    
    print(f"Hyperparameters:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Buffer size: {args.buffer_size}")
    print(f"  Learning starts: {args.learning_starts}")
    
    # Training variables
    episode_rewards = []
    episode_lengths = []
    best_score = -float('inf')
    episodes_completed = 0
    recent_scores = collections.deque(maxlen=100)
    
    # Training loop
    state, _ = env.reset()
    if state.shape == (84, 84, 4):
        state = state.transpose(2, 0, 1)
    
    episode_reward = 0
    episode_length = 0
    start_time = time.time()
    last_100_avg = -21.0
    
    training = True
    while training and agent.total_steps < args.training_steps:
        # Update epsilon
        agent.update_epsilon()
        
        # Select action
        action = agent.select_action(state)
        
        # Step environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        if next_state.shape == (84, 84, 4):
            next_state = next_state.transpose(2, 0, 1)
        done = terminated or truncated
        
        # Store experience
        agent.replay_buffer.push(state, action, reward, next_state, done)
        
        episode_reward += reward
        episode_length += 1
        agent.total_steps += 1
        
        # Perform update
        if (agent.total_steps > args.learning_starts and 
            agent.total_steps % args.update_freq == 0):
            loss = agent.update()
        
        # Update target network
        if agent.total_steps % args.target_update_freq == 0:
            agent.update_target_network()
            print(f"Updated target network at step {agent.total_steps}")
        
        if done:
            episodes_completed += 1
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            recent_scores.append(episode_reward)
            
            # Calculate current average
            current_avg = np.mean(recent_scores) if len(recent_scores) >= 10 else -21.0
            
            # Track best score and save model
            if episode_reward > best_score:
                best_score = episode_reward
                # Save full checkpoint for training continuation
                agent.save(f"qrdqn_{args.game}_best_full.pth", lightweight=False)
                # Save lightweight model for evaluation
                agent.save_for_evaluation(f"qrdqn_{args.game}_best.pth")
            
            # Check if target reached
            if current_avg >= args.target_score and len(recent_scores) >= 50:
                print(f"TARGET REACHED! Average score over last {len(recent_scores)} episodes: {current_avg:.2f}")
                # Save the final evaluation model
                agent.save_for_evaluation(f"qrdqn_{args.game}_model.pth")
                agent.save(f"qrdqn_{args.game}_final_full.pth", lightweight=False)
                training = False
            
            # Enhanced logging
            if episodes_completed % 10 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                elapsed_time = time.time() - start_time
                
                improvement = avg_reward - last_100_avg
                improvement_str = f" (+{improvement:.2f})" if improvement > 0 else f" ({improvement:.2f})"
                last_100_avg = avg_reward
                
                print(f"Episode {episodes_completed:4d} | "
                      f"Steps: {agent.total_steps:7d} | "
                      f"Reward: {episode_reward:6.1f} | "
                      f"Avg(100): {avg_reward:6.1f}{improvement_str} | "
                      f"Recent({len(recent_scores)}): {current_avg:6.1f} | "
                      f"Best: {best_score:6.1f} | "
                      f"Eps: {agent.epsilon:.3f} | "
                      f"Time: {elapsed_time/60:.1f}m")
                
                if len(agent.training_losses) > 0:
                    avg_loss = np.mean(agent.training_losses[-100:])
                    print(f"         Loss: {avg_loss:.4f} | Buffer: {len(agent.replay_buffer)}")
            
            # Reset for next episode
            state, _ = env.reset()
            if state.shape == (84, 84, 4):
                state = state.transpose(2, 0, 1)
            episode_reward = 0
            episode_length = 0
            
            # Save periodic checkpoint
            if episodes_completed % 100 == 0:
                agent.save(f"qrdqn_{args.game}_checkpoint_{episodes_completed}.pth", lightweight=False)
                print(f"Saved checkpoint at episode {episodes_completed}")
        else:
            state = next_state
    
    # Save final model (lightweight for evaluation)
    agent.save_for_evaluation(f"qrdqn_{args.game}_model.pth")
    # Also save full checkpoint if you want to resume training later
    agent.save(f"qrdqn_{args.game}_final_full.pth", lightweight=False)
    
    final_avg = np.mean(recent_scores) if len(recent_scores) > 0 else 0
    print(f"Training completed!")
    print(f"Best single episode score: {best_score}")
    print(f"Final average score (last {len(recent_scores)} episodes): {final_avg:.2f}")
    print(f"Lightweight evaluation model saved as: qrdqn_{args.game}_model.pth")
    if len(agent.training_losses) > 0:
        print(f"Final average loss: {np.mean(agent.training_losses[-1000:]):.4f}")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        gym.make("ALE/{}-v5".format(main_args.game), frameskip=main_args.frame_skip),
        main_args.seed, main_args.render_each)

    main(main_env, main_args)