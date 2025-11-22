#!/usr/bin/env python3
import argparse
import collections
import copy
import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2425.8")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="BipedalWalkerHardcore-v3", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--envs", default=32, type=int, help="Environments.")
parser.add_argument("--evaluate_each", default=500, type=int, help="Evaluate each number of updates.")
parser.add_argument("--evaluate_for", default=100, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=512, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="walker_hardcore.pt", type=str, help="Model path")
parser.add_argument("--replay_buffer_size", default=1_000_000, type=int, help="Replay buffer size")
parser.add_argument("--target_entropy", default=-5, type=float, help="Target entropy per action component.")
parser.add_argument("--target_tau", default=0.005, type=float, help="Target network update weight.")


class Agent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        self.args = args

        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        self.target_entropy = -act_dim

        class Actor(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.device = device
                self.base = torch.nn.Sequential(
                    torch.nn.Linear(obs_dim, args.hidden_layer_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(args.hidden_layer_size, args.hidden_layer_size),
                    torch.nn.ReLU()
                )
                self.mu_layer = torch.nn.Linear(args.hidden_layer_size, act_dim)
                self.log_std_layer = torch.nn.Linear(args.hidden_layer_size, act_dim)
                torch.nn.init.constant_(self.log_std_layer.bias, -1.0)

                high, low = env.action_space.high, env.action_space.low
                self.register_buffer("action_scale", torch.tensor((high - low) / 2, dtype=torch.float32).to(self.device))
                self.register_buffer("action_offset", torch.tensor((high + low) / 2, dtype=torch.float32).to(self.device))

                self.transform = torch.distributions.transforms.ComposeTransform([
                    torch.distributions.transforms.TanhTransform(),
                    torch.distributions.transforms.AffineTransform(loc=self.action_offset, scale=self.action_scale)
                ], cache_size=1)

                self.log_alpha = torch.nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))

            def forward(self, obs, sample: bool):
                hidden = self.base(obs)
                mu = self.mu_layer(hidden)
                std = torch.nn.functional.softplus(self.log_std_layer(hidden)).clamp(min=1e-4)
                dist = torch.distributions.Normal(mu, std)
                td = torch.distributions.TransformedDistribution(dist, self.transform)

                if sample:
                    action = td.rsample()
                    log_prob = td.log_prob(action).sum(-1)
                else:
                    action = torch.tanh(mu) * self.action_scale + self.action_offset
                    log_prob = torch.zeros(obs.shape[0], device=obs.device)
                alpha = self.log_alpha.exp()
                return action, log_prob, alpha

        self.actor = Actor(self.device).to(self.device)

        class Critic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(obs_dim + act_dim, args.hidden_layer_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(args.hidden_layer_size, args.hidden_layer_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(args.hidden_layer_size, 1)
                )

            def forward(self, obs, act):
                return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)

        self.critic1 = Critic().to(self.device)
        self.critic2 = Critic().to(self.device)
        self.target_critic1 = copy.deepcopy(self.critic1).eval()
        self.target_critic2 = copy.deepcopy(self.critic2).eval()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.learning_rate)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=args.learning_rate)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=args.learning_rate)
        self.alpha_optimizer = torch.optim.Adam([self.actor.log_alpha], lr=args.learning_rate)

        self.actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer, mode="max", factor=0.5, patience=10, threshold=1e-2, verbose=True
        )

        self.mse = torch.nn.MSELoss()

    @npfl139.typed_torch_function(device, torch.float32, torch.float32, torch.float32)
    def train(self, states, actions, targets):
        sampled_actions, log_probs, alpha = self.actor(states, sample=True)
        q1, q2 = self.critic1(states, sampled_actions), self.critic2(states, sampled_actions)
        actor_loss = (alpha.detach() * log_probs - torch.min(q1, q2)).mean()

        alpha_loss = -(self.actor.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

        q1_pred, q2_pred = self.critic1(states, actions), self.critic2(states, actions)
        critic1_loss = self.mse(q1_pred, targets)
        critic2_loss = self.mse(q2_pred, targets)

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        npfl139.update_params_by_ema(self.target_critic1, self.critic1, self.args.target_tau)
        npfl139.update_params_by_ema(self.target_critic2, self.critic2, self.args.target_tau)

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_values(self, next_states):
        with torch.no_grad():
            next_actions, log_probs, alpha = self.actor(next_states, sample=True)
            q1, q2 = self.target_critic1(next_states, next_actions), self.target_critic2(next_states, next_actions)
            return torch.min(q1, q2) - alpha * log_probs

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_sampled_actions(self, states):
        with torch.no_grad():
            return self.actor(states, sample=True)[0]

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_mean_actions(self, states):
        with torch.no_grad():
            return self.actor(states, sample=False)[0]

    def save_actor(self, path):
        torch.save(self.actor.state_dict(), path)

    def load_actor(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=self.device))

    def step_scheduler(self, metric: float):
        self.actor_scheduler.step(metric)


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()

    agent = Agent(env, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        state = env.reset(options={"start_evaluation": start_evaluation, "logging": logging})[0]
        rewards, done = 0, False
        while not done:
            action = agent.predict_mean_actions(state[np.newaxis])[0]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    if args.recodex:
        agent.load_actor(args.model_path)
        while True:
            evaluate_episode(True)

    vector_env = gym.make_vec(args.env, args.envs, gym.VectorizeMode.ASYNC,
                              vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})
    replay_buffer = npfl139.MonolithicReplayBuffer(args.replay_buffer_size, args.seed)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    state = vector_env.reset(seed=args.seed)[0]
    best_avg_return = -np.inf

    while True:
        for _ in range(args.evaluate_each):
            action = agent.predict_sampled_actions(state)
            next_state, reward, terminated, truncated, _ = vector_env.step(action)
            done = terminated | truncated
            reward = np.clip(reward, -1.0, 1.0)
            replay_buffer.append_batch(Transition(state, action, reward, done, next_state))
            state = next_state

            if len(replay_buffer) >= 1000 and len(replay_buffer) >= 10 * args.batch_size:
                for _ in range(2):
                    states, actions, rewards, dones, next_states = replay_buffer.sample(args.batch_size)
                    values = agent.predict_values(next_states)
                    returns = rewards + args.gamma * (1.0 - dones) * values
                    agent.train(states, actions, returns)

        returns = [evaluate_episode() for _ in range(args.evaluate_for)]
        avg_return = np.mean(returns)
        agent.step_scheduler(avg_return)

        if avg_return > best_avg_return and avg_return > 0:
            best_avg_return = avg_return
            print(f"New best average return {avg_return:.2f}, saving model.")
            agent.save_actor(args.model_path)

        if avg_return >= 130:
            print("Finished training, saving final model.")
            agent.save_actor(args.model_path)
            break

    while True:
        evaluate_episode(True)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make(main_args.env), main_args.seed, main_args.render_each)

    main(main_env, main_args)