#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from line_profiler import LineProfiler, profile
from helpers import save_states, append_batch_to_json_list
import time
import statistics
import os

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, obs):
        logits = self.net(obs)
        return Categorical(logits=logits)

class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)  # single scalar value
        )

    def forward(self, obs):
        # ensure output is always 1D tensor for advantage computation
        return self.net(obs).view(-1)

class ActorCriticAgent:
    def __init__(self, obs_dim, n_actions, lr=3e-4, gamma=0.99, device="cpu"):
        self.gamma = gamma
        self.device = device

        self.policy = PolicyNet(obs_dim, n_actions).to(device)
        self.value = ValueNet(obs_dim).to(device)

        self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr=lr)

    @profile
    def run_episode(self, env, max_steps=100000, log_states=False, log_n_entries=400):
        #print("Started episode")
        log_probs = []
        values = []
        rewards = []
        dones = []
        episode_actions = []

        obs = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        states = []
        if log_states:
            step = max(1, math.floor(max_steps / log_n_entries))
        else : 
            step = 100

        for i in range(max_steps):
            dist = self.policy(obs)
            value = self.value(obs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            if log_states and i % step == 0:
                x, v, a, next_obs, reward, done, success = env.step(action.item())
                states.append({"m": env.m, "x": x, "v": v, "a": a, "t": env.t})
            else:
                _, _, _, next_obs, reward, done, success = env.step(action.item())

            episode_actions.append(action.item())
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            dones.append(done)

            obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)

            if done:
                break

            # exceeded max episode steps
            if i+1 >= max_steps:
                print("Failure: Time limit reached")
                break

        # ensure obs has batch dimension for next_value
        obs_tensor = obs.unsqueeze(0) if obs.ndim == 1 else obs
        next_value = self.value(obs_tensor).detach().item() if not done else 0.0
            

        #print("Finished episode")
        if log_states:
            return log_probs, values, rewards, next_value, dones, success, episode_actions, states
        else: 
            return log_probs, values, rewards, next_value, dones, success, episode_actions

    @profile
    def update(self, log_probs, values, rewards, next_value, dones):
        returns = []
        R = next_value
        
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1.0 - float(done))  # cast done to float
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        values = torch.stack(values)

        advantages = returns - values

        policy_loss = (-torch.stack(log_probs) * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()  # MSE
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def train_ac(env, agent, n_episodes=1000, max_steps=100000, log_every=10, log_states=False, log_n_entries=400):
    action_buffer = []
    metadata_buffer = []
    reward_buffer = []
    times_buffer = []
    
    # convergence 
    alpha = 0.1
    tol = 0.0001
    ema_reward = None

    # logging
    folder = f"training_logs/ac_{n_episodes}_{max_steps}"
    actions_file = os.path.join(folder, "actions.json")
    metadata_file = os.path.join(folder, "metadata.json")
    
    episodes_folder = os.path.join(folder, "episodes")
    os.makedirs(episodes_folder, exist_ok=True)


    def flush_buffers():
        append_batch_to_json_list(actions_file, action_buffer)
        append_batch_to_json_list(metadata_file, metadata_buffer)

        action_buffer.clear()
        metadata_buffer.clear()
        
        reward_buffer.clear()
        times_buffer.clear()


    def save_episode_states(ep, states):
        save_states(os.path.join(episodes_folder, f"episode{ep}.json"), states)

    for ep in range(1, n_episodes + 1):
        start = time.perf_counter()  # start timer

        if log_states and ep % log_every == 0:
            log_probs, values, rewards, next_value, dones, success, episode_actions, states = agent.run_episode(env, max_steps, log_states, log_n_entries)
        else:
            log_probs, values, rewards, next_value, dones, success, episode_actions, = agent.run_episode(env, max_steps)
        
        loss = agent.update(log_probs, values, rewards, next_value, dones)
        total_reward = sum(rewards)

        end = time.perf_counter()
        elapsed = end - start

        times_buffer.append(elapsed)
        reward_buffer.append(total_reward)

        if log_states:
            action_buffer.append(episode_actions)
            metadata_buffer.append({
                "episode": ep,
                "reward": total_reward,
                "loss": loss,
                "success": success,
                "time": elapsed,
            })

            if ep % log_every == 0:
                mean_elapsed = statistics.mean(times_buffer)
                mean_rewards = np.mean(reward_buffer)

                print(
                    f"Episode {ep:4d} | "
                    f"Mean return per episode: {mean_rewards: .3e} | "
                    f"Mean elapsed time per episode: {mean_elapsed:.3f} s"
                )

                flush_buffers()
                save_episode_states(ep, states)
                states = None

        # convergence check
        if ema_reward is None:
            ema_reward = total_reward
        else:
            prev_ema = ema_reward
            ema_reward = alpha * total_reward + (1 - alpha) * ema_reward

            if abs(ema_reward - prev_ema) < tol:
                print(f"Converged at episode {ep}")

                if log_states:
                    flush_buffers()
                    if states is not None:
                        save_episode_states(ep, states)
                break

        #success_rate = sum(m["finish"] == "success" for m in metadata) / len(metadata)