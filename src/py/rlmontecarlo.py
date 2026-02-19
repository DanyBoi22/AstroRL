#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from line_profiler import LineProfiler, profile
from helpers import save_states
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


def compute_returns(rewards, gamma):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


class MonteCarloAgent:
    def __init__(self, obs_dim, n_actions, lr=3e-4, gamma=0.99, device="cpu"):
        
        self.gamma = gamma
        self.device = device

        self.policy = PolicyNet(obs_dim, n_actions).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    @profile
    def run_episode(self, env, max_steps=100000, log_states=False, log_n_entries=400):
        #print("Started episode")
        log_probs = []
        rewards = []

        obs = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        states = []
        if log_states:
            step = max(1, math.floor(max_steps / log_n_entries))
        else : 
            step = 100

        for i in range(max_steps):
            dist = self.policy(obs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            if log_states and i % step == 0:
                x, v, a, next_obs, reward, done = env.step(action.item())
                states.append({"m": env.m, "x": x, "v": v, "a": a, "t": env.t})
            else:
                _, _, _, next_obs, reward, done = env.step(action.item())

            #print(f"Log prob: {log_prob}")
            #print(f"Next_obs: {next_obs}")
            #print(f"Reward: {reward}")
            #print(f"Done: {done}")
            
            log_probs.append(log_prob)
            rewards.append(reward)

            next_obs = np.clip(next_obs, -5.0, 5.0)
            obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)

            if done:
                break

            # exceeded max episode steps
            if i >= max_steps:
                break
                
        #print("Finished episode")
        if log_states:
            return log_probs, rewards, states
        else: 
            return log_probs, rewards

    @profile
    def update_policy(self, log_probs, rewards):
        returns = compute_returns(rewards, self.gamma).to(self.device)

        # normalize returns (important for stability)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0.0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def train_mc(env, agent, n_episodes=1000, max_steps=100000, log_every=10, log_states=False, log_n_entries=400):
    states_list = []
    for ep in range(1, n_episodes + 1):
        if log_states and ep % log_every == 0:
            log_probs, rewards, states = agent.run_episode(env, max_steps, log_states, log_n_entries)
        else:
            log_probs, rewards = agent.run_episode(env, max_steps)
        
        loss = agent.update_policy(log_probs, rewards)
        #print("Updated policy")

        total_reward = sum(rewards)

        if ep % log_every == 0:
            print(
                f"Episode {ep:4d} | "
                f"Return: {total_reward: .3e} | "
                f"Loss: {loss: .3e}"
            )
            if log_states:
                states_list.append(states)
                save_states(f"training_logs/mc_{n_episodes}_{max_steps}/episode{ep}.json", states)

    return states_list




def train_mc(env, agent, n_episodes=1000, max_steps=100000, log_every=10, log_states=False, log_n_entries=400):
    states_list = []
    elapsed_times = []
    folder = f"training_logs/mc_{n_episodes}_{max_steps}"
    os.makedirs(folder, exist_ok=True)

    for ep in range(1, n_episodes + 1):
        start = time.perf_counter()  # start timer

        if log_states and ep % log_every == 0:
            log_probs, rewards, states = agent.run_episode(env, max_steps, log_states, log_n_entries)
        else:
            log_probs, rewards = agent.run_episode(env, max_steps)
        
        loss = agent.update_policy(log_probs, rewards)
        total_reward = sum(rewards)

        end = time.perf_counter()  # end timer
        elapsed = end - start
        elapsed_times.append(elapsed)

        if ep % log_every == 0:
            mean_elapsed = statistics.mean(elapsed_times)
            elapsed_times = []
            print(
                f"Episode {ep:4d} | "
                f"Return: {total_reward: .3e} | "
                f"Loss: {loss: .3e} | "
                f"Mean elapsed time per episode: {mean_elapsed:.3f} s"
            )

            if log_states:
                states_list.append(states)
                save_states(os.path.join(folder, f"episode{ep}.json"), states)

    return states_list