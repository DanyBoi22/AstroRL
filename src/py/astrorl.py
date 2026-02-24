#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import torch
import math
import json
import time
from IPython.display import HTML
from line_profiler import LineProfiler
from dataclasses import dataclass

from forcemodel import IForceModel, GravityForce
from integrator import IIntegrator, SymplecticEuler, VelocityVerlet
from env import SolarSystemEnv, run_simulation, run_script_simulation, adjust_barycentric
from ship import SimpleImpulseShip, Maneuver, simple_action_space, IShip
from rlmontecarlo import train_mc, MonteCarloAgent
from rlactorcritic import train_ac, ActorCriticAgent

@dataclass
class TrainingConfig:
    bodies: list
    dt: float
    max_steps: int
    n_episodes: int
    log_every: int
    log_states: bool
    log_n_entries: int
    ship_index: int
    mass: float
    thrust: float
    actions: object
    safety_radius: float
    escape_dist: float
    escape_vel: float
    target_dist: float
    target_vel: float
    reference_point_index: int
    reward_coef: float

def run_training_mc(forcemodel: IForceModel, integrator: IIntegrator, iship: IShip, iagent: MonteCarloAgent, training_config: TrainingConfig, root_folder: str):
    ship_kwargs = {k: getattr(training_config, k) for k in [
        "ship_index", "mass", "thrust", "actions",
        "safety_radius", "escape_dist", "escape_vel",
        "target_dist", "target_vel", "reference_point_index",
        "reward_coef"
    ]}
    ship = iship(**ship_kwargs)

    env = SolarSystemEnv(forcemodel, integrator, bodies=training_config.bodies, ship=ship, dt=training_config.dt)
    obs0 = env.reset()
    obs_dim = obs0.shape[0]
    
    agent = iagent(obs_dim, n_actions=training_config.actions.n)
    train_mc(env, agent, n_episodes=training_config.n_episodes, max_steps=training_config.max_steps, log_every=training_config.log_every, log_states=training_config.log_states, log_n_entries=training_config.log_n_entries, root_folder=root_folder)

def run_training_ac(forcemodel: IForceModel, integrator: IIntegrator, iship: IShip, iagent: ActorCriticAgent, training_config: TrainingConfig, root_folder: str):
    ship_kwargs = {k: getattr(training_config, k) for k in [
        "ship_index", "mass", "thrust", "actions",
        "safety_radius", "escape_dist", "escape_vel",
        "target_dist", "target_vel", "reference_point_index",
        "reward_coef"
    ]}
    ship = iship(**ship_kwargs)

    env = SolarSystemEnv(forcemodel, integrator, bodies=training_config.bodies, ship=ship, dt=training_config.dt)
    obs0 = env.reset()
    obs_dim = obs0.shape[0]
    
    agent = iagent(obs_dim, n_actions=training_config.actions.n)
    train_ac(env, agent, n_episodes=training_config.n_episodes, max_steps=training_config.max_steps, log_every=training_config.log_every, log_states=training_config.log_states, log_n_entries=training_config.log_n_entries, root_folder=root_folder)

def main():
    earth_radius = 6.371e6
    bodies = [
        np.array([      # Masses in kg
            5.972e24,       # Earth
            1.0e5,          # Ship
        ]),
        np.array([      # Positions (x, y) in meters
            [0.0, 0.0],
            [earth_radius + 2e6, 0.0], 
        ]),
        np.array([      # Velocities (vx, vy) in m/s
            [0.0, 0.0],
            [0.0, 6.9e3],
        ]),
        np.array([      # Names of bodies
            "Earth",
            "Ship"
        ]),
        np.array([      # Colors for visuals
            "blue",         # Earth
            "lightblue",    # Ship
        ]),
    ]

    dt = 1
    simulation_time = 10500 * 1 * 1  # seconds
    max_steps = int(simulation_time / dt)

    ship_index = len(bodies[0])-1
    mass = bodies[0][ship_index]
    reward_coef = 1.0 / max_steps
    thrust = 1e6 * dt

    training_config = TrainingConfig(
        bodies=bodies,
        dt=dt,
        max_steps=max_steps,
        n_episodes=800,
        log_every=10,
        log_states=True,
        log_n_entries=400,
        ship_index=ship_index,
        mass=mass,
        thrust=thrust,
        actions=simple_action_space, # 4 directions + no thrust
        safety_radius=earth_radius + 1e5, # 100km is an atmo entry
        escape_dist=1e8,
        escape_vel=1e5,
        target_dist=earth_radius + 4e6,
        target_vel=6.2e3,
        reference_point_index=0,
        reward_coef=reward_coef
    )

    forcemodel = GravityForce
    integrator = VelocityVerlet
    ship = SimpleImpulseShip
    #agent = MonteCarloAgent
    agent = ActorCriticAgent

    for seed_id in range(1, 11):

        np.random.seed(seed_id)
        random.seed(seed_id)
        torch.manual_seed(seed_id)

        print(f"------------------------------------\n"
            f"Running Training {seed_id} ...\n"
            f"------------------------------------\n")

        if agent == MonteCarloAgent:
            root_folder = (f"training_logs/mc/mc_{training_config.n_episodes}_{training_config.max_steps}_{seed_id}")
            run_training_mc(forcemodel, integrator, ship, agent, training_config, root_folder=root_folder)
        elif agent == ActorCriticAgent:
            root_folder = (f"training_logs/ac/ac_{training_config.n_episodes}_{training_config.max_steps}_{seed_id}")
            run_training_ac(forcemodel, integrator, ship, agent, training_config, root_folder=root_folder)
        else : 
            print("Plop")

if __name__ == "__main__":
    main()