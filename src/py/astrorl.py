#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
import json
import time
from IPython.display import HTML
from line_profiler import LineProfiler

from forcemodel import IForceModel, GravityForce
from integrator import IIntegrator, SymplecticEuler, VelocityVerlet
from env import SolarSystemEnv, run_simulation, run_script_simulation, adjust_barycentric
from vis import visualize_trajectories, plot_static_trajectories
from helpers import save_states, load_states
from ship import SimpleImpulseShip, Maneuver, simple_action_space
from rlmontecarlo import train_mc, MonteCarloAgent, PolicyNet, compute_returns

# Cool but unstable with really massive suns
# bodies = [
#     # Masses (kg)
#     np.array([
#         1.0e31,
#         1.0e31,
#         1.0e31,
#     ]),
#     # Positions (x, y) in meters
#     np.array([
#         [0.97e11, -0.243e11],   
#         [-0.97e11, 0.243e11],
#         [0.0, 0.0], 
#     ]),
#     # Velocities (vx, vy) in m/s
#     np.array([
#         [0.466e5, 0.432e5],    
#         [0.466e5, 0.432e5],
#         [-0.932e5, -0.864e5], 
#     ]),
#     # Names
#     np.array([
#         "One",
#         "Two",
#         "Three",
#     ]),
#     # Colors (reasonable defaults for visualization)
#     np.array([
#         "red",        
#         "blue",
#         "green",
#     ]),
# ]

bodies = [
    # Masses
    np.array([
        5.972e24,       # Earth
        7.342e22,       # Luna
        1.0e5,          # Ship
    ]),
    # Positions (x, y) in meters
    np.array([
        [0.0, 0.0],                # Earth
        [3.844e8, 0.0],      # Luna
        [3.6e7, 0.0],        # Ship
    ]),
    # Velocities (vx, vy) in m/s
    np.array([
        [0.0, 0.0],                   # Earth
        [0.0, 1022.0],          # Luna
        [0.0, 3055.0],         # Ship
    ]),
    # Names of celestial bodies
    np.array([
        "Earth",
        "Moon",
        "Ship"
    ]),
    # Colors for visuals
    np.array([
        "blue",       # Earth
        "gray",       # Moon (Luna)
        "lightblue",  # Ship
    ]),
]

bodies = [
    # Masses
    np.array([
        5.972e24,       # Earth
        1.0e5,          # Ship
    ]),
    # Positions (x, y) in meters
    np.array([
        [0.0, 0.0],                # Earth
        [2e6, 0.0],        # Ship
    ]),
    # Velocities (vx, vy) in m/s
    np.array([
        [0.0, 0.0],                   # Earth
        [0.0, 14000.0],         # Ship
    ]),
    # Names of celestial bodies
    np.array([
        "Earth",
        "Ship"
    ]),
    # Colors for visuals
    np.array([
        "blue",       # Earth
        "lightblue",  # Ship
    ]),
]

def main():
    ship_index = len(bodies[0])-1
    mass = bodies[0][-1]

    actions = simple_action_space 
    safety_radius = 1e4 # on a space scale 10 km is defenitely an atmo entry
    escape_dist = 1e8
    #escape_dist = 6e8   
    escape_vel = 4e5 
    #target_dist = 1e5 # for a low lunar orbit
    #target_vel = 1700 # for a low lunar orbit
    target_dist = 4e6 
    #target_dist = 3.6e7 # for a Far Earth orbit
    target_vel = 3055.0 # for a Far Earth orbit
    reference_point_index = 0

    n_actions = actions.n
    n_episodes=4

    dt=0.1
    simulation_time = 3600 * 1 * 1
    max_steps= int(simulation_time/dt)

    reward_coef = 1.0 / max_steps

    thrust = 1e6/dt # thrust changes with dt so keep that in mind

    ship = SimpleImpulseShip(ship_index=ship_index, mass=mass,thrust=thrust, actions=actions, safety_radius=safety_radius, escape_dist=escape_dist, escape_vel=escape_vel, target_dist=target_dist, target_vel=target_vel, reference_point_index=reference_point_index, reward_coef=reward_coef)

    env = SolarSystemEnv(GravityForce, VelocityVerlet, bodies, ship=ship, dt=dt)

    obs0 = env.reset()
    obs_dim = obs0.shape[0]


    # # Monte Carlo RL with NN-Policy Table #
    agent = MonteCarloAgent(obs_dim, n_actions)
    states = train_mc(env, agent, n_episodes=n_episodes, max_steps=max_steps, log_every=1, log_states=True, log_n_entries=400)

    #centre_body="Earth"
    #scale=1.0/1e3
    #xlim=(-1e4, 1e4)
    #ylim=(-1e4, 1e4)
    #xlabel="x [km]"
    #ylabel="y [km]"

    #for states_dict in states: 
    #   plot_static_trajectories(states=states_dict, body_names=bodies[3], colors=bodies[4], scale=scale, centre_body=centre_body, xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)

    #anim = visualize_trajectories(states=states, body_names=bodies[3], colors=bodies[4], interval=20, scale=scale, centre_body=centre_body, xlim=xlim, ylim=ylim, save_path="mcrl.gif")
    #HTML(anim.to_jshtml())

if __name__ == "__main__":
    main()