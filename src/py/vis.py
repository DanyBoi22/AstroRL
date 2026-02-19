#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def visualize_trajectories(states, body_names, colors=None, scale=1.0/1.496e11, centre_body=None, trail_length=50, xlim=(-10,10), ylim=(-10,10), interval=50, save_path=None, xlabel="x (scaled)", ylabel="y (scaled)", title="Celestial Trajectories"):
    """
    Visualize a precomputed trajectory of bodies with center on the Sun (0,0).

    states : list of dict
        Each dict must contain 'x' (Nx2 array) and 't' (float).
    body_names : list of str
        Names of bodies (length N).
    colors : list of str
        Colors for bodies.
    scale : float
        Scaling factor for positions.
    centre_body : str
        name of the body from body_names list to centre the visualisation around
    trail_length : int
        How many past positions to show in trail.
    xlim, ylim : tuple
        Plot limits.
    interval : int
        Milliseconds between frames.
    save_path : str or None
        If given, saves animation to file (requires ffmpeg).
    """
    colors = colors
    n_bodies = len(body_names)
    if((centre_body is not None) and (centre_body in body_names)):
        centre_body_index = np.where(body_names == centre_body)[0][0]
    else:
        centre_body_index = None

    # TODO: states preproc: scale and offset calc
    
    # Figure setup
    fig, ax = plt.subplots(figsize=(12,12))
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    #Scatter objects for bodies positions
    scatters = [ax.scatter([], [], color=colors[i], s=80, label=body_names[i]) 
                for i in range(n_bodies)]

    
    # Time text
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                        ha="left", va="top", fontsize=9,
                        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    
    def init():    
        if(centre_body_index is not None):
            centre_body_offset = states[0]['x'][centre_body_index]
        else:
            centre_body_offset = 0
            
        positions = (states[0]['x'] - centre_body_offset) * scale
        i = 0
        for sc in scatters:
            sc.set_offsets([positions[i]])
            i += 1
            
        time_text.set_text(f"t = {states[0]['t']:.1f} s")
        return tuple(scatters) + (time_text,)

    
    def update(frame):
        state = states[frame]
        
        if(centre_body_index is not None):
            centre_body_offset = state['x'][centre_body_index]
        else:
            centre_body_offset = 0
        
        positions = (state['x'] - centre_body_offset) * scale
        t = state['t']

        # Update bodies positions
        for i, name in enumerate(body_names):
            scatters[i].set_offsets([positions[i]])

        # Update time
        time_text.set_text(f"t = {t:.1f} s")
        return tuple(scatters) + (time_text,)

    
    anim = FuncAnimation(fig, update, frames=len(states), init_func=init,
                         blit=False, interval=interval)

    plt.legend()

    if save_path is not None:
        anim.save(save_path, writer="pillow")

    return anim


def plot_static_trajectories(states, body_names, colors=None, scale=1.0/1.496e11, centre_body=None, xlim=(-10,10), ylim=(-10,10), xlabel="x (scaled)", ylabel="y (scaled)", title="Celestial Trajectories", save_path=None):
    """
    Plot static trajectories of celestial bodies.

    states : list of dict
        Each dict contains 'x' (Nx2 array) and 't' (float)
    body_names : list of str
        Names of bodies
    colors : list of str
        Colors for bodies
    scale : float
        Scale factor for positions (e.g., meters -> AU)
    centre_body : str or None
        Body to center the plot on
    xlim, ylim : tuple
        Plot limits
    """
    colors = colors
    n_bodies = len(body_names)

    # Collect positions for each body
    trajectories = [ [] for _ in range(n_bodies) ]
    for state in states:
        x = state['x']
        if centre_body is not None and centre_body in body_names:
            centre_idx = np.where(np.array(body_names) == centre_body)[0][0]
            offset = x[centre_idx]
        else:
            offset = 0
        positions = (x - offset) * scale
        for i in range(n_bodies):
            trajectories[i].append(positions[i].copy())

    # Convert to arrays
    trajectories = [ np.array(traj) for traj in trajectories ]

    # Plot
    fig, ax = plt.subplots(figsize=(12,12))
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    for i, traj in enumerate(trajectories):
        ax.plot(traj[:,0], traj[:,1], color=colors[i], label=body_names[i])

    ax.legend()
    if save_path is not None:
        plt.savefig(save_path) 
    plt.show()
