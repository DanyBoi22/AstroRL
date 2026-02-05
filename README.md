# AstroRL
This project is ment to explore different RL algorithms for calculation optimal orbits between celestial bodies.

The simulation for the celestial bodies is a homemade implementation of N-Body simulation. The mprogramming language is Python utilizing Jupyter-Notebook.

## Simulation
### Problems
- A single timestep for all bodies. But a fine timesteps are crucial for very fast, very close moving objects. For better accuracy need to implement relative timestep depending on period or speed of an object.

- Better initial configuration. At the moment the intial simulation parameters assume all bodies are aligned at t=0 and their movement vector is perpendicular to the alignemnt. The numbers are textbook numbers. The problem is that he textbook numbers use different inertial systems. For example for correct real life geo orbits the initial parameters must be corrected for earths frame of reference.