# AstroRL
This project is meant to explore different reinforcement learning algorithms in orbital mechanics under realistic physical simulation. 
The environment is a custom-made implementation of an N-body simulation. The programming language used is Python, with Jupyter Notebook utilized for visualization and analysis.

## File Structure
- *`gifs/`* - Contains animations of simulation and training runs
- *`logs/`* - Contains log files from simulations and training runs
- *`pics/`* - Contains images used for visualization and analysis
- *`src/`* - Contains all source code
    - *`jupyter/`* - Jupyter notebooks with executable scenarios, visualization, and analysis scripts
    - *`py/`* - Python scripts required for running simulations and training 

## Simulation
The project uses a custom-built N-body orbital mechanics simulator written in Python. The environment models gravitational interactions between multiple bodies under realistic physical dynamics. Numerical integration is applied to update positions, velocities and accelerations over time, allowing stable orbit propagation and physically consistent trajectories. The environment is deterministic and fully physics-based, making it suitable for evaluating RL behavior under realistic constraints.

## Algorithms
The project evaluates two reinforcement learning approaches for continuous control in orbital mechanics: 
- Monte Carlo REINFORCE
- Advantage Actor-Critic

The focus is on comparing learning stability, convergence speed, fuel efficiency, and orbital accuracy under identical physical simulation conditions.

## Quick Start
To explore and understand the simulation environment, open and experiment with the notebook *`astrosim.ipynb`*. This notebook allows you to visualize orbital dynamics and interact directly with the physical simulation.

To run reinforcement learning training, configure the desired parameters in the script *`astrorl.py`* and execute it. The script initializes the environment and starts the training process. Log files are saved in the same directory as the script and can later be analyzed using the Jupyter notebook.
