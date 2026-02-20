#!/usr/bin/env python
# coding: utf-8

from forcemodel import IForceModel
from integrator import IIntegrator
from ship import IShip, Maneuver, action_from_maneuvers
import numpy as np
import math
from line_profiler import LineProfiler, profile


def make_step_state(m, x, v, a, t):
    return {
        "m": m.copy(),
        "x": x.copy(),
        "v": v.copy(),
        "a": a.copy(),
        "t": t,
    }


class SolarSystemEnv:
    def __init__(self, force_model: IForceModel, integrator: IIntegrator, bodies, ship: IShip | None = None, dt=60.0):
        self.dt = dt
        self.force_model = force_model()
        self.integrator = integrator()
        self.ship = ship

        self.bodies = bodies
        self.n_bodies = len(self.bodies[0])
        dim = 2
        
        self.m = np.zeros((self.n_bodies), dtype=np.float64)  # masses
        self.x = np.zeros((self.n_bodies, dim), dtype=np.float64)  # positions
        self.v = np.zeros((self.n_bodies, dim), dtype=np.float64)  # velocities
        self.a = np.zeros((self.n_bodies, dim), dtype=np.float64)  # accelerations
        self.t = 0.0
        
        self.reset()

    def reset(self):
        self.m = self.bodies[0]
        self.x = self.bodies[1]
        self.v = self.bodies[2]
        self.a = self.force_model.acceleration(self.x, self.m)
        self.t = 0.0

        if self.ship:
            self.ship.reset(self.x, self.v, self.a)
            return self.ship.observe(self.x, self.v, self.a)
        
        return None

    @profile
    def step(self, action: int | None = None):
        if self.ship and action is not None:
            # ship control
            control_acc = self.ship.control_acceleration(self.x, self.v, self.a, action)
            control_idx = self.ship.ship_index
        else:
            control_acc = None
            control_idx = None
    
        self.x, self.v, self.a = self.integrator.step(self.m, self.x, self.v, self.a, self.force_model, self.dt, control_acc=control_acc, control_idx=control_idx)
    
        self.t += self.dt
    
        if self.ship:
            obs = self.ship.observe(self.x, self.v, self.a)
            reward = self.ship.reward(self.x, self.v, self.a) 
            done, success = self.ship.done(self.x, self.v, self.a)
        else:
            obs, reward, done, success = None, None, None, None
        
        return self.x, self.v, self.a, obs, reward, done, success


def run_script_simulation(force_model: IForceModel, integrator: IIntegrator, bodies, ship: IShip | None = None, maneuvers: list[Maneuver] | None = None, dt=180.0, n_steps=100000, records_len=600):
    states = []
    step = max(1, math.floor(n_steps / records_len))

    env = SolarSystemEnv(force_model, integrator, bodies, ship=ship, dt=dt)
    env.reset()

    maneuvers = maneuvers or []

    for i in range(n_steps):
        t = env.t

        action = action_from_maneuvers(t, maneuvers)
        x, v, a, obs, reward, done, success = env.step(action)

        if i % step == 0:
            states.append(make_step_state(env.m, x, v, a, env.t))

        if done:
            break

    return states


def run_simulation(force_model: IForceModel, integrator: IIntegrator, bodies, ship: IShip | None = None, dt=180.0, n_steps=100000, records_len=600):
    states = []
    step = max(1, math.floor(n_steps / records_len))

    env = SolarSystemEnv(force_model, integrator, bodies, ship=ship, dt=dt)
    env.reset()

    for i in range(n_steps):
        x, v, a, obs, reward, done, success = env.step()
        env_state = make_step_state(env.m, x, v, a, env.t)
        
        # record every nth step
        if (i % step == 0):
            states.append(env_state)

    return states

def adjust_barycentric(bodies):
    m = bodies[0].astype(np.float64).copy()
    x = bodies[1].astype(np.float64).copy()
    v = bodies[2].astype(np.float64).copy()
    n = bodies[3].astype(str).copy()
    c = bodies[4].astype(str).copy()
    
    M = np.sum(m)

    # Center of mass position
    R_cm = np.sum(m[:, None] * x, axis=0) / M
    # Shift all positions to barycentric frame
    x -= R_cm
    
    # Total momentum
    P = np.sum(m[:, None] * v, axis=0)
    # Adjust momentum of all bodies to 0 initial momentum
    V_cm = P / M
    v -= V_cm
    
    """
    # Ajust Suns velocity to cancel total momentum
    index_sun = 0
    V_corr = P / m[index_sun]
    P_corr = V_corr * m[index_sun]
    v[index_sun] -= V_corr
    print("Suns P to correctr: ", P_corr)
    print("Suns V to correctr: ", V_corr)

    sum = 0
    for i in range(len(m)):
        P_i = m[i] * v[i]
        print("P_i for ", i, ": ", P_i)
        sum += P_i
    print("P_sum: ", sum)
    """
    
    # Center of Mass
    print("COM position:",np.sum(m[:,None] * x, axis=0) / M)
    # Total Momentum
    print("Total momentum:", np.sum(m[:,None] * v, axis=0))

    barycentric_bodies = []
    barycentric_bodies.append(m)
    barycentric_bodies.append(x)
    barycentric_bodies.append(v)
    barycentric_bodies.append(n)
    barycentric_bodies.append(c)
    
    return barycentric_bodies

