#!/usr/bin/env python
# coding: utf-8


from forcemodel import IForceModel
from line_profiler import LineProfiler, profile


# Integrator interface
class IIntegrator:
    def step(self, masses, positions, velocities, accelerations, force_model: IForceModel, dt, control_acc=None, control_idx=None):
        raise NotImplementedError


# Verlet integrator
class VelocityVerlet(IIntegrator):
    @profile
    def step(self, masses, positions, velocities, accelerations, force_model: IForceModel, dt, control_acc=None, control_idx=None):
        # add control acceleration at t
        if control_acc is not None and control_idx is not None:
            accelerations[control_idx] += control_acc

        # 1) position update
        x_new = positions + velocities * dt + 0.5 * accelerations * dt**2

        # 2) compute a(t+dt) from gravity
        a_new = force_model.acceleration(x_new, masses)

        # add control acceleration at t+dt
        if control_acc is not None and control_idx is not None:
            a_new[control_idx] += control_acc

        # 3) velocity update
        v_new = velocities + 0.5 * (accelerations + a_new) * dt

        positions = x_new
        velocities = v_new
        accelerations = a_new
        
        return positions, velocities, accelerations
    """
    OLD CODE WITH NO THRUST CONTROL
    def step(self, masses, positions, velocities, accelerations, force_model: IForceModel, dt, control_acc=None, control_idx=None):
        # 1) position update
        x_new = positions + velocities*dt + accelerations*(0.5*dt**2)
        # 2) acceleration at NEW positions
        a_new = force_model.acceleration(x_new, masses)
        # 3) velocity update
        v_new = velocities + (accelerations + a_new) * (0.5*dt)
        
        positions = x_new
        velocities = v_new
        accelerations = a_new
        
        return positions, velocities, accelerations
    """


class SymplecticEuler(IIntegrator):
    @profile
    def step(self, masses, positions, velocities, accelerations, force_model: IForceModel, dt, control_acc=None, control_idx=None):
        a_new = force_model.acceleration(positions, masses)
        v_new = velocities + a_new * dt
        x_new = positions + v_new * dt

        positions = x_new
        velocities = v_new
        accelerations = a_new

        return positions, velocities, accelerations
