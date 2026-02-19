#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numba import njit
from line_profiler import LineProfiler, profile


# Integrator interface
class IForceModel:
    def acceleration(self, positions, masses, velocities=None, accelerations=None, t=None):
        raise NotImplementedError


class GravityForce:
    def __init__(self, G=6.67430e-11):
        self.G = G

    @profile
    def acceleration(self, positions, masses):
        return gravity_acceleration_newtonian(positions, masses, self.G)


@njit
def gravity_acceleration_newtonian(positions, masses, G):
        N = len(masses)
        acc = np.zeros_like(positions)

        for i in range(N):
            for j in range(i+1, N):
                r_vec = positions[j] - positions[i]
                r2 = np.dot(r_vec, r_vec)
                inv_r3 = 1.0 / (r2 * np.sqrt(r2))
                a = G * r_vec * inv_r3
                acc[i] += masses[j] * a
                acc[j] -= masses[i] * a
        return acc
    

def acceleration_with_softening(positions, masses, G):
        """
        positions: (N, 2)
        masses: (N,)
        returns: (N, 2)
        """
        N = len(masses)
        acc = np.zeros_like(positions)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec) + 1e-9
                acc[i] += G * masses[j] * r_vec / r**3

        return acc