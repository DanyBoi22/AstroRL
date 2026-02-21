#!/usr/bin/env python
# coding: utf-8

from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass


class IShip(ABC):
    """
    Interface for controllable spacecraft.
    """

    @abstractmethod
    def reset(self, x, v, a):
        """
        Called on environment reset.
        Can initialize internal state.

        x: array of position vectors for every body in sim
        v: array of velocity vectors for every body in sim
        a: array of acceleration vectors for every body in sim
        """
        pass

    @abstractmethod
    def control_acceleration(self, x, v, a, action):
        """
        Returns acceleration vector to be added to physics.
        Shape: (2,) or (dim,)

        x: array of position vectors for every body in sim
        v: array of velocity vectors for every body in sim
        a: array of acceleration vectors for every body in sim
        """
        pass

    @abstractmethod
    def observe(self, x, v, a):
        """
        Returns observation vector for RL.

        x: array of position vectors for every body in sim
        v: array of velocity vectors for every body in sim
        a: array of acceleration vectors for every body in sim
        """
        pass

    @abstractmethod
    def reward(self, x, v, a):
        """
        Returns scalar reward.

        x: array of position vectors for every body in sim
        v: array of velocity vectors for every body in sim
        a: array of acceleration vectors for every body in sim
        """
        pass

    @abstractmethod
    def done(self,  x, v, a):
        """
        Returns termination flag.

        x: array of position vectors for every body in sim
        v: array of velocity vectors for every body in sim
        a: array of acceleration vectors for every body in sim
        """
        pass

    @abstractmethod
    def done(self,  x, v, a, max_steps, step_count):
        """
        Returns termination flag.

        x: array of position vectors for every body in sim
        v: array of velocity vectors for every body in sim
        a: array of acceleration vectors for every body in sim
        """
        pass

class Actions:
    def __init__(self, actions: dict[int, np.ndarray]):
        """
        actions: {action_id: direction_vector}
        """
        self.actions = actions
        self.n = len(actions)

    def sample(self):
        return np.random.randint(0, self.n)

    def vector(self, action: int):
        return self.actions[action]


@dataclass
class Maneuver:
    start: float        # seconds
    duration: float     # seconds
    action: int         # discrete action (from Actions)


def action_from_maneuvers(t: float, maneuvers: list[Maneuver]) -> int | None:
    for m in maneuvers:
        if m.start <= t < m.start + m.duration:
            return m.action
    return None


simple_action_space = Actions({
    0: np.array([0.0, 0.0]),
    1: np.array([1.0, 0.0]),
    2: np.array([-1.0, 0.0]),
    3: np.array([0.0, 1.0]),
    4: np.array([0.0, -1.0]),
})


class SimpleImpulseShip(IShip):
    def __init__(self,
                 ship_index: int, mass: float, thrust: float,
                 actions: Actions, reward_coef: float | None = None, 
                 safety_radius: float | None = None, escape_dist: float | None = None,  escape_vel: float | None = None,
                 target_dist: float | None = None, target_vel: float | None = None, reference_point_index: int | None = None):
        # Spacecraft should always have last index
        self.ship_index = ship_index
        self.mass = mass if mass > 0 else 0.00000001
        self.actions = actions
        self.done_flag = False
        self.success_flag = False

        self.last_action = None
        self.prev_rel_dist = None
        self.prev_rel_vel = None
        self.rel_dist = None
        self.rel_vel = None
        self.dist_norm = None
        self.vel_norm = None
        self.radial_alignment = None

        if escape_dist is not None:
            self.escape_dist = escape_dist if escape_dist > 0 else 1e9
        else:
            self.escape_dist = 1e9
        if escape_vel is not None:
            self.escape_vel = escape_vel if escape_vel > 0 else 1e5
        else:
            self.escape_vel = 1e5
            
        self.reference_point_index = reference_point_index
        
        if target_dist is not None:
            self.target_dist = target_dist if target_dist > 0 else 1e-3
        if target_vel is not None:
            self.target_vel = target_vel if target_vel > 0 else 1e-3
        
        self.thrust = thrust if thrust >= 0 else 0
        
        if safety_radius is not None:
            self.safety_radius = safety_radius if safety_radius >= 0 else 0

        if reward_coef is not None:
            self.reward_coef = reward_coef
        else:
            self.reward_coef = 1.0
        
    
    def reset(self, x, v, a):
        self.done_flag = False
        self.success_flag = False
        self.last_action = None
        self.prev_rel_dist = None
        self.prev_rel_vel = None
        self.rel_dist = None
        self.rel_vel = None
        self.dist_norm = None
        self.vel_norm = None
        self.radial_alignment = None

    
    def control_acceleration(self, x, v, a, action: int):
        self.last_action = action
        direction = self.actions.vector(action)
        return (self.thrust / self.mass) * direction

    def observe(self, x, v, a):
        if self.reference_point_index is None:
            return np.concatenate([x[self.ship_index], v[self.ship_index], np.array([np.linalg.norm(x[self.ship_index])]), np.array([np.linalg.norm(v[self.ship_index])]), np.array([0])])
        
        rel_x_vec = x[self.ship_index] - x[self.reference_point_index] # relative position vector
        rel_v_vec = v[self.ship_index] - v[self.reference_point_index] # relative velocity vector

        self.rel_dist = np.linalg.norm(rel_x_vec) # relative distance 
        self.rel_vel = np.linalg.norm(rel_v_vec) # relative speed

        x_norm = rel_x_vec / (self.rel_dist + 1e-8) # normalised relative position vector
        v_norm = rel_v_vec / (self.rel_vel + 1e-8) # normalised relative velocity vector

        self.dist_norm = (self.rel_dist - self.target_dist) / (self.target_dist + 1e-8) # normalised relative distance 
        self.vel_norm  = (self.rel_vel - self.target_vel) / (self.target_vel + 1e-8) # normalised relative speed

        self.radial_alignment = np.dot(v_norm, x_norm) # 0 perpendicular, 1 moving out, -1 mowing in
        
        # initialize memory
        if self.prev_rel_dist is None:
            self.prev_rel_dist = self.rel_dist
        if self.prev_rel_vel is None:
            self.prev_rel_vel = self.rel_vel
    
        return np.concatenate([x_norm, v_norm, np.array([self.dist_norm]), np.array([self.vel_norm]), np.array([self.radial_alignment])])

    def reward(self, x, v, a):    
        #self.reward_coef: total episode reward magnitude ≈ O(1)  so 1.0 / self.max_steps

        dist_error = abs(self.dist_norm)
        vel_error  = abs(self.vel_norm)
        rad_mag = abs(self.radial_alignment)  # 0 perpendicular, 1 moving out or mowing in
    
        r = 0.0

        # Tangentiality reward
        r += 1.0 * (1- rad_mag) 
        
        # Time pressure (very small, always on)
        r -= 0.2

        # Distance shaping
        r += 2.0 * (1- dist_error)

        #dist_progress = (self.prev_rel_dist - self.rel_dist) / (self.rel_dist + 1e-8)
        #r += 1.0 * dist_progress
        #self.prev_rel_dist = self.rel_dist
    
        # Velocity  
        r += 0.3 * (1- vel_error)

        #vel_progress = (self.prev_rel_vel - self.rel_vel) / (self.rel_vel + 1e-8)
        #r += 0.1 * vel_progress
        #self.prev_rel_vel = self.rel_vel
    
        # Thrust / acceleration penalty
        if self.last_action != 0:
            r -= 0.2

        # Apply reward normalisation
        r = r * self.reward_coef

        # Terminal conditions
        # success
        if dist_error < 0.05 and vel_error < 0.05 and rad_mag < 0.01:
            r += 10.0
            self.done_flag = True
            self.success_flag = True
            print("Success: Ship reached the target")
            return r
    
        # crash
        if self.rel_dist < self.safety_radius:
            r -= 4.0
            self.done_flag = True
            print("Failure: Ship crashed")
            return r
    
        # escape (distance OR velocity)
        if self.rel_dist > self.escape_dist or self.rel_vel > self.escape_vel:
            r -= 4.0
            self.done_flag = True
            print("Failure: Ship escaped the system or exceeded system escape velocity")
            return r
    
        return r
    
    def done(self, x, v, a):
        return self.done_flag, self.success_flag