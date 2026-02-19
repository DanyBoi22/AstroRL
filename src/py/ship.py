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

        self.last_action = None
        self.prev_rel_x = None
        self.prev_rel_v = None
        self.rel_x = None
        self.rel_v = None
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
        self.last_action = None
        self.prev_rel_x = None
        self.prev_rel_v = None
        self.rel_x = None
        self.rel_v = None
        self.radial_alignment = None

    
    def control_acceleration(self, x, v, a, action: int):
        self.last_action = action
        direction = self.actions.vector(action)
        return (self.thrust / self.mass) * direction

    def observe(self, x, v, a):
        if self.reference_point_index is None:
            return np.concatenate([x[self.ship_index], v[self.ship_index], np.array([np.linalg.norm(x[self.ship_index])]), np.array([np.linalg.norm(v[self.ship_index])]), np.array([0])])
        
        rel_x_vec = x[self.ship_index] - x[self.reference_point_index] # relative distance vector
        rel_v_vec = v[self.ship_index] - v[self.reference_point_index] # relative velocity vector

        self.rel_x = np.linalg.norm(rel_x_vec) # relative distance 
        self.rel_v = np.linalg.norm(rel_v_vec) # relative velocity magnitude

        x_norm = rel_x_vec / (self.rel_x + 1e-8) # normalised 
        v_norm = rel_v_vec / (self.rel_v + 1e-8) # normalised 

        self.radial_alignment = np.dot(v_norm, x_norm) # 0 perpendicular, 1 moving out, -1 mowing in
        
        # initialize memory
        if self.prev_rel_x is None:
            self.prev_rel_x = self.rel_x
        if self.prev_rel_v is None:
            self.prev_rel_v = self.rel_v
    
        return np.concatenate([x_norm, v_norm, np.array([self.rel_x]), np.array([self.rel_v]), np.array([self.radial_alignment])])

    def reward(self, x, v, a):    
        #self.reward_coef: total episode reward magnitude ≈ O(1)  so 1.0 / self.max_steps

        dist_error = abs(self.rel_x - self.target_dist) / (self.target_dist + 1e-8)
        vel_error  = abs(self.rel_v - self.target_vel) / (self.target_vel + 1e-8)
    
        r = 0.0

        # Tangentiality reward
        rad_mag = abs(self.radial_alignment)
        tangential_reward = 1.0 - rad_mag
        r += 1.0 * tangential_reward * self.reward_coef # 0 perpendicular, 1 moving out, -1 mowing in
        
        # Time pressure (very small, always on)
        r -= 0.2 * self.reward_coef

        # Distance shaping
        dist_reward = 1.0 / (1.0 + dist_error)
        r += 2.0 * dist_reward * self.reward_coef

        #dist_progress = self.prev_rel_x - self.rel_x
        #r += 1.0 * dist_progress * self.reward_coef
        #self.prev_rel_x = self.rel_x
    
        # Velocity  
        #vel_reward = 1.0 / (1.0 + vel_error)
        #r += 0.3 * vel_reward * self.reward_coef

        #vel_progress = self.prev_rel_v - self.rel_v
        #r += 0.1 * vel_progress * self.reward_coef
        #self.prev_rel_v = self.rel_v
    
        # Thrust / acceleration penalty
        if self.last_action != 0:
            r -= 0.2 * self.reward_coef
    
        # Terminal conditions
        # success
        if dist_error < 0.05 and vel_error < 0.1 and rad_mag < 0.1:
            r += 10.0
            self.done_flag = True
            print("Success: Ship reached the target")
            return r
    
        # crash
        if self.rel_x < self.safety_radius:
            r -= 2.0
            self.done_flag = True
            print("Failure: Ship crashed")
            return r
    
        # escape (distance OR velocity)
        if self.rel_x > self.escape_dist or self.rel_v > self.escape_vel:
            r -= 2.0
            self.done_flag = True
            print("Failure: Ship escaped the system or exceeded system escape velocity")
            return r
    
        return r
    
    def done(self, x, v, a):
        return self.done_flag

