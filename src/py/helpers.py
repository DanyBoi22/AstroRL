#!/usr/bin/env python
# coding: utf-8

import numpy as np
import json
import os

# Some helpers
DAY   = 24 * 3600.0
YEAR  = 365.25 * DAY
AU    = 1.496e11

ORBITAL_PERIODS = {
    # Inner planets
    "Mercury": 87.969 * DAY,
    "Venus":   224.701 * DAY,
    "Earth":   YEAR,
    "Moon":    27.321661 * DAY,
    "Mars":    686.98 * DAY,

    # Asteroid belt reference (optional)
    "Ceres":   1680.0 * DAY,

    # Jupiter system
    "Jupiter": 11.862 * YEAR,
    "Io":      1.769137786 * DAY,
    "Europa":  3.551181 * DAY,
    "Ganymede":7.154553 * DAY,
    "Callisto":16.6890184 * DAY,

    # Saturn system
    "Saturn":  29.457 * YEAR,
    "Titan":   15.945421 * DAY,
    "Rhea":    4.518212 * DAY,
    "Iapetus": 79.3215 * DAY,
    "Dione":   2.736915 * DAY,
    "Tethys":  1.887802 * DAY,
    "Enceladus":1.370218 * DAY,

    # Uranus system
    "Uranus":  84.0205 * YEAR,
    "Titania": 8.705872 * DAY,
    "Oberon":  13.463234 * DAY,
    "Umbriel": 4.144177 * DAY,
    "Ariel":   2.520379 * DAY,
    "Miranda": 1.413479 * DAY,

    # Neptune system
    "Neptune": 164.8 * YEAR,
    "Triton":  5.876854 * DAY,
    "Nereid":  360.1362 * DAY,

    # Dwarf planets
    "Pluto":   248.00 * YEAR,
    "Charon":  6.387230 * DAY,
}

def steps_for_duration(duration_s, dt):
    """
    Number of integration steps for a given physical duration.
    """
    return int(np.ceil(duration_s / dt))

def steps_for_orbits(orbital_period_s, n_orbits, dt):
    """
    orbital_period_s: period of the body (seconds)
    n_orbits: how many full revolutions
    """
    total_time = orbital_period_s * n_orbits
    return int(np.ceil(total_time / dt))

def dt_from_fastest_orbit(orbital_periods_s, samples_per_orbit=2000):
    """
    orbital_periods_s: iterable of orbital periods (seconds)
    samples_per_orbit: 50 (bare minimum), 100+ recommended
    """
    return orbital_periods_s / samples_per_orbit

def compute_dt_from_perihelion(masses, positions, central_body_index=0, safety_factor=0.01, G=6.67430e-11):
    """
    Compute a stable timestep based on perihelion curvature
    for Newtonian gravity with Velocity Verlet.

    Parameters
    ----------
    masses : (N,) ndarray
        Masses of all bodies [kg].
    positions : (N, 2) ndarray
        Current positions of bodies [m].
    central_body_index : int
        Index of dominant mass (Sun, Jupiter, etc.).
    safety_factor : float
        Stability factor (0.005–0.02 recommended).

    Returns
    -------
    dt : float
        Stable timestep in seconds.
    """

    M_c = masses[central_body_index]
    r_c = positions[central_body_index]

    # Distances from central body
    r = np.linalg.norm(positions - r_c, axis=1)

    # Ignore the central body itself
    r = np.delete(r, central_body_index)

    # Smallest perihelion distance
    r_min = np.min(r)

    # Local dynamical timescale
    tau = np.sqrt(r_min**3 / (G * M_c))

    # Stable timestep
    dt = safety_factor * tau

    return dt

def to_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    return obj

def save_states(path, states):
    with open(path, "w") as f:
        json.dump(to_json_safe(states), f)


def load_states(path):
    with open(path, "r") as f:
        data = json.load(f)

    def convert_state(s):
        if isinstance(s, dict):
            if "x" in s:
                s["x"] = np.array(s["x"])
            if "v" in s:
                s["v"] = np.array(s["v"])
            if "a" in s:
                s["a"] = np.array(s["a"])
            if "m" in s:
                s["m"] = np.array(s["m"])
        return s

    # Case 1: dict containing list of states
    if isinstance(data, dict) and "states" in data:
        states = data["states"]
    else:
        states = data

    # Case 2: list of states
    if isinstance(states, list):

        # nested list (e.g. episodes → states)
        if len(states) > 0 and isinstance(states[0], list):
            return [[convert_state(s) for s in episode] for episode in states]

        # flat list
        return [convert_state(s) for s in states]

    # Case 3: single state
    return convert_state(states)

def save_loglist(path, loglist):
    with open(path, "w") as f:
        json.dump(loglist, f)

def load_loglist(path):
    with open(path, "r") as f:
        loglist = json.load(f)
    return loglist

def append_to_json_list(path, item):
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(item)

    with open(path, "w") as f:
        json.dump(data, f)


def append_batch_to_json_list(path, items):
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.extend(items)

    with open(path, "w") as f:
        json.dump(data, f)