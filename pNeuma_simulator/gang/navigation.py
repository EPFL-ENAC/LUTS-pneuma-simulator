from math import exp

import numpy as np
from numba import jit
from numpy import argmin, argwhere, array
from scipy.signal import find_peaks

from pNeuma_simulator import params
from pNeuma_simulator.gang import collisions
from pNeuma_simulator.gang.neighborhood import neighborhood
from pNeuma_simulator.gang.particle import Particle


def navigate(ego: Particle, agents: list[Particle]) -> tuple:
    """Anticipatory operational navigation

    Args:
        ego (Particle): ego vehicle to be updated
        agents (list): list of agents in the simulation

    Returns:
        tuple: target direction in radians,
        corresponding distance to collision in meters and
        time to collision in seconds.
    """
    neighbors = neighborhood(ego, agents)
    if ego.mode == "Moto":
        # exponential decay of deviations
        alphas = decay(ego.speed, ego.theta)
        # compute distance to collision
        f_a = []
        v_0 = ego.v0
        for alpha in alphas:
            ttc = collisions(ego, v_0, alpha + ego.theta, neighbors)
            if ttc:
                f = ttc * v_0
            else:
                f = params.d_max
            f_a.append(min([f, params.d_max]))
        f_a = array(f_a)
        # maximize distance to collision
        optima = find_peaks(f_a)[0]
        peaks = array(f_a)[optima]
        maxima = argwhere(peaks == max(peaks)).flatten()
        indices = optima[maxima]
        if len(indices) > 1:
            # get the most central optimum (less deviation)
            optimum = indices[argmin(abs(alphas[indices]))]
        else:
            optimum = indices[0]
        # desired direction
        a0 = alphas[optimum]
    else:
        a0 = 0
        f_a = None
    # actual time to collision
    ttc = collisions(ego, ego.speed, ego.theta, neighbors)
    # store the result
    return (a0, f_a, ttc)


@jit(nopython=True)
def decay(speed: float, theta: float) -> np.ndarray:
    """
    Calculate the choice set for a given speed and angle.

    Args:
        speed (float): The speed value.
        theta (float): The angle value.

    Returns:
        alphas (ndarray): An array of angles in radians.
    """
    gamma_max = round(exp(params.XM * speed * params.factor + params.CM) / params.da) * params.da
    # angular resolution in the reference system of the road
    gamma = np.linspace(gamma_max, -gamma_max, int(2 * gamma_max / params.da) + 1)
    # in the reference system of the agent
    alphas = np.radians(gamma) - theta
    return alphas
