import numpy as np
from numba import jit
from numpy import where
from numpy.linalg import norm

from pNeuma_simulator import params
from pNeuma_simulator.gang import collisions
from pNeuma_simulator.gang.neighborhood import neighborhood
from pNeuma_simulator.gang.particle import Particle


def navigate(ego: Particle, agents: list[Particle], integer: int, d_max) -> tuple:
    """Anticipatory operational navigation

    Args:
        ego (Particle): ego vehicle to be updated
        agents (list): list of agents in the simulation
        integer (int): random seed
        d_max (float, optional): horizon distance.

    Returns:
        tuple: target and desired directions in radians,
        angle choice set in radians,
        corresponding distance to collision in meters and
        time to collision in seconds.
    """
    rng = np.random.default_rng(integer)
    neighbors = neighborhood(ego, agents)
    if ego.mode == "Moto":
        # exponential decay of deviations
        alphas = decay(ego.vel, ego.theta)
        # compute distance to collision
        f_a = []
        v_0 = ego.v0
        for alpha in alphas:
            ttc = collisions(ego, v_0, alpha + ego.theta, neighbors)
            if ttc:
                f = ttc * v_0
            else:
                f = d_max
            f_a.append(min([f, d_max]))
        f_a = np.array(f_a)
        # target and desired direction
        a0, a_des = target(alphas, f_a, d_max, rng)
    else:
        a0, a_des = 0, 0
        alphas, f_a = None, None
    # actual time to collision
    ttc = collisions(ego, np.linalg.norm(ego.vel), ego.theta, neighbors)
    # store the result
    return (a0, a_des, alphas, f_a, ttc)


def egress(alphas, indices, counts, cond, rng) -> float:
    """Find interior egress point in the horizon.

    Args:
        alphas (list): angle choice set in radians.
        indices (list): indices of non zero intervals.
        counts (list): difference between consecutive indices.
        cond (list): condition, boolean list.

    Returns:
        float: egreess point in radians
    """
    supremum = max(counts[cond])
    argmax = np.argwhere(counts[cond] == supremum)
    start = indices[cond][rng.choice(argmax)]
    a0 = alphas[start + np.round((supremum - 1) / 2).astype(int)][0]
    return a0


def target(alphas, f_a, d_max, rng) -> tuple:
    """Find consecutive runs and length of runs with condition.

    Args:
        alphas (list): angle choice set in radians.
        f_a (list): distance to collision values in meters.
        d_max (int): distance to horizon in meters.

    Returns:
        tuple: target and desired directions in radians
    """
    # Find consecutive runs and length of runs with condition
    # https://stackoverflow.com/questions/71746585
    # set equal, consecutive elements to 0
    intervals = np.hstack([True, ~np.isclose(f_a[:-1], f_a[1:])])
    # get indices of non zero elements
    indices = np.flatnonzero(intervals)
    # difference between consecutive indices are the length
    counts = np.diff(indices, append=len(f_a))
    cond = (counts > 1) & (f_a[indices] == max(f_a))
    # determine a0
    if len(counts[cond]) > 0:
        # check for saturated flanks
        if (f_a[0] == d_max) | (f_a[-1] == d_max):
            # check left flank
            if f_a[0] == d_max:
                cond[0] = False
            # check right flank
            if f_a[-1] == d_max:
                cond[-1] = False
            # pick saturated interior point
            if len(counts[cond]) > 0:
                a0 = egress(alphas, indices, counts, cond, rng)
            # keep steady when only the flanks are saturated
            else:
                a0 = 0
        else:
            a0 = egress(alphas, indices, counts, cond, rng)
    else:
        maxima = where(f_a == f_a.max())[0]
        if len(maxima) == 1:
            maximum = maxima[0]
        else:
            maximum = rng.choice(maxima)
        a0 = alphas[maximum]
    # Objective: squared distance to destination (for performance)
    d_a = d_max**2 + f_a**2 - 2 * d_max * f_a * np.cos(a0 - alphas)
    # minimize distance to destination
    minima = where(d_a == d_a.min())[0]
    if len(minima) == 1:
        minimum = minima[0]
    else:
        minimum = rng.choice(minima)
    # chosen direction
    a_des = alphas[minimum]
    return (a0, a_des)


@jit(nopython=True)
def decay(vel: float, theta: float) -> np.ndarray:
    """
    Calculate the decay angles for a given velocity and angle.
    Parameters:
    - vel (float): The velocity value.
    - theta (float): The angle value.
    Returns:
    - alphas (numpy.ndarray): An array of decay angles.
    """

    param = {"const": 4.026397, "x1": -0.062306}
    phi_max = int(np.exp(param["x1"] * norm(vel) * params.factor + param["const"]))
    # half degree resolution
    phi_range = np.linspace(phi_max, -phi_max, 4 * phi_max + 1)
    alphas = np.radians(phi_range) - theta
    return alphas
