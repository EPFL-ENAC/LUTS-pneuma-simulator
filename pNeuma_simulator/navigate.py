import numpy as np

from pNeuma_simulator.collisions import collisions
from pNeuma_simulator.decay import decay
from pNeuma_simulator.neighborhood import neighborhood
from pNeuma_simulator.particle import Particle
from pNeuma_simulator.target import target


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
