from math import cos, pi

import numpy as np
from scipy.optimize import root_scalar
from scipy.stats import distributions

from pNeuma_simulator import params
from pNeuma_simulator.initialization import budget, f, vo
from pNeuma_simulator.utils import truncated_rvs


def synthetic_fd(n_veh: int, random_state, mode: str = "Car", distributed: bool = True):
    """
    Generate synthetic fundamental diagram data.

    Args:
        n_veh (int): Number of vehicles.
        random_state (object): Random number generator.
        mode (str, optional): Mode of transportation. Defaults to "Car".
        distributed (bool, optional): Flag indicating if the data should be distributed. Defaults to True.

    Returns:
        tuple: Tuple containing the following:
            - list: List of marginal distributions.
            - numpy.ndarray: Array of lambda values.
            - numpy.ndarray: Array of desired speeds.
            - numpy.ndarray: Array of jam spacings.
    """
    factor = params.factor
    lam, v0, d = 0, 0, 0
    marginals = []
    marginal_dists = [
        distributions.fisk,
        distributions.lognorm,
        distributions.mielke,
    ]
    if mode == "Car":
        args = params.car_args
        bounds = params.car_bounds
    else:
        args = params.moto_args
        bounds = params.moto_bounds
    for r, marginal_dist in enumerate(marginal_dists):
        marginal = marginal_dist(*args[r])
        marginals.append(marginal)
    if distributed:
        synthetic_data = []
        for i in range(3):
            rng = random_state
            size = n_veh
            dist = marginal_dists[i]
            x_min, x_max = bounds[i]
            if len(args[i]) == 3:
                s, loc, scale = args[i]
                k = None
            else:
                k, s, loc, scale = args[i]
            values = truncated_rvs(rng, size, dist, x_min, x_max, k, s, loc, scale)
            synthetic_data.append(values)
        # slope at jam spacing s^-1
        lam = synthetic_data[0]
        # desired speed in m/s
        v0 = synthetic_data[1] / factor
        # jam spacing m
        d = synthetic_data[2].copy()
    else:
        lam = np.repeat(marginals[0].mean(), n_veh)
        v0 = np.repeat(marginals[1].mean() / factor, n_veh)
        d = np.repeat(marginals[2].mean(), n_veh)

    return marginals, lam, v0, d


def equilibrium(L: float, lanes: int, n_cars: int, n_moto: int, rng: object, distributed: bool = True) -> tuple:
    """
    Calculate the equilibrium state of a simulation, assuming lane-based equilibrium

    Args:
        L (float): Length of the simulation.
        lanes (int): Number of lanes in the simulation.
        n_cars (int): Number of cars in the simulation.
        n_moto (int): Number of motorcycles in the simulation.
        rng (object): Random number generator.
        distributed (bool, optional): Whether the vehicles are distributed or not. Defaults to True.

    Returns:
        tuple: A tuple for longitundinal dynamics, containing the adaptation time (tau), lambda (lam), initial velocity
        (v0), and distance (d).

    Note:
    The function assumes that the length (L) is large enough for convergence.
    The function uses synthetic_fd to generate synthetic fundamental diagrams for cars and motorcycles.
    The equilibrium speed (v_eq) is calculated using root_scalar.
    The function checks if the total sum of speeds and distances equals the expected value (lanes * L).
    The function calculates the adaptation time (tau) based on the equilibrium speeds and fundamental diagrams.
    The adaptation time is limited by the minimum of 2 / lambda and the calculated tau values.
    """
    while True:
        # Caution !!! If L is too small this will never converge !!!
        n_veh = lanes * n_cars + n_moto
        _, c_lam, c_v0, c_d = synthetic_fd(n_cars * lanes, rng, mode="Car", distributed=distributed)
        _, m_lam, m_v0, m_d = synthetic_fd(n_moto, rng, mode="Moto", distributed=distributed)
        lam = np.concatenate((c_lam, m_lam))
        v0 = np.concatenate((c_v0, m_v0))
        d = np.concatenate((c_d, m_d))
        lengths = np.concatenate((np.repeat(params.c_l, lanes * n_cars), np.repeat(params.m_l, n_moto)))

        def g(x):
            return budget(x, lam, v0, d, lengths, lanes * L)

        try:
            sol = root_scalar(g, bracket=[0, min(v0)])
        except ValueError:
            continue
        # Equilibrium speed
        v_eq = sol.root
        s_eq = vo(v_eq, lam, v0, d)
        total = round(np.sum(s_eq) + np.sum(lengths), 3)
        if total == lanes * L:
            f_eq = f(s_eq, lam, v0, d)
            # Adaptation time
            prefactor = 1 + cos(2 * pi / n_veh)
            tau = 1 / (prefactor * f_eq)
            if all(np.isfinite(tau)):
                tau = np.minimum(tau, 1 / lam)
                break
    return (tau, lam, v0, d)
