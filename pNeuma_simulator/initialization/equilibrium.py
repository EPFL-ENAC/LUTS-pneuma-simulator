# from numpy import concatenate, repeat, cos, pi, isfinite, minimum, all
import numpy as np
from scipy.optimize import root_scalar

from pNeuma_simulator import params
from pNeuma_simulator.initialization import budget, f, vo
from pNeuma_simulator.initialization.synthetic_fd import synthetic_fd


def equilibrium(L: float, lanes: int, n_cars: int, n_moto: int, rng: object, distributed: bool = True) -> tuple:
    """
    Calculate the equilibrium state of a simulation, assuming lane-based equilibrium

    Parameters:
    - L (float): Length of the simulation.
    - lanes (int): Number of lanes in the simulation.
    - n_cars (int): Number of cars in the simulation.
    - n_moto (int): Number of motorcycles in the simulation.
    - rng (object): Random number generator.
    - distributed (bool, optional): Whether the vehicles are distributed or not. Defaults to True.

    Returns:
    - tuple: A tuple for longitundinal dynamics, containing the adaptation time (tau), lambda (lam), initial velocity
    (v0), and distance (d).

    Note:
    - The function assumes that the length (L) is large enough for convergence.
    - The function uses synthetic_fd to generate synthetic fundamental diagrams for cars and motorcycles.
    - The equilibrium speed (v_eq) is calculated using root_scalar.
    - The function checks if the total sum of speeds and distances equals the expected value (lanes * L).
    - The function calculates the adaptation time (tau) based on the equilibrium speeds and fundamental diagrams.
    - The adaptation time is limited by the minimum of 2 / lambda and the calculated tau values.
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
            prefactor = 1 + np.cos(2 * np.pi / n_veh)
            tau = 1 / (prefactor * f_eq)
            if all(np.isfinite(tau)):
                tau = np.minimum(tau, 2 / lam)
                break
    return (tau, lam, v0, d)
