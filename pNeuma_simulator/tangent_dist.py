import numpy as np
from numba import jit


@jit(nopython=True)
def tangent_dist(theta_i: float, theta_j: float, a_i: float, b_i: float) -> float:
    """
    Calculate the tangent distance between two angles.

    Parameters:
    - theta_i (float): The first angle in radians.
    - theta_j (float): The second angle in radians.
    - a_i (float): The value of a_i.
    - b_i (float): The value of b_i.

    Returns:
    - k_prime (float): The tangent distance between theta_i and theta_j.
    """
    if round(np.cos(theta_j - theta_i), 15) != 0:
        c_prime = np.sin(theta_j - theta_i) / np.cos(theta_j - theta_i)
        d_prime = np.sqrt(b_i**2 + a_i**2 * c_prime**2)
        k_prime = d_prime / np.sqrt(c_prime**2 + 1)
    else:
        k_prime = a_i
    return k_prime
