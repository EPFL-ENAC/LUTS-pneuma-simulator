import numpy as np
from numba import jit


@jit(nopython=True)
def direction(theta_i: float) -> tuple:
    """
    Calculate the current direction vector and the normal vector to the current direction.
    Parameters:
    theta_i (float): The angle in radians representing the current direction.
    Returns:
    tuple: A tuple containing the current direction vector (e_i) and the normal vector to the current direction (e_i_n).
    """

    # Current direction vector
    e_i = np.array([np.cos(theta_i), np.sin(theta_i)])
    # Normal vector to current direction
    # https://stackoverflow.com/questions/1243614/
    e_i_n = np.array([np.sin(theta_i), -np.cos(theta_i)])
    return e_i, e_i_n
