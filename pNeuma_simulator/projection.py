import numpy as np
from numba import jit


@jit(nopython=True)
def projection(e_i_n, e_i_j, s_i_j: float) -> float:
    """
    Calculates the projection of vector e_i_j onto vector e_i_n.
    Parameters:
    e_i_n (array-like): The vector onto which the projection is made.
    e_i_j (array-like): The vector being projected.
    s_i_j (float): The magnitude of vector e_i_j.
    Returns:
    float: The projection of vector e_i_j onto vector e_i_n.
    """

    proj = abs(np.dot(e_i_n, e_i_j)) * s_i_j
    return proj
