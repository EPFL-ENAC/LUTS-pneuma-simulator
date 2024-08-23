import numpy as np
from numba import jit
from numpy.linalg import norm


@jit(nopython=True)
def infront(e_i, pos_i, pos_j):
    """
    Determines if a neighbor is in front of a given position.

    Args:
        e_i: The direction vector of the current position.
        pos_i: The current position.
        pos_j: The position of the neighbor.

    Returns:
        front: A boolean indicating if the neighbor is in front of the current position.
        e_i_j: The unit vector from the current position to the neighbor.
        s_i_j: The distance from the current position to the neighbor.
    """

    # Distance from i to j
    s_i_j = norm(pos_j - pos_i)
    # Unit vector from i to j
    e_i_j = (pos_j - pos_i) / s_i_j
    # Check if neighbor is in front
    front = np.dot(e_i, e_i_j) > 0
    return front, e_i_j, s_i_j
