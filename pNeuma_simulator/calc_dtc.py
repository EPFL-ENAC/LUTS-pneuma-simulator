import numpy as np
from numba import jit

from pNeuma_simulator.ellipses import ellipses


@jit(nopython=True)
def calc_dtc(
    l_j: float,
    w_j: float,
    l_i: float,
    w_i: float,
    x_j: float,
    y_j: float,
    x_i: float,
    y_i: float,
    theta_j: float,
    theta_i: float,
) -> float:
    """
    Calculates the distance to closest approach (DTC) between two objects.
    Parameters:
    l_j (float): Length of object j.
    w_j (float): Width of object j.
    l_i (float): Length of object i.
    w_i (float): Width of object i.
    x_j (float): x-coordinate of object j.
    y_j (float): y-coordinate of object j.
    x_i (float): x-coordinate of object i.
    y_i (float): y-coordinate of object i.
    theta_j (float): Orientation angle of object j.
    theta_i (float): Orientation angle of object i.
    Returns:
    float: The distance to closest approach (DTC) between the two objects.
    """

    # distance from i to j
    s_i_j = np.sqrt((x_j - x_i) ** 2 + (y_j - y_i) ** 2)
    # distance of closest approach between i and j
    min_d = ellipses(l_j, w_j, l_i, w_i, x_j, y_j, x_i, y_i, theta_j, theta_i)
    dtc = s_i_j - min_d
    return dtc
