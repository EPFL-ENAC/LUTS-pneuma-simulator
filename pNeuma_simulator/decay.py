import numpy as np
from numba import jit


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
    phi_max = int(np.exp(param["x1"] * np.norm(vel) * np.factor + param["const"]))
    # half degree resolution
    phi_range = np.linspace(phi_max, -phi_max, 4 * phi_max + 1)
    alphas = np.radians(phi_range) - theta
    return alphas
