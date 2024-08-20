import numpy as np


def ov(x: float, lam: float, v0: float, d: float) -> float:
    """
    Calculates the optimal velocity of a vehicle.

    Parameters:
    - x: float or numpy array, representing the position of the vehicle(s).
    - lam: float, representing the slope.
    - v0: float, representing the desired speed.
    - d: float, representing the jam spacing.

    Returns:
    - y: float or numpy array, representing the optimal velocity of the vehicle(s).
    """

    y = np.maximum(0, v0 - v0 * np.exp((-lam / v0) * (x - d)))
    return y


def vo(x: float, lam: float, v0: float, d: float) -> float:
    """
    Calculates the velocity of a vehicle in free flow.

    Parameters:
    - x: float or numpy array, representing the position of the vehicle(s).
    - lam: float, representing the slope.
    - v0: float, representing the desired speed.
    - d: float, representing the jam spacing.

    Returns:
    - y: float or numpy array, representing the velocity of the vehicle(s) in free flow.
    """
    y = (-v0 / lam) * np.log(1 - x / v0) + d
    return y


def f(x: float, lam: float, v0: float, d: float) -> float:
    """
    Calculates the flow of vehicles.

    Parameters:
    - x: float or numpy array, representing the position of the vehicle(s).
    - lam: float, representing the slope.
    - v0: float, representing the desired speed.
    - d: float, representing the jam spacing.

    Returns:
    - y: float or numpy array, representing the flow of vehicles.
    """

    y = np.heaviside((x - d), 1) * (lam * np.exp((-lam / v0) * (x - d)))
    return y


def budget(v: float, lam: list[float], v0: list[float], d: list[float], lengths: list[float], L: float) -> float:
    """
    Calculates the remaining budget for road space allocation.

    Parameters:
    - v: float or numpy array, representing the position of the vehicle(s).
    - lam: list of floats, representing the slopes for different vehicles.
    - v0: list of floats, representing the desired speeds for different vehicles.
    - d: list of floats, representing the jam spacings for different vehicles.
    - lengths: list of floats, representing the lengths of different vehicles.
    - L: float, representing the total road space.

    Returns:
    - budget: float, representing the remaining budget for road space allocation.
    """

    budget = L
    for i, lam_i in enumerate(lam):
        v0_i = v0[i]
        d_i = d[i]
        l_i = lengths[i]
        s_i = vo(v, lam_i, v0_i, d_i)
        budget -= s_i
        budget -= l_i
    return budget
