import numpy as np
from numpy.typing import ArrayLike


def budget(v: float, lam: list[float], v0: list[float], s0: list[float], lengths: list[float], L: float) -> float:
    """
    Calculates the remaining budget for road space allocation.

    Args:
        v: float or numpy array, representing the position of the vehicle(s).
        lam: list of floats, representing the slopes for different vehicles.
        v0: list of floats, representing the desired speeds for different vehicles.
        s0: list of floats, representing the jam spacings for different vehicles.
        lengths: list of floats, representing the lengths of different vehicles.
        L: float, representing the total road space.

    Returns:
        budget: float, representing the remaining budget for road space allocation.
    """

    budget = L
    for i, lam_i in enumerate(lam):
        v0_i = v0[i]
        s0_i = s0[i]
        l_i = lengths[i]
        s_i = vo(v, lam_i, v0_i, s0_i)
        budget -= s_i
        budget -= l_i
    return budget


def f(x: float, lam: float, v0: float, s0: float) -> float:
    """
    Calculates the derivate of the OV function.

    Args:
        x: float or numpy array, representing vehicle spacing.
        lam: float, representing the slope.
        v0: float, representing the desired speed.
        s0: float, representing the jam spacing.

    Returns:
        y: float or numpy array, representing the derivative.
    """

    y = np.heaviside((x - s0), 1) * (lam * np.exp((-lam / v0) * (x - s0)))
    return y


def ov(x: ArrayLike, lam: ArrayLike, v0: ArrayLike, s0: ArrayLike) -> ArrayLike:
    """
    Calculates the optimal velocity of a vehicle.

    Args:
        x: float or numpy array, representing vehicle spacing.
        lam: float, representing the slope.
        v0: float, representing the desired speed.
        s0: float, representing the jam spacing.

    Returns:
        y: float or numpy array, representing the optimal velocity.
    """

    y = np.maximum(0, v0 - v0 * np.exp((-lam / v0) * (x - s0)))
    return y


def vo(x: float, lam: float, v0: float, s0: float) -> float:
    """
    Calculates the velocity of a vehicle in free flow.

    Parameters:
        x: float or numpy array, representing vehicle velocity.
        lam: float, representing the slope.
        v0: float, representing the desired speed.
        s0: float, representing the jam spacing.

    Returns:
        y: float or numpy array, representing optimal spacing.
    """
    # https://stackoverflow.com/questions/21610198/
    with np.errstate(invalid="ignore"):
        arg = 1 - x / v0
        y = (-v0 / lam) * np.log(arg, where=arg > 0) + s0
    return y
