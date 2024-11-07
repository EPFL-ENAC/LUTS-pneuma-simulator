import numpy as np
from numba import jit

from pNeuma_simulator.gang.particle import Particle


@jit(nopython=True)
def direction(theta_i: float) -> tuple:
    """
    Calculate the current direction vector and the normal vector to the current direction.

    Args:
        theta_i (float): The angle in radians representing the current direction.

    Returns:
        tuple: A tuple containing the current direction vector (e_i) and the normal vector to the current direction
        (e_i_n).
    """

    # Current direction vector
    e_i = np.array([np.cos(theta_i), np.sin(theta_i)])
    # Normal vector to current direction
    # https://stackoverflow.com/questions/1243614/
    e_i_n = np.array([np.sin(theta_i), -np.cos(theta_i)])
    return e_i, e_i_n


@jit(nopython=True)
def projection(e_i_n, e_i_j, s_i_j: float) -> float:
    """
    Calculates the projection of vector e_i_j onto vector e_i_n.

    Args:
        e_i_n (array-like): The vector onto which the projection is made.
        e_i_j (array-like): The vector being projected.
        s_i_j (float): The magnitude of vector e_i_j.

    Returns:
        float: The projection of vector e_i_j onto vector e_i_n.
    """

    proj = abs(np.dot(e_i_n, e_i_j)) * s_i_j
    return proj


def radial(agent: Particle, xv: float, yv: float) -> float:
    """
    Calculate the radial distance between an agent and a point in a 2D plane.

    Args:
        agent (dict): A dictionary representing the agent with the following keys:
            - "theta" (float): The angle of the agent in radians.
            - "pos" (tuple): The position of the agent as a tuple of (x, y) coordinates.
            - "l" (float): The length of the agent.
            - "w" (float): The width of the agent.

        xv (float): The x-coordinate of the point.
        yv (float): The y-coordinate of the point.

    Returns:
        rad (float): The radial distance between the agent and the point.
    """

    cos_angle = np.cos(np.pi - agent["theta"])
    sin_angle = np.sin(np.pi - agent["theta"])
    xc = xv - agent["pos"][0]
    yc = yv - agent["pos"][1]
    # https://stackoverflow.com/questions/37031356/
    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle
    rad = xct**2 / agent["l"] ** 2 + yct**2 / agent["w"] ** 2
    return rad


@jit(nopython=True)
def tangent_dist(theta_i: float, theta_j: float, a_i: float, b_i: float) -> float:
    """
    Calculate the tangent distance between two angles.

    Args:
        theta_i (float): The first angle in radians.
        theta_j (float): The second angle in radians.
        a_i (float): The value of a_i.
        b_i (float): The value of b_i.

    Returns:
        k_prime (float): The tangent distance between theta_i and theta_j.
    """
    if round(np.cos(theta_j - theta_i), 15) != 0:
        c_prime = np.sin(theta_j - theta_i) / np.cos(theta_j - theta_i)
        d_prime = np.sqrt(b_i**2 + a_i**2 * c_prime**2)
        k_prime = d_prime / np.sqrt(c_prime**2 + 1)
    else:
        k_prime = a_i
    return k_prime


def truncated_rvs(
    rng,
    size: int,
    dist,
    x_min: float,
    x_max: float,
    k,
    s: float,
    loc: float,
    scale: float,
):
    """
    Generate truncated random variables.

    Args:
        rng (object): Random number generator.
        size (int): Number of random variables to generate.
        dist (object): Probability distribution object.
        x_min (float): Minimum value for truncation.
        x_max (float): Maximum value for truncation.
        k (float): First shape parameter for the distribution.
        s (float): Second shape parameter for the distribution.
        loc (float): Location parameter for the distribution.
        scale (float): Scale parameter for the distribution.

    Returns:
        numpy.ndarray: Array of truncated random variables.
    """
    # https://stackoverflow.com/questions/47933019
    if k:
        low = dist.cdf(x_min, k, s, loc=loc, scale=scale)
        high = dist.cdf(x_max, k, s, loc=loc, scale=scale)

        return dist.ppf(
            rng.uniform(low=low, high=high, size=size),
            k,
            s,
            loc=0,
            scale=scale,
        )
    else:
        low = dist.cdf(x_min, s, loc=loc, scale=scale)
        high = dist.cdf(x_max, s, loc=loc, scale=scale)

        return dist.ppf(
            rng.uniform(low=low, high=high, size=size),
            s,
            loc=0,
            scale=scale,
        )
