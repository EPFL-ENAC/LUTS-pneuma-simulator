import numpy as np

from pNeuma_simulator.particle import Particle


def radial(agent: Particle, xv: float, yv: float) -> float:
    """
    Calculate the radial distance between an agent and a point in a 2D plane.
    Parameters:
    - agent (dict): A dictionary representing the agent with the following keys:
        - "theta" (float): The angle of the agent in radians.
        - "pos" (tuple): The position of the agent as a tuple of (x, y) coordinates.
        - "l" (float): The length of the agent.
        - "w" (float): The width of the agent.
    - xv (float): The x-coordinate of the point.
    - yv (float): The y-coordinate of the point.
    Returns:
    - rad (float): The radial distance between the agent and the point.
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
