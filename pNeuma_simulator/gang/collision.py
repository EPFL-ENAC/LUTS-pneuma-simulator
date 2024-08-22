from math import cos, radians, sin

from numba import jit

from pNeuma_simulator import params
from pNeuma_simulator.dca import calc_dtc
from pNeuma_simulator.gang.particle import Particle
from pNeuma_simulator.utils import tangent_dist


def collisions(ego: Particle, speed: float, theta: float, neighbors: list[Particle]):
    """Returns time to collision in seconds if defined.

    Args:
        ego (Particle): ego vehicle.
        speed (float): check for collision at this speed
        theta (float): check for collision in this direction
        neighbors (list[Particle]): list of potentially colliding vehicles

    Returns:
        float: time to collision (or None if not defined).
    """
    l_ttc = []
    # initialize i
    l_i, w_i = ego.l, ego.w
    v_i = speed
    theta_i = theta
    vx_i, vy_i = v_i * cos(theta_i), v_i * sin(theta_i)
    x_i, y_i = ego.x, ego.y
    # analytically compute wall collision time
    # possible dvision by zero
    if vy_i != 0:
        k_w = tangent_dist(theta_i, 0, l_i, w_i)
        if theta_i >= radians(0.5):
            ttc_w = (params.lane - (y_i + k_w)) / vy_i
        elif theta_i <= -radians(0.5):
            ttc_w = (-params.lane - (y_i - k_w)) / vy_i
        else:
            ttc_w = None
    else:
        ttc_w = None
    if ttc_w is not None:
        l_ttc.append(ttc_w)
    # numerically compute anticipated collision time
    for neighbor in neighbors:
        # initialize j
        l_j, w_j = neighbor.l, neighbor.w
        theta_j = neighbor.theta
        x_j, y_j = neighbor.x, neighbor.y
        vx_j, vy_j = neighbor.vx, neighbor.vy
        ttc = newton_iteration(
            l_j,
            w_j,
            l_i,
            w_i,
            x_j,
            y_j,
            x_i,
            y_i,
            theta_j,
            theta_i,
            vx_j,
            vy_j,
            vx_i,
            vy_i,
        )
        if len(ttc) > 0:
            l_ttc.extend(ttc)
    if len(l_ttc) > 0:
        return min(l_ttc)
    else:
        return None


@jit(nopython=True)
def newton_iteration(
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
    vx_j: float,
    vy_j: float,
    vx_i: float,
    vy_i: float,
    max_iterations: int = 10,
    tolerance: float = 0.05,
    epsilon: float = 0.001,
    h: float = 0.001,
):
    """
    Perform Newton's iteration to find the time-to-collision (TTC) between two objects.

    Args:
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
        vx_j (float): x-component of velocity of object j.
        vy_j (float): y-component of velocity of object j.
        vx_i (float): x-component of velocity of object i.
        vy_i (float): y-component of velocity of object i.
        max_iterations (int): Maximum number of iterations.
        epsilon (float): Small value to avoid division by zero.
        tolerance (float): Desired tolerance.
        h (float): Step size for Newton's iteration.

    Returns:
        l_ttc (list): List of time-to-collision values within the desired tolerance and maximum number of iterations.
    """
    l_ttc = []
    ttc_0 = 0
    for _ in range(max_iterations):
        x_j_0 = x_j + vx_j * ttc_0
        y_j_0 = y_j + vy_j * ttc_0
        x_i_0 = x_i + vx_i * ttc_0
        y_i_0 = y_i + vy_i * ttc_0
        dtc = calc_dtc(l_j, w_j, l_i, w_i, x_j_0, y_j_0, x_i_0, y_i_0, theta_j, theta_i)
        if (dtc <= 0) and (ttc_0 > 0):
            l_ttc.append(ttc_0)  # a solution within tolerance and maximum number of iterations
            break
        x_j_0_ = x_j + vx_j * (ttc_0 + h)
        y_j_0_ = y_j + vy_j * (ttc_0 + h)
        x_i_0_ = x_i + vx_i * (ttc_0 + h)
        y_i_0_ = y_i + vy_i * (ttc_0 + h)
        dtc_ = calc_dtc(l_j, w_j, l_i, w_i, x_j_0_, y_j_0_, x_i_0_, y_i_0_, theta_j, theta_i)
        if dtc_ < dtc:
            dtc_prime = (dtc_ - dtc) / h
            if abs(dtc_prime) < epsilon:  # Give up if the denominator is too small
                break
            ttc_1 = ttc_0 - dtc / dtc_prime  # Do Newton's computation
            if abs(ttc_1 - ttc_0) <= tolerance:  # Stop when the result is within the desired tolerance
                l_ttc.append(ttc_1)  # a solution within tolerance and maximum number of iterations
                break

            ttc_0 = ttc_1  # Update x0 to start the process again
    return l_ttc
