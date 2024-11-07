from math import cos, radians, sin

from numba import jit

from pNeuma_simulator import params
from pNeuma_simulator.contact_distance import calc_dtc
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
        ttc, _ = newton_iteration(
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
        if ttc:
            l_ttc.append(ttc)
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
    max_iterations: int = 1000,
    tolerance: float = 1e-3,
    delta: float = 1e-9,
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
        max_iterations (int, optional): Maximum number of iterations. Default is 1000.
        tolerance (float, optional): Desired tolerance. Default is 0.001.
        delta (float, optional): Small value to avoid division by zero. Default is 1e-9.

    Returns:
        tuple: A tuple containing:
            - float: Time-to-collision value within the desired tolerance and maximum number of iterations, or None if
            no solution is found.
            - int: Number of iterations performed, or None if no solution is found.
    """
    t0 = 0
    for iterations in range(max_iterations):
        # Coordinates at t0
        y_j0 = y_j + vy_j * t0
        x_i0 = x_i + vx_i * t0
        x_j0 = x_j + vx_j * t0
        y_i0 = y_i + vy_i * t0
        d0 = calc_dtc(l_j, w_j, l_i, w_i, x_j0, y_j0, x_i0, y_i0, theta_j, theta_i)

        # Coordinates at (t0 + delta)
        x_j1 = x_j + vx_j * (t0 + delta)
        y_i1 = y_i + vy_i * (t0 + delta)
        y_j1 = y_j + vy_j * (t0 + delta)
        x_i1 = x_i + vx_i * (t0 + delta)
        d1 = calc_dtc(l_j, w_j, l_i, w_i, x_j1, y_j1, x_i1, y_i1, theta_j, theta_i)

        if d1 > d0:  # Give up if agents are diverging
            break
        if d1 == d0:
            return (t0, iterations)  # Equilibrium solution
        dprime = (d1 - d0) / delta
        t1 = t0 - d0 / dprime  # Do Newton's computation
        if abs(t1 - t0) <= tolerance:  # Stop when the result is within the desired tolerance
            return (
                t1,
                iterations,
            )  # a solution within tolerance and maximum number of iterations
        t0 = t1  # Update t0 to start the process again

    return (None, None)
