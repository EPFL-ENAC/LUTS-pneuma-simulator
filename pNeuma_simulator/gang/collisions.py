from math import cos, radians, sin

from pNeuma_simulator import params
from pNeuma_simulator.gang.particle import Particle
from pNeuma_simulator.newton_iteration import newton_iteration
from pNeuma_simulator.utils.tangent_dist import tangent_dist


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
