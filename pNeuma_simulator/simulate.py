import warnings
from copy import deepcopy
from math import cos, inf, isinf, pi, radians, sin
from typing import Callable

import numpy as np
from joblib import Parallel, delayed, parallel_backend
from numba import jit
from numpy.linalg import norm

from pNeuma_simulator import params
from pNeuma_simulator.contact_distance import ellipses
from pNeuma_simulator.gang import navigate
from pNeuma_simulator.gang.neighborhood import neighborhood
from pNeuma_simulator.initialization import PoissonDisc, equilibrium, ov
from pNeuma_simulator.shadowcasting import shadowcasting
from pNeuma_simulator.utils import direction, projection, tangent_dist


def main(n_cars: int, n_moto: int, seed: int, parallel: Callable, COUNT: int = 500, distributed: bool = True):
    """
    Simulates the main loop of a pNeuma simulator.

    Args:
        n_cars (int): Number of cars.
        n_moto (int): Number of motorcycles.
        seed (int): Seed for the random number generator.
        parallel (Callable): Callable object for parallel execution.
        COUNT (int, optional): Number of iterations in the main loop. Defaults to 500.
        distributed (bool, optional): Flag indicating if the simulation is distributed. Defaults to True.

    Returns:
        Tuple: A tuple containing the list of serialized agents at each iteration and an empty list.
    """
    # Code implementation...

    rng = np.random.default_rng(seed)
    ###############################################
    # Main loop
    ###############################################
    E = np.zeros((COUNT, 2 * n_cars + n_moto))
    sampler = PoissonDisc(
        n_cars, n_moto, cell=params.cell, L=params.L, W=params.cell * 3, k=params.k, clearance=params.clearance, rng=rng
    )
    # [car, car, ..., moto]
    samples, _ = sampler.sample(rng)
    agents = samples[: 2 * n_cars]
    if n_moto > 0:
        agents.extend(rng.choice(samples[2 * n_cars :], n_moto, replace=False))
    l_agents = []
    tau, lam, v0, d = equilibrium(
        params.L,
        params.lanes,
        n_cars,
        n_moto,
        rng,
        distributed=distributed,
    )
    l_a = []
    l_b = []
    l_A = np.repeat(params.A, len(agents))
    l_B = np.repeat(params.B, len(agents))
    for n, agent in enumerate(agents):
        # Reassign IDs
        agent.ID = n + 1
        agent.image = None
        agent.styles = None
        agent.tau = tau[n]
        agent.lam = lam[n]
        agent.v0 = v0[n]
        agent.d = d[n]
        l_a.append(agent.a)
        l_b.append(agent.b)
    for t in range(COUNT - 1):
        ######################
        # Periodic boundary
        ######################
        images = []
        serial_agents = []
        for agent in agents:
            if agent.x < -(params.L / 2 - (params.d_max + agent.l)):
                image = deepcopy(agent)
                image.x += params.L
                images.append(image)
                agent.image = image
            elif agent.x > params.L / 2 - (params.d_max + agent.l):
                image = deepcopy(agent)
                image.x -= params.L
                images.append(image)
                agent.image = image
            serial_agent = deepcopy(agent)
            serial_agent.pos = serial_agent.pos.tolist()
            serial_agent.vel = serial_agent.vel.tolist()
            serial_agents.append(serial_agent.encode())
        l_agents.append(serial_agents)
        ##############################
        # Field of View analysis
        ##############################
        for image in agents + images:
            cos_angle = cos(pi - image.theta)
            sin_angle = sin(pi - image.theta)
            xc = params.xv - image.x
            yc = params.yv - image.y
            # https://stackoverflow.com/questions/37031356/
            xct = xc * cos_angle - yc * sin_angle
            yct = xc * sin_angle + yc * cos_angle
            rad = xct**2 / image.l**2 + yct**2 / image.w**2
            image.rad = rad
        matrices = []
        origins = []
        for agent in agents:
            matrix = np.zeros(params.shape)
            matrix[[0, -1]] = 1
            for image in agents + images:
                if image.ID != agent.ID:
                    matrix = identify(matrix, image.rad, image.ID)
            matrices.append(matrix)
            origin = np.unravel_index(agent.rad.argmin(), params.shape)
            origins.append(origin)
        tuples = parallel(
            delayed(shadowcasting)(i, j, params.grid, params.L, params.d_max) for i, j in zip(matrices, origins)
        )
        for n, agent in enumerate(agents):
            interactions = tuples[n]
            agent.interactions = interactions.tolist()
        ##################################################
        # Navigation module
        ##################################################
        navigators = []
        for agent in agents:
            interactions = agent.interactions
            if len(interactions) > 0:
                navigators.append(agent)
        if len(navigators) > 0:
            integers = rng.integers(1e8, size=len(navigators))
            tuples = parallel(
                delayed(navigate)(navigator, agents, integer, params.d_max)
                for integer, navigator in zip(integers, navigators)
            )
            for n, agent in enumerate(navigators):
                a0, a_des, f_a, ttc = tuples[n]
                agent.ttc = ttc
                if agent.mode == "Moto":
                    agent.a0 = a0
                    agent.a_des = a_des
                    agent.f_a = f_a.tolist()
        ################################
        # Longitudinal dynamics
        ################################
        l_theta = []
        l_gap = []
        l_pseudottc = []
        l_vel = []
        for agent in agents:
            l_vel.append(agent.vel)
            # Updated direction of i
            if agent.mode == "Moto":
                new_theta = agent.theta + params.dt * (agent.a_des - agent.theta) / params.adaptation_time
                theta_i = new_theta
            else:
                theta_i = agent.theta
            l_theta.append(theta_i)
            # Semiaxis dimensions of i
            l_i, w_i = agent.l, agent.w
            # Absolute position of i
            pos_i = agent.pos
            x_i, y_i = pos_i
            # Distance from walls
            k_w = tangent_dist(theta_i, 0, l_i, w_i)
            if theta_i >= radians(0.5):
                gap_w = (params.lane - y_i - k_w) / sin(theta_i)
            elif theta_i <= -radians(0.5):
                gap_w = (params.lane + y_i - k_w) / sin(-theta_i)
            else:
                gap_w = inf
            interactions = agent.interactions
            if len(interactions) > 0:
                gaps = []
                neighbors = neighborhood(agent, agents)
                # Direction vector and its normal
                e_i, e_i_n = direction(theta_i)
                for neighbor in neighbors:
                    # Semiaxis dimensions of j
                    l_j, w_j = neighbor.l, neighbor.w
                    # Absolute position of j
                    pos_j = neighbor.pos
                    x_j, y_j = pos_j
                    # Direction of j
                    theta_j = neighbor.theta
                    # Check if neighbor is in front
                    front, e_i_j, s_i_j = infront(e_i, pos_i, pos_j)
                    if front:
                        # Distance from tangent parallel to i
                        k_h = tangent_dist(theta_j, theta_i, l_j, w_j)
                        proj = projection(e_i_n, e_i_j, s_i_j)
                        if proj <= params.scaling * w_i + k_h:  # This is extremely important!!!
                            # Distance of closest approach between i and j
                            if proj == 0:
                                min_d = l_i + l_j
                            else:
                                min_d = ellipses(
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
                                )
                            gap = s_i_j - min_d
                        else:
                            gap = inf
                    else:
                        gap = inf
                    gaps.append(gap)
                if np.isfinite(gaps).sum() > 0:
                    leader = neighbors[np.argmin(gaps)]
                    gap = min(gaps)
                    agent.leader = leader.ID
                    agent.gap = gap
                else:
                    agent.leader = None
                    if isinf(gap_w):
                        agent.gap = params.d_max
                    else:
                        agent.gap = min([gap_w, params.d_max])
            else:
                agent.leader = None
                if isinf(gap_w):
                    agent.gap = params.d_max
                else:
                    agent.gap = min([gap_w, params.d_max])
            if agent.gap <= 0:
                return tuple(agent.pos)
                raise CollisionException("Accident occurred")
            # Retrieve inverse ttc
            if agent.ttc is None:
                pseudottc = 0
            else:
                pseudottc = -1 / agent.ttc
            l_pseudottc.append(pseudottc)
            l_gap.append(agent.gap)
        dW = params.sqrtdt * rng.standard_normal(len(agents))
        E[t + 1] = (1 - params.dt / np.array(l_b)) * E[t] + np.array(l_a) * dW
        V = norm(l_vel, axis=1)
        OV = ov(np.array(l_gap), lam, v0, d) + E[t + 1]
        V_des = OV * (0.5 * (1 + np.tanh(l_A * (np.array(l_pseudottc) + l_B))))
        new_V = V + ((V_des - V) / tau) * params.dt
        new_V = np.maximum(new_V, 0)
        new_theta = np.array(l_theta)
        ##################################
        # Advance the simulation
        ##################################
        for n, agent in enumerate(agents):
            agent.advance(params.dt, new_V[n], new_theta[n])
            agent.image = None

    return (l_agents, [])


def batch(seed: int, permutation: tuple, n_jobs: int):
    """
    Run a batch simulation with the given seed and permutation.

    Args:
        seed (int): The seed for random number generation.
        permutation (tuple): A tuple containing the number of cars and motorcycles.
        n_jobs (int): Number of parallel jobs.

    Returns:
        tuple: A tuple containing the simulation results for cars and motorcycles.
    """

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    n_cars, n_moto = permutation

    with parallel_backend("loky", inner_max_num_threads=n_jobs):
        with Parallel(n_jobs=n_jobs, prefer="processes") as parallel:
            try:
                item = main(n_cars, n_moto, seed, parallel, params.COUNT)
            except CollisionException:
                item = (None, None)
    return item


class CollisionException(Exception):
    """Raised when agents collide"""

    # https://stackoverflow.com/questions/1319615
    def __init__(self, message, payload=None):
        self.message = message
        self.payload = payload

    def __str__(self):
        return str(self.message)


@jit(nopython=True)
def identify(matrix: np.ndarray, image_rad: np.ndarray, ID: int) -> np.ndarray:
    """
    Identifies and replaces pixels in the matrix with the given ID based on a condition.

    Args:
        matrix (ndarray): The input matrix.
        image_rad (ndarray): The flattened array of image radii.
        ID (int): The ID to replace the pixels with.

    Returns:
        ndarray: The modified matrix with replaced pixels.
    """
    # matrix to array
    ar_matrix = matrix.flatten()
    ar_rad = image_rad.flatten()
    ar_matrix[np.where(ar_rad < 1)] = ID
    matrix = ar_matrix.reshape(matrix.shape)
    return matrix


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
