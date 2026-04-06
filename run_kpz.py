import argparse
import json
import os
import warnings
import zipfile
from copy import deepcopy
from math import cos, inf, sin

# from os import path
import numpy as np
from joblib import Parallel, delayed  # pyright: ignore[reportMissingImports]
from joblib.externals.loky import get_reusable_executor  # pyright: ignore[reportMissingImports]
from numba import jit  # pyright: ignore[reportMissingImports]
from numpy import unravel_index
from numpy.linalg import norm
from numpy.typing import NDArray

# from tqdm.notebook import tqdm
from pNeuma_simulator import params
from pNeuma_simulator.contact_distance import ellipses
from pNeuma_simulator.gang.collision import collisions
from pNeuma_simulator.gang.neighborhood import neighborhood
from pNeuma_simulator.initialization import PoissonDisc
from pNeuma_simulator.shadowcasting.shadowcasting import shadowcasting
from pNeuma_simulator.utils import direction, projection, tangent_dist

warnings.filterwarnings("ignore")


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

    # Check if neighbor is in front
    r_i_j = pos_j - pos_i
    front = np.dot(e_i, r_i_j) > 0
    e_i_j = np.zeros(2)
    s_i_j = 0.0
    if front:
        # Distance from i to j
        s_i_j = float(norm(r_i_j))
        # Unit vector from i to j
        e_i_j = r_i_j / s_i_j
    return front, e_i_j, s_i_j


def ov(
    s: float | NDArray[np.floating], v: float | NDArray[np.floating], lam: float, v0: float, s0: float, T: float
) -> float | NDArray[np.floating]:
    x = v0 * (1 - np.exp((-lam / v0) * (s - s0 + T * v)))
    return x


def main(
    p=0.0001,
    seed=1,
    time_duration=10,
    brake_duration=3,
    discounting=0.6,
    n_cars=58,  # 232
    n_moto=0,
):

    COUNT = int(time_duration * 60 / params.dt)
    rng = np.random.default_rng(seed)
    T_a = 0.7993
    lam_a = 0.801
    v0_a = 11.36
    s0_a = 1.657
    lamprime_a = 1.085
    tauprime_a = 0.5969
    tau_a = 2.098

    sampler = PoissonDisc(
        n_cars,
        n_moto,
        lanes=[0],
        cell=params.cell,
        L=params.L,
        W=params.cell * 3,
        k=params.k,
        clearance=params.clearance,
        rng=rng,
    )
    samples, images = sampler.sample(rng)
    agents = samples
    brake_timer = np.zeros(len(agents), dtype=int)
    brake_acc = np.zeros(len(agents))
    l_agents = []
    # l_acc = []
    T = np.repeat(T_a, n_cars)
    lam = np.repeat(lam_a, n_cars)
    tau = np.repeat(tau_a, n_cars)
    v0 = np.repeat(v0_a, n_cars)
    s0 = np.repeat(s0_a, n_cars)
    lamprime = np.repeat(lamprime_a, n_cars)
    tauprime = np.repeat(tauprime_a, n_cars)
    for n, agent in enumerate(agents):
        # Reassign IDs
        agent.ID = n + 1
        agent.image = None
        agent.styles = None
        agent.tau = tau[n]
        agent.lam = lam[n]
        agent.v0 = v0[n]
        agent.s0 = s0[n]
    for t in range(COUNT - 1):
        rand = rng.random(len(agents))
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
            serial_agents.append(agent.x)
        l_agents.append(serial_agents)
        ##############################
        # Field of View analysis
        ##############################
        if t == 0:  # % skip == 0:
            for image in agents + images:
                cos_angle = -cos(image.theta)
                sin_angle = sin(image.theta)
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
                # top/bottom boundaries
                matrix[[0, -1], :] = 1
                for image in agents + images:
                    if image.ID != agent.ID:
                        assert image.rad is not None
                        matrix[image.rad < 1] = image.ID
                matrices.append(matrix)
                assert agent.rad is not None
                origin = unravel_index(agent.rad.argmin(), params.shape)
                origins.append(origin)
            tuples = []
            for i, j in zip(matrices, origins):
                shadowcast = shadowcasting(i, j, params.grid, params.L, params.d_max)
                tuples.append(shadowcast)
            for n, agent in enumerate(agents):
                assert tuples is not None
                interactions = tuples[n]
                agent.interactions = interactions.tolist()
        ################################
        # Dynamics
        ################################
        l_theta = []
        l_speed = []
        l_gap = []
        l_ttc = []
        for agent in agents:
            l_speed.append(agent.speed)
            theta_i = agent.theta
            l_theta.append(theta_i)
            # Semiaxis dimensions of i
            l_i, w_i = agent.l, agent.w
            # Absolute position of i
            pos_i = agent.pos
            x_i, y_i = pos_i
            interactions = agent.interactions
            if len(interactions) > 0:
                gaps = []
                neighbors = neighborhood(agent, agents)
                # Direction vector and its normal
                e_i, e_i_n = direction(theta_i)
                front_neighbors = []
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
                        front_neighbors.append(neighbor)
                        # Distance from tangent parallel to i
                        k_h = tangent_dist(theta_j, theta_i, l_j, w_j)
                        proj = projection(e_i_n, e_i_j, s_i_j)
                        # This is extremely important!!!
                        if proj <= params.scaling * w_i + k_h:
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
                    # actual time to collision
                    assert agent.speed is not None
                    ttc = collisions(agent, agent.speed, 0, [leader])
                    agent.ttc = ttc
                else:
                    agent.leader = None
                    agent.ttc = None
                    agent.gap = params.d_max
            else:
                agent.leader = None
                agent.ttc = None
                agent.gap = params.d_max
            assert agent.gap is not None
            if agent.gap <= 0:
                print("Accident occurred")
            # Retrieve inverse ttc
            if agent.ttc is None:
                ttc = -1
            else:
                ttc = agent.ttc
            l_ttc.append(ttc)
            l_gap.append(agent.gap)
        # Stochastic component
        ttc = np.array(l_ttc)
        gaps = np.array(l_gap)
        DeltaV = gaps / ttc
        V = np.array(l_speed)
        OV = np.array(ov(gaps, V, lam[0], v0[0], s0[0], T[0]))
        # --- deterministic acceleration ---
        acc_det = (OV - V) / tau - (np.maximum(0, DeltaV) / tauprime) * (np.exp(-(lamprime / v0) * (gaps - s0 + T * V)))
        # --------------------------------------------------
        # 1. Decrement timers from previous timestep
        # --------------------------------------------------
        brake_timer[brake_timer > 0] -= 1
        # --------------------------------------------------
        # 2. Trigger new braking events
        # --------------------------------------------------
        new_brake = (brake_timer == 0) & (rand < p)
        if np.any(new_brake):
            idx = np.where(new_brake)[0]
            durations = np.repeat(brake_duration, len(idx))
            brake_timer[idx] = np.ceil(durations / params.dt).astype(int)
            brake_acc[idx] = (discounting * OV[idx] - V[idx]) / tau[idx] - (
                (np.maximum(0, DeltaV)[idx]) / tauprime[idx]
            ) * (np.exp(-(lamprime[idx] / v0[idx]) * (gaps[idx] - s0[idx] + T[idx] * V[idx])))
            # l_acc.extend(brake_acc[idx])
        # --------------------------------------------------
        # 3. Recompute active braking AFTER timer update
        # --------------------------------------------------
        braking = brake_timer > 0
        # --------------------------------------------------
        # 4. Compute acceleration
        # --------------------------------------------------
        acc = np.where(braking, brake_acc, acc_det)
        # --------------------------------------------------
        # 5. Update velocity
        # --------------------------------------------------
        new_V = np.maximum(V + acc * params.dt, 0.0)
        ##################################
        # Advance the simulation
        ##################################
        for n, agent in enumerate(agents):
            agent.advance(params.dt, new_V[n])
            agent.image = None
            agent.trajectory = None
    return l_agents  # , l_acc


def execute(epochs=128, n_jobs=16, p=0.0001, time_duration=40, brake_duration=3, discounting=0.6, n_cars=232):
    seeds = np.arange(epochs)
    path = (
        f"./notebooks/output/sketches/nasch_INM_{round(p, 4)}_{time_duration}_{brake_duration}_{discounting}_{n_cars}"
    )
    zip_path = f"{path}.zip"
    jsonl_name = f"{os.path.basename(path)}.jsonl"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        with zf.open(jsonl_name, "w", force_zip64=True) as f:
            for i in range(0, epochs, n_jobs):
                batch = seeds[i : i + n_jobs]
                with Parallel(n_jobs=n_jobs) as parallel:
                    items = parallel(
                        delayed(main)(
                            p=p,
                            seed=seed,
                            time_duration=time_duration,
                            brake_duration=brake_duration,
                            discounting=discounting,
                            n_cars=n_cars,
                            n_moto=0,
                        )
                        for seed in batch
                    )
                    # https://stackoverflow.com/questions/67495271/
                    get_reusable_executor().shutdown(wait=True)
                if items:
                    # write batch immediately, then discard
                    for item in items:
                        f.write((json.dumps(item) + "\n").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="User input", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-e", "--epochs", default=128, help="number of epochs")
    parser.add_argument("-j", "--n_jobs", default=16, help="number of jobs")
    parser.add_argument("-d", "--time_duration", default=40, help="time duration")
    parser.add_argument("n_cars", help="number of cars")
    args = parser.parse_args()
    config = vars(args)
    n_cars = int(config["n_cars"])
    epochs = int(config["epochs"])
    n_jobs = int(config["n_jobs"])
    time_duration = int(config["time_duration"])
    print(config)
    execute(
        epochs=epochs,
        n_jobs=n_jobs,
        p=0.0001,
        time_duration=time_duration,
        brake_duration=3,
        discounting=0.6,
        n_cars=n_cars,
    )
    print("Done!")
