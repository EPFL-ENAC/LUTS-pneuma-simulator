import json
import os
import zipfile

import numpy as np
from scipy.stats import binned_statistic

from pNeuma_simulator import params
from pNeuma_simulator.gang import decay
from pNeuma_simulator.initialization import ov


def loader(permutation, path: str, verbose: bool = True):
    """
    Loads and returns the items from a JSON file within a zip archive.

    Args:
        permutation: The permutation to be used in the zip file name.
        path (str): The path to the directory containing the zip file.
        verbose (bool): Specify whether to print the file name

    Returns:
        list: The items loaded from the JSON file.
    """
    with zipfile.ZipFile(f"{path}{permutation}.zip", "r") as ziph:
        # ziph is zipfile handle
        for filename in ziph.namelist():
            if filename.endswith(").json"):
                # Opening JSON file
                with ziph.open(filename, "r") as openfile:
                    # Reading from JSON file
                    items = json.load(openfile)
                    if verbose:
                        print(openfile.name)
    return items


def aggregate(l_agents, n_cars: int, n_moto: int):
    """
    Calculate various aggregate metrics based on the given list of agents.

    Args:
        l_agents (list[Particle]): A list of agents. (#list(list(Particles))?)
        n_cars (int): The number of cars.
        n_moto (int): The number of motorcycles.

    Returns:
        VKT_cars (float): Vehicle Kilometers Traveled by cars.
        VHT_cars (float): Vehicle Hours Traveled by cars.
        VKT_moto (float): Vehicle Kilometers Traveled by motorcycles.
        VHT_moto (float): Vehicle Hours Traveled by motorcycles.
    """

    l_cars_dx = []
    l_moto_dx = []

    for t, agents in enumerate(l_agents):
        cars_dx = 0
        moto_dx = 0
        for agent in agents:
            dx = agent["vel"][0] * params.dt
            if agent["mode"] == "Car":
                cars_dx += dx
            else:
                moto_dx += dx
        # Ignore transient
        if t >= int(len(l_agents) * (1 - params.keep)):
            l_cars_dx.append(cars_dx)
            if n_moto > 0:
                l_moto_dx.append(moto_dx)
    VKT_cars = 1e-3 * sum(l_cars_dx)
    VHT_cars = 2e-3 * n_cars * len(l_cars_dx) * params.dt / params.factor
    if n_moto > 0:
        VKT_moto = 1e-3 * sum(l_moto_dx)
        VHT_moto = 1e-3 * n_moto * len(l_moto_dx) * params.dt / params.factor
    else:
        VKT_moto = None
        VHT_moto = None
    return VKT_cars, VHT_cars, VKT_moto, VHT_moto


def intersect(p1, p2, p3, p4):
    # intersection between line(p1, p2) and line(p3, p4)
    # https://stackoverflow.com/questions/3252194/
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:  # parallel
        return None
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    if ua < 0 or ua > 1:  # out of range
        return None
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    if ub < 0 or ub > 1:  # out of range
        return None
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    return (x, y)


def normalized(surface, section):
    data = surface.get_array().data
    curves = []
    values = []
    for n, collection in enumerate(surface.collections):
        paths = collection.get_paths()
        if len(paths) > 0:
            for path in paths:
                curves.append(path)
                values.append(data[n])
    diagonals = []
    for collection in section.collections:
        path = collection.get_paths()[0]
        diagonals.append(path)
    # find intersections
    l_points = []
    l_response = []
    for diagonal in diagonals[:]:
        points = []
        response = []
        # https://stackoverflow.com/questions/38151445/
        diagonal_segments = list(zip(diagonal.vertices, diagonal.vertices[1:]))
        intersecting = []
        mask = []
        for curve in curves:
            if diagonal.intersects_path(curve):
                mask.append(False)
                intersecting.append(curve)
            else:
                mask.append(True)
        elevations = np.array(values)[~np.array(mask)]
        for diagonal_segment in diagonal_segments:
            p1, p2 = diagonal_segment
            for n, curve in enumerate(intersecting):
                # https://stackoverflow.com/questions/38151445/
                curve_segments = list(zip(curve.vertices, curve.vertices[1:]))
                for curve_segment in curve_segments:
                    p3, p4 = curve_segment
                    intersection = intersect(p1, p2, p3, p4)
                    if intersection:
                        points.append(intersection)
                        response.append(elevations[n])
        l_points.append(points)
        l_response.append(response)
    return l_points, l_response


def percolate(items, n_moto, start: int = 1):
    """
    Analyzes the percolation of vehicles and motorcycles in a given dataset.

    Parameters:
    items (list): A list of items where each item is a list of frames. Each frame is a dictionary containing vehicle
        data.
    n_moto (int): The number of motorcycles in the dataset.
    start (int, optional): The starting frame index to consider for analysis. Defaults to 1.

    Returns:
    tuple: A tuple containing four lists:
        - x (list): The bin centers for the binned data.
        - y (list): The mean difference in velocity between motorcycles and cars for each bin.
        - low (list): The lower bound of the confidence interval for each bin.
        - high (list): The upper bound of the confidence interval for each bin.
    """
    l_T = []
    l_DPhi = []
    for item in items:
        if isinstance(item[0], list):
            for t, frame in enumerate(item[0]):
                if t > start:
                    deg_range = []
                    vel_car = []
                    vel_x = []
                    vel_y = []
                    for j, _ in enumerate(frame):
                        vel = frame[j]["vel"]
                        v0 = frame[j]["v0"]
                        lam = frame[j]["lam"]
                        d = frame[j]["d"]
                        v_max = ov(params.d_max, lam, v0, d)
                        if frame[j]["mode"] == "Car":
                            vel_car.append(vel[0] / v_max)
                        else:
                            alphas = decay(np.array(vel), frame[j]["theta"])
                            degs = np.degrees(alphas)
                            deg_range.append(degs[0] - degs[-1])
                            vel_x.append(vel[0] / v_max)
                            vel_y.append(vel[1] / v_max)
                    l_T.append(np.mean(deg_range))
                    phi_cars = np.mean(vel_car)
                    phi_moto = np.norm([np.sum(vel_x), np.sum(vel_y)]) / n_moto
                    l_DPhi.append(phi_moto - phi_cars)
    l_T = np.round(l_T) / 2
    bins = np.sort(np.unique(l_T))
    y, bin_edges, _ = binned_statistic(l_T, l_DPhi, statistic="mean", bins=bins)
    x = (bin_edges[1:] + bin_edges[:-1]) / 2
    low, _, _ = binned_statistic(l_T, l_DPhi, statistic=lambda y: confidence_interval(y, "low"), bins=bins)
    high, _, _ = binned_statistic(l_T, l_DPhi, statistic=lambda y: confidence_interval(y, "high"), bins=bins)

    return list(x), list(y), list(low), list(high)


def zipdir(path: str, ziph) -> None:
    """
    Zip the directory at the given path.

    Args:
        path (str): The path of the directory to be zipped.
        ziph: The zipfile handle.
    """
    # ziph is zipfile handle
    # https://stackoverflow.com/questions/1855095/
    # https://stackoverflow.com/questions/36740683/
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(").json"):
                os.chdir(root)
                ziph.write(file)
                os.remove(file)
                path_parent = os.path.dirname(os.getcwd())
                os.chdir(path_parent)


def confidence_interval(data, rng, setting="sem"):
    """
    Calculate the confidence interval or standard error of the mean (SEM) for a given dataset.

    Parameters:
    data (array-like): The dataset for which the confidence interval or SEM is to be calculated.
    rng (numpy.random.Generator): A random number generator instance for reproducibility.
    setting (str, optional): The type of result to return. Options are:
        - "low": Return the lower bound of the confidence interval.
        - "high": Return the upper bound of the confidence interval.
        - "sem": Return the standard error of the mean (default).

    Returns:
    float or None: The requested confidence interval bound or SEM if the dataset has more than one element, otherwise
    None.
    """
    if len(data) > 1:
        # https://github.com/scipy/scipy/issues/14645
        res = np.bootstrap(
            (data,),
            np.mean,
            batch=10,
            confidence_level=0.95,
            random_state=rng,
            method="basic",
        )
        if setting == "low":
            return res.confidence_interval.low
        elif setting == "high":
            return res.confidence_interval.high
        elif setting == "sem":
            return res.standard_error
    else:
        return None
