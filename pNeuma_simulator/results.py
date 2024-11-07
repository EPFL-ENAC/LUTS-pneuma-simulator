import json
import zipfile

import numpy as np

from pNeuma_simulator import params


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
