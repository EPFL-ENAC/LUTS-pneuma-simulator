import json
import zipfile

import numpy as np

from pNeuma_simulator import params
from pNeuma_simulator.initialization import ov


def loader(permutation, path: str):
    with zipfile.ZipFile(f"{path}{permutation}.zip", "r") as ziph:
        # ziph is zipfile handle
        for filename in ziph.namelist():
            if filename.endswith(").json"):
                # Opening JSON file
                with ziph.open(filename, "r") as openfile:
                    # Reading from JSON file
                    items = json.load(openfile)
                    print(openfile.name)
    return items


def aggregate(l_agents, n_cars: int, n_moto: int, d_max: float, critical: int = 3):
    """
    Calculate various aggregate metrics based on the given list of agents.
    Parameters:
    - l_agents (list[Particle]): A list of agents. (#list(list(Particles))?)
    - n_cars (int): The number of cars.
    - n_moto (int): The number of motorcycles.
    - d_max (float): The maximum distance.
    - critical (int, optional): The critical value. Defaults to 3.
    Returns:
    - VKT (float): Vehicle Kilometers Traveled.
    - VHT (float): Vehicle Hours Traveled.
    - VKT_cars (float): Vehicle Kilometers Traveled by cars.
    - VHT_cars (float): Vehicle Hours Traveled by cars.
    - E_cars (float): Energy consumption by cars.
    - VKT_moto (float): Vehicle Kilometers Traveled by motorcycles.
    - VHT_moto (float): Vehicle Hours Traveled by motorcycles.
    - E_moto (float): Energy consumption by motorcycles.
    - risk (float or None): Risk value, only applicable if there are motorcycles.
    """

    duration = int((len(l_agents)) * params.dt)
    l_mean_dx = []
    l_cars_dx = []
    l_cars_de = []
    l_moto_dx = []
    l_moto_de = []
    exposures = np.zeros(2 * n_cars + n_moto)
    for t, agents in enumerate(l_agents):
        mean_dx = 0
        cars_dx = 0
        cars_de = 0.0
        moto_dx = 0
        moto_de = 0.0
        for n, agent in enumerate(agents):
            dx = agent["vel"][0] * params.dt
            lam = agent["lam"]
            v0 = agent["v0"]
            d = agent["d"]
            mean_dx += dx
            if agent["mode"] == "Car":
                cars_dx += dx
                cars_de += dx / ov(d_max, lam, v0, d)
            else:
                # Ignore transient
                if t >= int(len(l_agents) * (1 - params.keep)):
                    ttc = agent["ttc"]
                    if ttc:
                        if ttc < critical:
                            exposures[n] += 1
                moto_dx += dx
                moto_de += dx / ov(d_max, lam, v0, d)
        cars_de = cars_de / (2 * n_cars)
        if n_moto > 0:
            moto_de = moto_de / (n_moto)
        # Ignore transient
        if t >= int(len(l_agents) * (1 - params.keep)):
            l_mean_dx.append(mean_dx)
            l_cars_dx.append(cars_dx)
            l_cars_de.append(cars_de)
            if n_moto > 0:
                l_moto_dx.append(moto_dx)
                l_moto_de.append(moto_de)
    VKT = sum(l_mean_dx) / 1000
    VHT = (2 * n_cars + n_moto) * len(l_mean_dx) * params.dt / 3600
    VKT_cars = sum(l_cars_dx) / 1000
    VHT_cars = (2 * n_cars) * len(l_cars_dx) * params.dt / 3600
    E_cars = sum(l_cars_de) / (params.keep * duration)
    if n_moto > 0:
        risk = np.mean(exposures[-n_moto:] / (params.keep * len(l_agents)))
        VKT_moto = sum(l_moto_dx) / 1000
        VHT_moto = (n_moto) * len(l_moto_dx) * params.dt / 3600
        E_moto = sum(l_moto_de) / (params.keep * duration)
    else:
        risk = None
        VKT_moto = None
        VHT_moto = None
        E_moto = None
    return VKT, VHT, VKT_cars, VHT_cars, E_cars, VKT_moto, VHT_moto, E_moto, risk


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


def normalized(CS, CD):
    data = CS.get_array().data
    contours = []
    values = []
    for n, collection in enumerate(CS.collections):
        paths = collection.get_paths()
        if len(paths) > 0:
            for path in paths:
                contours.append(path)
                values.append(data[n])
    diagonals = []
    for collection in CD.collections:
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
        for contour in contours:
            if diagonal.intersects_path(contour):
                mask.append(False)
                intersecting.append(contour)
            else:
                mask.append(True)
        elevations = np.array(values)[~np.array(mask)]
        for diagonal_segment in diagonal_segments:
            p1, p2 = diagonal_segment
            for n, contour in enumerate(intersecting):
                # https://stackoverflow.com/questions/38151445/
                contour_segments = list(zip(contour.vertices, contour.vertices[1:]))
                for contour_segment in contour_segments:
                    p3, p4 = contour_segment
                    intersection = intersect(p1, p2, p3, p4)
                    if intersection:
                        points.append(intersection)
                        response.append(elevations[n])
        l_points.append(points)
        l_response.append(response)
    return contours, l_points, l_response
