import numpy as np
from scipy.stats import binned_statistic

from pNeuma_simulator import params
from pNeuma_simulator.confidence_interval import confidence_interval
from pNeuma_simulator.gang import decay
from pNeuma_simulator.initialization import ov


def percolate(items, n_moto, start=1):
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
