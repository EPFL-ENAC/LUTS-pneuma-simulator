import numpy as np
from numpy import where


def egress(alphas, indices, counts, cond, rng) -> float:
    """Find interior egress point in the horizon.

    Args:
        alphas (list): angle choice set in radians.
        indices (list): indices of non zero intervals.
        counts (list): difference between consecutive indices.
        cond (list): condition, boolean list.

    Returns:
        float: egreess point in radians
    """
    supremum = max(counts[cond])
    argmax = np.argwhere(counts[cond] == supremum)
    start = indices[cond][rng.choice(argmax)]
    a0 = alphas[start + np.round((supremum - 1) / 2).astype(int)][0]
    return a0


def target(alphas, f_a, d_max, rng) -> tuple:
    """Find consecutive runs and length of runs with condition.

    Args:
        alphas (list): angle choice set in radians.
        f_a (list): distance to collision values in meters.
        d_max (int): distance to horizon in meters.

    Returns:
        tuple: target and desired directions in radians
    """
    # Find consecutive runs and length of runs with condition
    # https://stackoverflow.com/questions/71746585
    # set equal, consecutive elements to 0
    intervals = np.hstack([True, ~np.isclose(f_a[:-1], f_a[1:])])
    # get indices of non zero elements
    indices = np.flatnonzero(intervals)
    # difference between consecutive indices are the length
    counts = np.diff(indices, append=len(f_a))
    cond = (counts > 1) & (f_a[indices] == max(f_a))
    # determine a0
    if len(counts[cond]) > 0:
        # check for saturated flanks
        if (f_a[0] == d_max) | (f_a[-1] == d_max):
            # check left flank
            if f_a[0] == d_max:
                cond[0] = False
            # check right flank
            if f_a[-1] == d_max:
                cond[-1] = False
            # pick saturated interior point
            if len(counts[cond]) > 0:
                a0 = egress(alphas, indices, counts, cond, rng)
            # keep steady when only the flanks are saturated
            else:
                a0 = 0
        else:
            a0 = egress(alphas, indices, counts, cond, rng)
    else:
        maxima = where(f_a == f_a.max())[0]
        if len(maxima) == 1:
            maximum = maxima[0]
        else:
            maximum = rng.choice(maxima)
        a0 = alphas[maximum]
    # Objective: squared distance to destination (for performance)
    d_a = d_max**2 + f_a**2 - 2 * d_max * f_a * np.cos(a0 - alphas)
    # minimize distance to destination
    minima = where(d_a == d_a.min())[0]
    if len(minima) == 1:
        minimum = minima[0]
    else:
        minimum = rng.choice(minima)
    # chosen direction
    a_des = alphas[minimum]
    return (a0, a_des)
