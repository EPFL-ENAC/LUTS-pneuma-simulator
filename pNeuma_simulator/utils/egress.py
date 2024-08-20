import numpy as np


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
