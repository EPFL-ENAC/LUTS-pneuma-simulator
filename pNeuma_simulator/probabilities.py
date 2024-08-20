import numpy as np
from scipy.stats import distributions


def truncated_rvs(
    rng,
    size: int,
    dist,
    x_min: float,
    x_max: float,
    c: float,
    loc: float,
    scale: float,
):
    """
    Generate truncated random variables.

    Args:
        rng (object): Random number generator.
        size (int): Number of random variables to generate.
        dist (object): Probability distribution object.
        x_min (float): Minimum value for truncation.
        x_max (float): Maximum value for truncation.
        c (float): Shape parameter for the distribution.
        loc (float): Location parameter for the distribution.
        scale (float): Scale parameter for the distribution.

    Returns:
        numpy.ndarray: Array of truncated random variables.
    """
    # https://stackoverflow.com/questions/47933019
    low = dist.cdf(x_min, c, loc=loc, scale=scale)
    high = dist.cdf(x_max, c, loc=loc, scale=scale)

    return dist.ppf(
        rng.uniform(low=low, high=high, size=size),
        c,
        loc=0,
        scale=scale,
    )


def synthetic_fd(n_veh: int, random_state, mode: str = "Car", distributed: bool = True):
    """
    Generate synthetic fundamental diagram data.

    Args:
        n_veh (int): Number of vehicles.
        random_state (object): Random number generator.
        mode (str, optional): Mode of transportation. Defaults to "Car".
        distributed (bool, optional): Flag indicating if the data should be distributed. Defaults to True.

    Returns:
        tuple: Tuple containing the following:
            - list: List of marginal distributions.
            - numpy.ndarray: Array of lambda values.
            - numpy.ndarray: Array of desired speeds.
            - numpy.ndarray: Array of jam spacings.
    """
    factor = 3.6
    lam, v0, d = 0, 0, 0
    marginals = []
    marginal_dists = [
        distributions.fisk,
        distributions.lognorm,
        distributions.weibull_min,
    ]
    if mode == "Car":
        args = [
            (4.751065953663232, 0, 1.257554949836412),
            (0.2206026972757819, 0, 29.814414236513073),
            (2.002603415738683, 0, 1.4203922933206412),
        ]
        bounds = [
            (0.6968633413268207, 2.4866413425909792),
            (20.825241115718224, 43.3883779433966),
            (0.19690785189237897, 2.321541307355339),
        ]
    else:
        args = [
            (3.0162661900328684, 0, 2.1577514618921123),
            (0.27640817683206864, 0, 31.783886290615683),
            (1.2934090505801255, 0, 1.3335939613574372),
        ]
        bounds = [
            (1.038156373262415, 5.734572922682492),
            (20.601924472034902, 47.33181402666869),
            (0.13632279926915733, 3.2677733493753554),
        ]
    for r, marginal_dist in enumerate(marginal_dists):
        marginal = marginal_dist(*args[r])
        marginals.append(marginal)
    if distributed:
        synthetic_data = []
        for i in range(3):
            rng = random_state
            size = n_veh
            dist = marginal_dists[i]
            x_min, x_max = bounds[i]
            c, loc, scale = args[i]
            values = truncated_rvs(rng, size, dist, x_min, x_max, c, loc, scale)
            synthetic_data.append(values)
        # slope at jam spacing s^-1
        lam = synthetic_data[0]
        # desired speed in m/s
        v0 = synthetic_data[1] / factor
        # jam spacing m
        d = synthetic_data[2].copy()
    else:
        lam = np.repeat(marginals[0].mean(), n_veh)
        v0 = np.repeat(marginals[1].mean() / factor, n_veh)
        d = np.repeat(marginals[2].mean(), n_veh)

    return marginals, lam, v0, d
