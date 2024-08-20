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
