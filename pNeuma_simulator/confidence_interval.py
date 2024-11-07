import numpy as np


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
