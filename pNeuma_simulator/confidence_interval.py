import numpy as np


def confidence_interval(data, rng, setting="sem"):
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
