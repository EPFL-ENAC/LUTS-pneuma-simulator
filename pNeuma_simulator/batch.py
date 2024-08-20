import warnings

from joblib import Parallel

from pNeuma_simulator import params
from pNeuma_simulator.collision_exception import CollisionException
from pNeuma_simulator.main import main


def batch(seed: int, permutation: tuple):
    """
    Run a batch simulation with the given seed and permutation.
    Args:
        seed (int): The seed for random number generation.
        permutation (tuple): A tuple containing the number of cars and motorcycles.
    Returns:
        tuple: A tuple containing the simulation results for cars and motorcycles.
    """

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    n_cars, n_moto = permutation
    with Parallel(n_jobs=-1, prefer="processes") as parallel:
        try:
            item = main(n_cars, n_moto, seed, parallel, params.COUNT)
        except CollisionException:
            item = (None, None)
    return item
