import itertools
import json
import os
import sys
import warnings
import zipfile

import numpy as np
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from numpy import arange
from tqdm.notebook import tqdm

from pNeuma_simulator.results import zipdir
from pNeuma_simulator.simulate import batch

warnings.filterwarnings("ignore")

path = "./output/"
os.makedirs(path, exist_ok=True)
# Combinatorial configurations
n_veh = 8
scale = 2
l_cars = scale * arange(1, n_veh, 1)
l_moto = scale * arange(0, n_veh - 1, 1)
permutations = list(itertools.product(l_cars, l_moto))


def execute(n_cars, n_moto, epochs=64, n_jobs=64, n_threads=1, distributed=True, stochastic=True, save=True):
    name = (n_cars, n_moto)
    default_rng = np.random.default_rng(1024)
    seeds = default_rng.integers(1e8, size=epochs * len(permutations))
    for n, permutation in tqdm(enumerate(permutations)):
        start = n * epochs
        end = (n + 1) * epochs
        if permutation == name:
            # https://stackoverflow.com/questions/67891651/
            with Parallel(n_jobs=n_jobs) as parallel:
                items = parallel(
                    delayed(batch)(seed, permutation, n_threads, distributed, stochastic) for seed in seeds[start:end]
                )
            # https://stackoverflow.com/questions/67495271/
            get_reusable_executor().shutdown(wait=True)
            if save:
                # Dump to JSONL https://jsonlines.org/examples/
                # https://stackoverflow.com/questions/38915183/
                with open(f"{path}{permutation}.jsonl", "w") as outfile:
                    for item in items:
                        json.dump(item, outfile)
                        outfile.write("\n")
                # Compress as Zip and delete original JSONL
                with zipfile.ZipFile(f"{path}{permutation}_da.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
                    zipdir(path, zipf)
            print(permutation)


if __name__ == "__main__":
    execute(int(sys.argv[1]), int(sys.argv[2]))
