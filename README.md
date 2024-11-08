
[![Project Status](https://img.shields.io/badge/status-under%20development-yellow)](https://github.com/EPFL-ENAC/pNeuma-simulator)

# pNeuma-simulator

## Installation

First, clone the repository. Then, install the package using

```bash
$ pip install -e .[dev]
```

If installing the package on macOS, replace `.[dev]` by `'.[dev]'`

## Usage

The refactored `pNEUMA-simulator.ipynb`, using the modular structure of the package, is located in [notebooks/pNEUMA-simulator.ipynb](pNeuma_simulator/notebooks/pNEUMA-simulator.ipynb).

## Building the docs

Currently, the documentation can only be created locally. To do so, `cd` to the `docs` subfolder and run

```bash
$ make html SPHINXOPTS="-d _build/doctrees"
```
