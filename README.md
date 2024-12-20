
[![Project Status](https://img.shields.io/badge/status-under%20development-yellow)](https://github.com/EPFL-ENAC/pNeuma-simulator)

# pNeuma-simulator

The pNeuma-simulator is a Python-based tool for simulating multispecies urban traffic flow dynamics. It aims to provide an open-source, reproducible, and FAIR framework for traffic engineers and researchers in physics to analyze urban mobility scenarios.

## Installation

First, clone the repository. Then, install the package using

```bash
$ pip install -e .[dev]
```

For macOS users, replace `.[dev]` by `'.[dev]'`

## Usage

The refactored `pNEUMA-simulator.ipynb`, using the modular structure of the package, is located in [notebooks/pNEUMA-simulator.ipynb](pNeuma_simulator/notebooks/pNEUMA-simulator.ipynb).

## Building the docs

Documentation hosting is planned once the repository becomes public. To install the dependencies required to build the documentation locally, add the `docs` extra when installing, e.g. `pip install -e .[docs]` (the `dev` extra include the `docs` extra). Then, run

```bash
$ make docs
```
