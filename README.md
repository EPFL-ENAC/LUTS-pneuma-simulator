
[![Project Status](https://img.shields.io/badge/status-under%20development-yellow)](https://github.com/EPFL-ENAC/pNeuma-simulator)

# pNeuma-simulator

## Installation

First, clone the repository. Then, install the package using

```bash
$ pip install -e .[dev]
```

If installing the package on macOS, replace `.[dev]` by `'.[dev]'`

## Project structure

This repository contains the following files and folders:

```
📦 Repository
 ┣ 📁 .github : contain the github settings
 ┃ ┗  📁 ISSUE_TEMPLATE : contains issues templates
 ┃    ┗ 📜 *.yaml
 ┃ ┗  📁 workflows : contains CICD processes
 ┃    ┣ 📜 code_quality.yml : Ruff + Black + mypy
 ┃    ┗ 📜 tests.yml : pytest + CodeCov
 ┣ 📁 docs: contains the documentation.
 ┣ 📁 pNeuma_simulator: contains the project code.
 ┃ ┗ 📜 *.py
 ┣ 📁 test: contains the project tests.
 ┃ ┗ 📜 test_*.py
 ┣ 📜 .gitignore: lists the files/folders to ignore for git.
 ┣ 📜 pre-commit-config.yaml: configuration file for pre-commit.
 ┣ 📜 CITATION.cff: citation information.
 ┣ 📜 CODE_OF_CONDUCT.md: code of conduct.
 ┣ 📜 CONTRIBUTING.md: contributing guidelines.
 ┣ 📜 LICENSE: license file.
 ┣ 📜 pyproject.toml: project configuration file.
 ┣ 📜 README.md: markdown file containing the project's readme.
 ```

## Building the docs

```bash
$ cd docs && make html SPHINXOPTS="-d _build/doctrees"
```
