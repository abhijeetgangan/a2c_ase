# a2c_ase

[![CI](https://github.com/abhijeetgangan/a2c_ase/actions/workflows/ci.yml/badge.svg)](https://github.com/abhijeetgangan/a2c_ase/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/abhijeetgangan/a2c_ase/branch/main/graph/badge.svg)](https://codecov.io/gh/abhijeetgangan/a2c_ase)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

An ASE-friendly implementation of the amorphous-to-crystalline (a2c) workflow.

## Installation

### From Source

```bash
git clone https://github.com/abhijeetgangan/a2c_ase.git
cd a2c_ase
pip install .
```
## Usage
See [example/Si64.py](https://github.com/abhijeetgangan/a2c_ase/blob/main/example/Si64.py) for basic usage.

To use a specific calculator you need to install the corresponding package.

In the example above, MACE is used as the calculator, so you need to install the corresponding package.

```bash
pip install mace-torch
```

## Workflow Overview

1. **Initial Structure**: Generate a random atomic configuration with specified composition and volume.
2. **Melt-Quench**: Run MD simulation to create an amorphous structure.
3. **Subcell Extraction**: Identify potential crystalline motifs within the amorphous structure.
4. **Structure Optimization**: Relax subcells to find stable crystalline phases.
5. **Analysis**: Characterize discovered structures using symmetry analysis.

## References

1. Aykol, M., Merchant, A., Batzner, S. et al. Predicting emergence of crystals from amorphous precursors with deep learning potentials. Nat Comput Sci 5, 105–111 (2025). [DOI: 10.1038/s43588-024-00752-y](https://doi.org/10.1038/s43588-024-00752-y)
2. Reference implementation: [a2c-workflow](https://github.com/jax-md/jax-md/blob/main/jax_md/a2c/a2c_workflow.py)
