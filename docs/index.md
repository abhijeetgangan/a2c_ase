# a2c_ase

[![CI](https://github.com/abhijeetgangan/a2c_ase/actions/workflows/ci.yml/badge.svg)](https://github.com/abhijeetgangan/a2c_ase/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/abhijeetgangan/a2c_ase/branch/main/graph/badge.svg)](https://codecov.io/gh/abhijeetgangan/a2c_ase)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

An ASE-friendly implementation of the **amorphous-to-crystalline (a2c) workflow** for predicting crystal emergence from amorphous precursors using machine learning interatomic potentials.

## Overview

The a2c workflow predicts crystalline structures from amorphous materials through:

- Melt-quench molecular dynamics simulations
- Systematic subcell extraction and optimization
- Structure validation and space group analysis

Learn more: [Workflow Guide](user-guide/workflow.md)

## Getting Started

- **[Installation](getting-started/installation.md)**
- **[Quick Start](getting-started/quickstart.md)**
- **[Examples](examples/index.md)**

## References

1. Aykol, M., Merchant, A., Batzner, S. et al. **Predicting emergence of crystals from amorphous precursors with deep learning potentials.** *Nat Comput Sci* 5, 105–111 (2025). [DOI: 10.1038/s43588-024-00752-y](https://doi.org/10.1038/s43588-024-00752-y)

2. Reference implementation: [a2c-workflow](https://github.com/jax-md/jax-md/blob/main/jax_md/a2c/a2c_workflow.py)
