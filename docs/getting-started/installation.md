# Installation

## Requirements

- Python 3.10 or higher
- pip or uv package manager

## From Source

The current recommended way to install `a2c_ase` is from source:

=== "pip"

    ```bash
    git clone https://github.com/abhijeetgangan/a2c_ase.git
    cd a2c_ase
    pip install .
    ```

=== "uv"

    ```bash
    git clone https://github.com/abhijeetgangan/a2c_ase.git
    cd a2c_ase
    uv pip install .
    ```

## With Development Dependencies

If you plan to contribute or run tests:

=== "pip"

    ```bash
    pip install -e ".[dev,test]"
    ```

=== "uv"

    ```bash
    uv pip install -e ".[dev,test]"
    ```

## Calculator Dependencies

`a2c_ase` works with any ASE-compatible calculator. You'll need to install the specific calculator package you want to use.

### Other Calculators

For the MACE machine learning potential:

=== "pip"

    ```bash
    pip install mace-torch
    ```

=== "uv"

    ```bash
    uv pip install mace-torch
    ```

## Development Setup

For contributors, set up the development environment:

1. Clone the repository:
```bash
git clone https://github.com/abhijeetgangan/a2c_ase.git
cd a2c_ase
```

2. Install with development dependencies:

=== "pip"

    ```bash
    pip install -e ".[dev,test]"
    ```

=== "uv"

    ```bash
    uv pip install -e ".[dev,test]"
    ```

3. Set up pre-commit hooks:
```bash
pre-commit install
```

4. Run tests to verify everything works:
```bash
pytest
```

## Troubleshooting

### Calculator Not Found

Make sure you've installed the calculator package you're trying to use. For example, for MACE:

=== "pip"

    ```bash
    pip install mace-torch
    ```

=== "uv"

    ```bash
    uv pip install mace-torch
    ```

## Next Steps

Now that you have `a2c_ase` installed, proceed to the [Quick Start Guide](quickstart.md) to run your first workflow!
