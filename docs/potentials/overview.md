# Potentials Overview

`a2c_ase` includes built-in implementations of classical interatomic potentials that can be used for testing, development, and certain production workflows.

## Available Potentials

### Soft Sphere

A simple repulsive potential useful for initial packing optimization

[Learn more →](soft-sphere.md)

### Multi-Species Lennard-Jones

Lennard-Jones potential with support for binary and multi-component systems.

[Learn more →](multi-lj.md)

## Using External Calculators

### MACE

```python
from mace.calculators.foundations_models import mace_mp

calculator = mace_mp(
    model="medium",
    device="cuda",
    dtype="float32"
)
```

## Next Steps

- [Soft Sphere Potential](soft-sphere.md)
- [Multi-Lennard-Jones Potential](multi-lj.md)
- [API Reference](../api/potentials.md)
