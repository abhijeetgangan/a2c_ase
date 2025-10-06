# Soft Sphere Potential

The soft sphere potential is a simple repulsive interaction used primarily for structure generation and packing optimization.

## Theory

### Potential Form

The pairwise energy between atoms \(i\) and \(j\) is:

\[
u_{ij}(r) = \begin{cases}
\frac{\epsilon}{\alpha} \left(1 - \frac{r_{ij}}{\sigma}\right)^\alpha & \text{if } r_{ij} < \sigma \\
0 & \text{if } r_{ij} \geq \sigma
\end{cases}
\]

Where:

- \(r_{ij}\): Distance between atoms \(i\) and \(j\)
- \(\sigma\): Particle diameter (cutoff distance)
- \(\epsilon\): Energy scale
- \(\alpha\): Stiffness exponent

### Force

The force is derived from the potential:

\[
\mathbf{F}_{ij} = -\nabla u_{ij} = -\frac{\epsilon}{\sigma} \left(1 - \frac{r_{ij}}{\sigma}\right)^{\alpha-1} \frac{\mathbf{r}_{ij}}{r_{ij}}
\]

for \(r_{ij} < \sigma\), and \(\mathbf{F}_{ij} = 0\) otherwise.

## Usage

### Basic Example

```python
from a2c_ase.potentials.soft_sphere import SoftSphere
from ase import Atoms
import numpy as np

# Create atoms
atoms = Atoms(
    'Si8',
    positions=np.random.random((8, 3)) * 10,
    cell=[10, 10, 10],
    pbc=True
)

# Create calculator
calculator = SoftSphere(
    sigma=2.5,      # Particle diameter (Å)
    epsilon=1.0,    # Energy scale (eV)
    alpha=2,        # Stiffness exponent
    skin=0.2       # Neighbor list skin
)

# Attach and compute
atoms.calc = calculator
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sigma` | float | 1.0 | Particle diameter (Å) |
| `epsilon` | float | 1.0 | Energy scale (eV) |
| `alpha` | int | 2 | Stiffness exponent |
| `skin` | float | 0.2 | Neighbor list skin (Å) |

## See Also

- [Multi-Lennard-Jones Potential](multi-lj.md)
- [Workflow Overview](../user-guide/workflow.md)
- [API Reference](../api/potentials.md)
