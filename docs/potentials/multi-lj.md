# Multi-Lennard-Jones Potential

The Multi-Lennard-Jones calculator implements the classic Lennard-Jones potential with support for multiple chemical species and custom mixing rules.

## Theory

### Potential Form

The Lennard-Jones potential between atoms \(i\) and \(j\) is:

\[
u_{ij}(r) = 4\epsilon_{ij} \left[\left(\frac{\sigma_{ij}}{r}\right)^{12} - \left(\frac{\sigma_{ij}}{r}\right)^6\right]
\]

Where:

- \(\sigma_{ij}\): Distance at which potential is zero
- \(\epsilon_{ij}\): Depth of potential well
- \(r\): Interatomic distance

### Force

\[
\mathbf{F}_{ij} = 24\epsilon_{ij} \left[2\left(\frac{\sigma_{ij}}{r}\right)^{12} - \left(\frac{\sigma_{ij}}{r}\right)^6\right] \frac{\mathbf{r}_{ij}}{r^2}
\]

## Usage

### Single Species

```python
from a2c_ase.potentials.mlj import MultiLennardJones
from ase import Atoms

# Create atoms
atoms = Atoms('Ar10', positions=..., cell=..., pbc=True)

# Create calculator
calculator = MultiLennardJones(
    epsilon=1.0,    # eV
    sigma=3.4,      # Å
    rc=10.0        # Cutoff (Å)
)

atoms.calc = calculator
energy = atoms.get_potential_energy()
```

### Multiple Species

```python
# Dictionary specification
calculator = MultiLennardJones(
    epsilon={"Fe": 0.5, "B": 0.3},
    sigma={"Fe": 2.5, "B": 1.8},
    rc=10.0
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epsilon` | float/dict | 1.0 | Well depth (eV) |
| `sigma` | float/dict | 1.0 | Zero-crossing distance (Å) |
| `rc` | float | None | Cutoff distance (auto: 3σ) |
| `ro` | float | None | Smooth cutoff onset (auto: 0.66rc) |
| `smooth` | bool | False | Use smooth cutoff |
| `mixing_rule` | str | "lorentz_berthelot" | Mixing rule |
| `cross_interactions` | dict | None | Explicit cross-interactions |

## Mixing Rules

### Lorentz-Berthelot (Default)

\[
\sigma_{ij} = \frac{\sigma_i + \sigma_j}{2}
\]

\[
\epsilon_{ij} = \sqrt{\epsilon_i \epsilon_j}
\]

```python
calculator = MultiLennardJones(
    epsilon={"A": 1.0, "B": 1.5},
    sigma={"A": 1.0, "B": 0.8},
    mixing_rule="lorentz_berthelot"
)
```

### Geometric

\[
\sigma_{ij} = \sqrt{\sigma_i \sigma_j}
\]

\[
\epsilon_{ij} = \sqrt{\epsilon_i \epsilon_j}
\]

```python
calculator = MultiLennardJones(
    epsilon={"A": 1.0, "B": 1.5},
    sigma={"A": 1.0, "B": 0.8},
    mixing_rule="geometric"
)
```

### Custom Cross-Interactions

Override mixing rules for specific pairs:

```python
calculator = MultiLennardJones(
    epsilon={"A": 1.0, "B": 1.5},
    sigma={"A": 1.0, "B": 0.88},
    mixing_rule="lorentz_berthelot",
    cross_interactions={
        ("A", "B"): {"sigma": 0.8, "epsilon": 1.5}
    }
)
```

## Cutoff Methods

### Shifted Cutoff (Default)

Simple shift to make energy continuous:

\[
u_{\text{shifted}}(r) = u(r) - u(r_c)
\]

```python
calculator = MultiLennardJones(
    epsilon=1.0,
    sigma=3.4,
    rc=10.0,
    smooth=False  # Default
)
```

### Smooth Cutoff

Smoothly goes to zero between `ro` and `rc`:

\[
u_{\text{smooth}}(r) = u(r) \times S(r)
\]

where \(S(r)\) is a switching function:

\[
S(r) = \begin{cases}
1 & r < r_o \\
\frac{(r_c - r)^2(r_c + 2r - 3r_o)}{(r_c - r_o)^3} & r_o \leq r < r_c \\
0 & r \geq r_c
\end{cases}
\]

```python
calculator = MultiLennardJones(
    epsilon=1.0,
    sigma=3.4,
    rc=10.0,
    ro=6.6,      # Onset of cutoff
    smooth=True
)
```

## Examples

### Argon (Noble Gas)

```python
from ase.lattice.cubic import FaceCubicFactory
from a2c_ase.potentials.mlj import MultiLennardJones

# Argon parameters (from literature)
calc = MultiLennardJones(
    epsilon=0.0103,  # eV (0.0103 eV ≈ 120 K)
    sigma=3.405,     # Å
    rc=10.0
)

# Create FCC lattice
atoms = FaceCubicFactory()(symbol='Ar', size=(3,3,3), latticeconstant=5.26)
atoms.calc = calc

energy = atoms.get_potential_energy()
print(f"Energy per atom: {energy/len(atoms):.4f} eV")
```

### Kob-Andersen Binary Mixture

Classic binary LJ mixture:

```python
calc = MultiLennardJones(
    epsilon={"A": 1.0, "B": 0.5},
    sigma={"A": 1.0, "B": 0.88},
    mixing_rule="lorentz_berthelot",
    cross_interactions={
        ("A", "B"): {"sigma": 0.8, "epsilon": 1.5}
    },
    rc=3.0,
    smooth=True
)
```

## See Also

- [Soft Sphere Potential](soft-sphere.md)
- [Potentials Overview](overview.md)
- [Workflow Overview](../user-guide/workflow.md)
- [API Reference](../api/potentials.md)
