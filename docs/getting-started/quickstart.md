# Quick Start

Get started with `a2c_ase` in minutes.

## Prerequisites

- `a2c_ase` installed ([Installation Guide](installation.md))
- A calculator installed (MACE)

=== "uv"
    ```bash
    uv pip install mace-torch
    ```

=== "pip"
    ```bash
    pip install mace-torch
    ```

## Five-Step Workflow

### 1. Generate Random Structure

```python
from a2c_ase.utils import random_packed_structure
from pymatgen.core.composition import Composition
import numpy as np

comp = Composition("Si64")
cell = np.array([[11.1, 0.0, 0.0], [0.0, 11.1, 0.0], [0.0, 0.0, 11.1]])

atoms, log = random_packed_structure(comp, cell, auto_diameter=True)
```

### 2. Run Melt-Quench MD

```python
from a2c_ase.runner import melt_quench_md

amorphous, md_log = melt_quench_md(
    atoms, calculator,
    T_high=2000.0, T_low=300.0,
    equi_steps=2500, cool_steps=2500, final_steps=2500
)
```

### 3. Extract Subcells

```python
from a2c_ase.utils import extract_crystallizable_subcells

subcells = extract_crystallizable_subcells(
    amorphous, d_frac=0.2, n_min=2, n_max=8, cubic_only=True
)
```

### 4. Optimize Structures

```python
from a2c_ase.runner import relax_unit_cell

for subcell in subcells:
    relaxed, logger = relax_unit_cell(subcell, calculator, fmax=0.01)
    # Analyze relaxed structure
```

### 5. Analyze Results

```python
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer

# Find lowest energy and determine space group
# Compare to reference structures
```

---

## Complete Example

See the full working example: [**example/Si64.py**](../examples/index.md)

For silicon, the workflow predicts the diamond cubic structure (Fd-3m).

---

## Next Steps

- **Understand the workflow**: [User Guide](../user-guide/workflow.md)
- **API documentation**: [Runner](../api/runner.md) | [Utils](../api/utils.md)
- **Full example**: [Si64.py Example](../examples/index.md)

---

## Tips

!!! warning "Memory"
    Large systems with many subcells need significant memory. Use larger `d_frac` or smaller `n_max` to reduce subcell count.
