# Examples

Practical examples demonstrating the a2c workflow.

## Available Examples

### Si64.py - Silicon Crystallization

Predicts the crystal structure of silicon from an amorphous precursor using MACE.

**Source**: [`example/Si64.py`](https://github.com/abhijeetgangan/a2c_ase/blob/main/example/Si64.py)

**What it does**:

1. Generates random Si64 structure (64 atoms)
2. Melts at 2000K using MACE potential
3. Quenches to 300K to create amorphous structure
4. Extracts crystalline subcells
5. Optimizes each subcell
6. Symmetry analysis the structure to match the diamond cubic structure

**Run it**:
```bash
git clone https://github.com/abhijeetgangan/a2c_ase.git
cd a2c_ase
pip install -e . && pip install mace-torch
python example/Si64.py
```

## Key Parameters Reference

| Parameter | Typical Value | Purpose |
|-----------|---------------|---------|
| `T_high` | 1500-2500 K | Melting temperature |
| `T_low` | 300 K | Final temperature |
| `equi_steps` | 2500 | High-T equilibration |
| `cool_steps` | 2500 | Cooling duration |
| `d_frac` | 0.2 | Subcell grid spacing |
| `n_min, n_max` | 2, 8 | Atom count range |
| `fmax` | 0.01 eV/Å | Force convergence |

---

## Next Steps

- **Understand the workflow**: [User Guide](../user-guide/workflow.md)
- **API documentation**: [Runner](../api/runner.md) | [Utils](../api/utils.md)
- **Quick overview**: [Quick Start](../getting-started/quickstart.md)
