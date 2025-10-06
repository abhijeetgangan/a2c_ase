# Examples

Practical examples demonstrating the a2c workflow.

## Available Examples

### Si64.py - Silicon Crystallization

Complete a2c workflow with executed outputs from CI.

[View Executed Notebook →](Si64.ipynb){ .md-button .md-button--primary }

[Source on GitHub →](https://github.com/abhijeetgangan/a2c_ase/blob/main/example/Si64.py){ .md-button }

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
