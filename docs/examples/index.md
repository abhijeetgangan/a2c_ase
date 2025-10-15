# Examples

Practical examples demonstrating the a2c workflow.

---

## 1. Si64 - Silicon Crystallization

Complete a2c workflow for predicting silicon's crystal structure from an amorphous precursor.

[Tutorial Docs](Si64.ipynb){ .md-button }
[Source Code](https://github.com/abhijeetgangan/a2c_ase/blob/main/example/Si64.py){ .md-button }

**Demonstrates**: Random structure generation, melt-quench MD, subcell extraction, optimization, and space group analysis.

---

## 2. Hull - Kob-Andersen Binary System Hull

Hull exploration with a2c workflow for the classic Kob-Andersen binary Lennard-Jones glass former (Ni80P20).

[Tutorial Docs](hull.ipynb){ .md-button }
[Source Code](https://github.com/abhijeetgangan/a2c_ase/blob/main/example/hull.py){ .md-button }

**Demonstrates**: Binary systems, metal/LJ units, custom cross-interactions, composition analysis.

---

## 3. Cell Extraction - Sodium Crystallization Analysis

Extract and analyze crystallizable subcells from a pre-generated amorphous sodium structure using ML potential.

[Tutorial Docs](cell_extraction.ipynb){ .md-button }
[Source Code](https://github.com/abhijeetgangan/a2c_ase/blob/main/example/cell_extraction.py){ .md-button }

**Demonstrates**: Loading existing structures, subcell extraction, ML potential, space group distribution analysis, visualization.

---

## See Also

- **Understand the workflow**: [User Guide](../user-guide/workflow.md)
- **API documentation**: [Runner](../api/runner.md) | [Utils](../api/utils.md)
- **Quick overview**: [Quick Start](../getting-started/quickstart.md)
