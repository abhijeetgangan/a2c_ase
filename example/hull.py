# %% [markdown]
# <details>
#   <summary>Dependencies</summary>
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "a2c_ase @ git+https://github.com/abhijeetgangan/a2c_ase.git",
#     "ase",
#     "numpy",
#     "pymatgen",
#     "tqdm",
#     "plotly",
#     "kaleido",
# ]
# ///
# </details>

# %% [markdown]
# Kob-Andersen Binary System - a2c Workflow

# Hull exploration with a2c workflow for the classic Kob-Andersen binary Lennard-Jones glass former.
# Uses reduced units (LJ natural units: sigma, epsilon).

# %% [markdown]
# ## Setup and Imports
#
# The Kob-Andersen model is a classic binary Lennard-Jones system that forms metallic glasses.
# We use reduced units: distances in sigma, energies in epsilon.

# %%
import os
from collections import Counter

import numpy as np
from ase import Atoms
from ase.build import bulk
from pymatgen.analysis.phase_diagram import PDEntry, PDPlotter, PhaseDiagram
from pymatgen.core.composition import Composition
from tqdm import tqdm

from a2c_ase.potentials.mlj import MultiLennardJones
from a2c_ase.runner import melt_quench_md, relax_unit_cell
from a2c_ase.utils import extract_crystallizable_subcells, random_packed_structure

IS_CI = os.getenv("CI") is not None

# %% [markdown]
# ## Kob-Andersen Parameters
#
# Classic binary LJ glass former with reduced units (dimensionless):
# - A particles (Ni): 80%, σ=1.0, ε=1.0 (reference)
# - B particles (P): 20%, σ=0.88, ε=0.5
# - Cross: σ_AB=0.8, ε_AB=1.5
# - Glass transition: T_g ≈ 0.435 (in reduced units)

# %%
# System configuration (80:20 A:B composition)
comp = Composition("Ni80P20")

# Cell in LJ units (sigma as length unit)
cell_size = 15.0
cell = np.array([[cell_size, 0.0, 0.0], [0.0, cell_size, 0.0], [0.0, 0.0, cell_size]])

# Kob-Andersen calculator (reduced/LJ units)
calculator = MultiLennardJones(
    sigma={"Ni": 1.0, "P": 0.88},  # LJ sigma in natural units
    epsilon={"Ni": 1.0, "P": 0.5},  # LJ epsilon in natural units
    cross_interactions={("Ni", "P"): {"sigma": 0.8, "epsilon": 1.5}},
    rc=2.5,  # Cutoff in units of sigma
    smooth=True,
)

# %% [markdown]
# ## Simulation Parameters
#
# All parameters in reduced LJ units (dimensionless):
# - Energy: ε (epsilon) = 1.0
# - Distance: σ (sigma) = 1.0
# - Temperature: T* = kT/ε
# - Time: τ = √(mσ²/ε) (dimensionless)

# %%
global_seed = 42
fmax = 0.01  # Force convergence in reduced units

# Reduce parameters for CI testing
max_iter = 20 if IS_CI else 100

# MD parameters (LJ units)
md_log_interval = 50
md_equi_steps = 100 if IS_CI else 2500
md_cool_steps = 100 if IS_CI else 2500
md_final_steps = 100 if IS_CI else 2500
md_T_high = 4.0  # High T* (reduced units, above glass transition ~0.8)
md_T_low = 0.4  # Low T* (reduced units, below glass transition)
md_time_step = 0.001  # Timestep in reduced units
md_friction = 1 / (100 * md_time_step)  # Friction coefficient

if IS_CI:
    print("Running in CI mode with reduced parameters")

# %% [markdown]
# ## Step 1: Generate Random Packed Structure
#
# Create initial random configuration with A and B particles.

# %%
packed_atoms, log_data = random_packed_structure(
    composition=comp,
    cell=cell,
    seed=global_seed,
    diameter=1.0,  # In units of sigma
    max_iter=max_iter,
    fmax=fmax,
    verbose=True,
    auto_diameter=False,
)
print(f"Generated packed structure: {packed_atoms}")
print(f"Number of Ni (A) atoms: {sum(1 for s in packed_atoms.symbols if s == 'Ni')}")
print(f"Number of P (B) atoms: {sum(1 for s in packed_atoms.symbols if s == 'P')}")

# %% [markdown]
# ## Step 2: Melt-Quench MD Simulation
#
# Heat to T*=4.0, quench to T*=0.4 (near T_g≈0.435).

# %%
amorphous_atoms, md_log = melt_quench_md(
    atoms=packed_atoms,
    calculator=calculator,
    equi_steps=md_equi_steps,
    cool_steps=md_cool_steps,
    final_steps=md_final_steps,
    T_high=md_T_high,
    T_low=md_T_low,
    time_step=md_time_step,
    friction=md_friction,
    seed=global_seed,
    verbose=True,
    log_interval=md_log_interval,
)
print(f"Amorphous structure ready: {amorphous_atoms}")

# %% [markdown]
# ## Step 3: Extract Crystallizable Subcells
#
# Divide the glass into overlapping subcells to search for local crystalline order.

# %%
crystallizable_cells = extract_crystallizable_subcells(
    atoms=amorphous_atoms,
    d_frac=0.25,  # Grid spacing (larger for binary system)
    n_min=2,
    n_max=16,  # Allow larger subcells for binary compounds
    cubic_only=False,  # Allow non-cubic structures
    allowed_atom_counts=None,  # Don't restrict by count
)
print(f"Extracted {len(crystallizable_cells)} candidate subcells")

# %% [markdown]
# ## Step 4: Optimize Subcells
#
# Relax each subcell to find stable crystalline phases.

# %%
relaxed_structures = []
print("Optimizing candidate structures...")

for atoms in tqdm(crystallizable_cells[:20] if IS_CI else crystallizable_cells):
    try:
        relaxed, logger = relax_unit_cell(
            atoms=atoms, calculator=calculator, max_iter=max_iter, fmax=fmax, verbose=False
        )

        final_energy = relaxed.get_potential_energy()
        energy_per_atom = final_energy / len(relaxed)

        relaxed_structures.append((relaxed, energy_per_atom, final_energy))
    except Exception as e:
        print(f"Optimization failed: {e}")
        continue

print(f"Successfully optimized {len(relaxed_structures)} structures")

# %% [markdown]
# ## Step 5: Construct Convex Hull
#
# Determine thermodynamic stability using pymatgen's phase diagram.

# %%
# Build convex hull with proper composition handling
entries = []
for atoms, _, total_e in relaxed_structures:
    # Get composition from symbol counts (more explicit and correct)
    symbol_counts = Counter(atoms.get_chemical_symbols())
    comp_obj = Composition(symbol_counts)  # e.g., {'Ni': 2, 'P': 1} -> Ni2P
    entries.append(PDEntry(comp_obj, total_e))

# Compute reference energies using proper crystal structures
print("\nComputing reference energies...")

# Pure Ni: FCC structure
ni_fcc = bulk("Ni", "fcc", a=1.5)
ni_relaxed, _ = relax_unit_cell(ni_fcc, calculator, max_iter=max_iter, fmax=fmax, verbose=False)
e_ni_total = ni_relaxed.get_potential_energy()
print(f"Ni (FCC): {e_ni_total / len(ni_relaxed):.4f} ε/atom")

# Pure P: FCC structure
p_fcc = bulk("P", "fcc", a=1.3)
p_relaxed, _ = relax_unit_cell(p_fcc, calculator, max_iter=max_iter, fmax=fmax, verbose=False)
e_p_total = p_relaxed.get_potential_energy()
print(f"P (FCC): {e_p_total / len(p_relaxed):.4f} ε/atom")

# Ni4P structure
alat = 4.0
ni4p = Atoms(
    "Ni4P",
    positions=[
        [0.0, 0.0, 0.0],
        [0.0, alat / 2, alat / 2],
        [alat / 2, 0.0, alat / 2],
        [alat / 2, alat / 2, 0.0],
        [alat / 2, alat / 2, alat / 2],
    ],
    cell=[alat, alat, alat],
    pbc=True,
)
ni4p_relaxed, _ = relax_unit_cell(ni4p, calculator, max_iter=max_iter, fmax=fmax, verbose=False)
e_ni4p_total = ni4p_relaxed.get_potential_energy()
print(f"Ni4P (Rocksalt): {e_ni4p_total / len(ni4p_relaxed):.4f} ε/atom")

# Add references with TOTAL energies
entries.extend(
    [
        PDEntry(Composition("Ni"), e_ni_total),
        PDEntry(Composition("P"), e_p_total),
        PDEntry(Composition("Ni4P"), e_ni4p_total),
    ]
)
pd = PhaseDiagram(entries)

for i, (atoms, e_per_atom, total_e) in enumerate(relaxed_structures):
    symbol_counts = Counter(atoms.get_chemical_symbols())
    comp_obj = Composition(symbol_counts)
    entry = PDEntry(comp_obj, total_e)
    e_above_hull = pd.get_e_above_hull(entry)

    print(
        f"{i + 1:2d}. {comp_obj.reduced_formula:10s} | "
        f"E/atom: {e_per_atom:8.4f} ε | "
        f"E_hull: {e_above_hull:8.4f} ε/atom"
    )

print(f"\nTotal structures analyzed: {len(relaxed_structures)}")
print(f"Total entries in phase diagram: {len(entries)}")

# Plot and save phase diagram
plotter = PDPlotter(pd, show_unstable=1000.0, backend="plotly")  # Very large value to show all
plotter.get_plot()
plotter.show()
