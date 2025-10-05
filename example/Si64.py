"""Test script for the a2c workflow for Si64 using MACE."""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "a2c_ase @ git+https://github.com/abhijeetgangan/a2c_ase.git",
#     "ase",
#     "numpy",
#     "pymatgen",
#     "tqdm",
#     "mace-torch",
# ]
# ///

import os
from collections import defaultdict

import numpy as np
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
from tqdm import tqdm

from a2c_ase.runner import melt_quench_md, relax_unit_cell
from a2c_ase.utils import extract_crystallizable_subcells, random_packed_structure

# Check if running in CI for fast testing
IS_CI = os.getenv("CI") is not None

# System configuration
comp = Composition("Si64")
cell = np.array([[11.1, 0.0, 0.0], [0.0, 11.1, 0.0], [0.0, 0.0, 11.1]])

# Optimization parameters
global_seed = 42
fmax = 0.01  # Force convergence criterion in eV/Å
max_iter = 10 if IS_CI else 100

# Molecular dynamics parameters
md_log_interval = 50
md_equi_steps = 10 if IS_CI else 2500  # High temperature equilibration steps
md_cool_steps = 10 if IS_CI else 2500  # Cooling steps
md_final_steps = 10 if IS_CI else 2500  # Low temperature equilibration steps
md_T_high = 2000.0  # Initial melting temperature (K)
md_T_low = 300.0  # Final temperature (K)
md_time_step = 2.0  # fs
md_friction = 0.01  # Langevin friction

if IS_CI:
    print("Running in CI mode with reduced parameters for fast testing")

# Initialize MACE calculator
mace_checkpoint_url = (
    "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model"
)
device = "cpu" if IS_CI else "cuda"
calculator = mace_mp(model=mace_checkpoint_url, device=device, dtype="float32", enable_cueq=False)

# Generate initial random structure
packed_atoms, log_data = random_packed_structure(
    composition=comp,
    cell=cell,
    seed=global_seed,
    fmax=fmax,
    max_iter=max_iter,
    verbose=True,
    auto_diameter=True,
    trajectory_file=None,
)
print(f"Soft sphere packed structure is ready {packed_atoms}")

# Run melt-quench MD to generate amorphous structure
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
    trajectory_file=None,
    seed=global_seed,
    verbose=True,
    log_interval=md_log_interval,
)
print(f"Final amorphous structure is ready {amorphous_atoms}")

# Extract potential crystalline regions
crystallizable_cells = extract_crystallizable_subcells(
    atoms=amorphous_atoms,
    d_frac=0.2,  # Size of subcell as fraction of original cell
    n_min=2,  # Min atoms per subcell
    n_max=8,  # Max atoms per subcell
    cubic_only=True,
    allowed_atom_counts=[2, 4, 8],
)
print(f"Generated {len(crystallizable_cells)} candidate structures for crystallization")

# Relax all candidate structures and find the lowest energy one
relaxed_atoms_list = []
print("Relaxing candidate structures...")
for atoms in tqdm(crystallizable_cells):
    # Relax the structure
    relaxed_atoms, logger = relax_unit_cell(
        atoms=atoms, calculator=calculator, max_iter=max_iter, fmax=fmax, verbose=False
    )

    # Get final energy and pressure
    final_energy = relaxed_atoms.get_potential_energy()
    final_pressure = -np.trace(relaxed_atoms.get_stress(voigt=False)) / 3.0

    # Store the relaxed structure and its properties
    relaxed_atoms_list.append((relaxed_atoms, logger, final_energy, final_pressure))

# Find lowest energy structure
lowest_e_candidate = min(relaxed_atoms_list, key=lambda x: x[-2] / len(x[0]))
lowest_e_atoms, lowest_e_logger, lowest_e_energy, lowest_e_pressure = lowest_e_candidate

# Convert to pymatgen structure for space group analysis
pymatgen_struct = Structure(
    lattice=lowest_e_atoms.get_cell(),
    species=lowest_e_atoms.get_chemical_symbols(),
    coords=lowest_e_atoms.get_positions(),
    coords_are_cartesian=True,
)

# Analyze space group
spg = SpacegroupAnalyzer(pymatgen_struct)
print("Space group of predicted crystallization product:", spg.get_space_group_symbol())
print(
    f"Final energy: {lowest_e_energy:.4f} eV, "
    f"Energy per atom: {lowest_e_energy / len(lowest_e_atoms):.4f} eV/atom"
)
print(f"Final pressure: {lowest_e_pressure:.6f} eV/Å³")

# Count frequency of space groups across all candidates
spg_counter = defaultdict(lambda: 0)

for s in relaxed_atoms_list:
    pymatgen_struct = Structure(
        lattice=s[0].get_cell(),
        species=s[0].get_chemical_symbols(),
        coords=s[0].get_positions(),
        coords_are_cartesian=True,
    )
    try:
        sp = SpacegroupAnalyzer(pymatgen_struct).get_space_group_symbol()
        spg_counter[sp] += 1
    except TypeError:
        continue


print("All space groups encountered:", dict(spg_counter))

# Compare to reference diamond structure
si_diamond = bulk("Si", "diamond", a=5.43)
pymatgen_ref_struct = Structure(
    lattice=si_diamond.get_cell(),
    species=si_diamond.get_chemical_symbols(),
    coords=si_diamond.get_positions(),
    coords_are_cartesian=True,
)
print(
    "Prediction matches diamond-cubic Si?",
    StructureMatcher().fit(pymatgen_struct, pymatgen_ref_struct),
)

# Save the lowest energy structure
lowest_e_atoms.write("final_crystal_structure.cif")
print("Lowest energy structure saved to final_crystal_structure.cif")
