# %% [markdown]
# <details>
#   <summary>Dependencies</summary>
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "a2c-ase>=0.0.3",
#     "numpy",
#     "pymatgen",
#     "tqdm",
#     "mace-torch",
#     "matplotlib",
# ]
# ///
# </details>

# %% [markdown]
# Sodium Crystallization from Amorphous Phase - Cell Extraction Example

# This example demonstrates how to extract crystallizable subcells from an amorphous
# sodium structure, relax them using the MACE machine learning potential, and analyze
# the resulting space group distribution. This is a simplified workflow that focuses on
# the cell extraction and analysis steps without running the full melt-quench MD simulation.

# %% [markdown]
# ## Setup and Imports
#
# We'll use MACE for accurate energy and force calculations, pymatgen for space group
# analysis, and matplotlib for visualization.

# %%
import os
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from ase.io import read
from mace.calculators.foundations_models import mace_mp  # type: ignore
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

from a2c_ase.runner import relax_unit_cell
from a2c_ase.utils import extract_crystallizable_subcells

# Set matplotlib backend for CI
if os.getenv("CI") is not None:
    matplotlib.use("Agg")

# %% [markdown]
# ## Configuration
#
# Define paths and simulation parameters.
# In CI mode, we use reduced parameters for faster testing.

# %%
IS_CI = os.getenv("CI") is not None

# Get the path to the data file relative to this script
try:
    script_dir = Path(__file__).parent
except NameError:
    script_dir = Path.cwd()

# Search for data file in multiple locations
for parent in [script_dir.parent, script_dir, script_dir.parent.parent]:
    data_file = parent / "data" / "Na_2000.xyz"
    if data_file.exists():
        break
else:
    msg = f"Could not find data/Na_2000.xyz. Run from project root. CWD: {Path.cwd()}"
    raise FileNotFoundError(msg) from None

# Relaxation parameters
max_iter = 20 if IS_CI else 200  # Maximum optimization steps
fmax = 0.01 if IS_CI else 0.05  # Force convergence criterion in eV/Å

if IS_CI:
    print("Running in CI mode with reduced parameters for fast testing")

# %% [markdown]
# ## Step 1: Load Amorphous Structure
#
# Load the pre-generated amorphous sodium structure from the data directory.
# This structure was obtained from a melt-quench simulation.

# %%
amorphous_atoms = read(data_file, index="0")
print(f"Loaded structure with {len(amorphous_atoms)} atoms")
print(f"Cell dimensions: {amorphous_atoms.cell.lengths()}")  # type: ignore

# %% [markdown]
# ## Step 2: Extract Crystallizable Subcells
#
# Search for periodic subcells within the amorphous structure that could represent
# crystalline unit cells. The algorithm uses a grid-based search to identify regions
# with translational symmetry.

# %%

crystallizable_cells = extract_crystallizable_subcells(
    atoms=amorphous_atoms,  # type: ignore
    d_frac=0.15,  # Grid spacing as fraction of cell dimensions
    n_min=2,  # Minimum grid divisions per dimension
    n_max=12,  # Maximum grid divisions per dimension
    cubic_only=False,  # Allow non-cubic structures
    allowed_atom_counts=None,  # No restriction on number of atoms per cell
)

print(f"Found {len(crystallizable_cells)} crystallizable cells")

# %% [markdown]
# ## Step 3: Initialize MACE Calculator

# %%
device = "cpu" if IS_CI else "cuda"
calculator = mace_mp(model="small-omat-0", device=device, dtype="float32")
print(f"MACE calculator initialized on {device}")

# %% [markdown]
# ## Step 4: Relax Candidate Structures
#
# For each extracted subcell, we perform a full structure relaxation (both atomic positions
# and cell parameters) using the FIRE optimizer. This identifies the energetically favorable
# crystalline structures.

# %%
space_groups = []
relaxed_cells = []

print("Relaxing structures...")
for atoms in tqdm(crystallizable_cells[:20] if IS_CI else crystallizable_cells):
    relaxed_cell, log_dict = relax_unit_cell(
        atoms=atoms,
        calculator=calculator,
        max_iter=max_iter,
        fmax=fmax,
        verbose=False,
    )

    # Convert to pymatgen and get space group
    adaptor = AseAtomsAdaptor()
    structure = adaptor.get_structure(relaxed_cell)
    sga = SpacegroupAnalyzer(structure, symprec=0.1)
    space_group = sga.get_space_group_symbol()

    space_groups.append(space_group)
    relaxed_cells.append(relaxed_cell)

print(f"Successfully relaxed {len(relaxed_cells)} structures")

# %% [markdown]
# ## Step 5: Analyze Space Group Distribution
#
# Analyze the symmetry of relaxed structures to understand what crystal structures
# are accessible from the amorphous precursor. For sodium, we expect to see body-centered
# cubic (bcc, space group Im-3m) as the most stable phase at ambient conditions.

# %%
space_group_counts = Counter(space_groups)
sorted_groups = sorted(space_group_counts.items(), key=lambda x: x[1], reverse=True)

# Create visualization
fig, ax = plt.subplots(figsize=(9, 6))
labels = [sg[0] for sg in sorted_groups]
counts = [sg[1] for sg in sorted_groups]

ax.bar(range(len(labels)), counts)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_xlabel("Space Group")
ax.set_ylabel("Count")
ax.set_title("Space Group Distribution of Relaxed Structures")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

# Save plot (skip display in CI)
if not IS_CI:
    output_file = script_dir / "space_group_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {output_file}")
    plt.show()
else:
    plt.close()  # Clean up in CI

# %% [markdown]
# ## Results Summary
#
# Print detailed statistics about the space group distribution.
# This helps identify the most frequently occurring crystal structures.

# %%
print("\nSpace Group Statistics:")
for sg, count in sorted_groups:
    percentage = count / len(space_groups) * 100
    print(f"{sg}: {count} structures ({percentage:>5.1f}%)")
print(f"Total: {len(space_groups)} unique structures analyzed")
