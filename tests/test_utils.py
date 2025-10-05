"""Basic tests for a2c_ase.utils module."""

import numpy as np
from ase import Atoms
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure

from a2c_ase.utils import (
    default_subcell_filter,
    get_diameter,
    get_target_temperature,
    min_distance,
    valid_subcell,
)


def test_get_diameter():
    """Test the get_diameter function."""
    comp = Composition("Si")
    diameter = get_diameter(comp)
    assert diameter > 0, "Diameter should be positive"

    comp = Composition("Fe2O3")
    diameter = get_diameter(comp)
    assert diameter > 0, "Diameter should be positive"


def test_get_target_temperature():
    """Test the get_target_temperature function."""
    # During high-temp phase
    temp = get_target_temperature(50, 100, 200, 2000.0, 300.0)
    assert temp == 2000.0

    # During cooling phase (halfway)
    temp = get_target_temperature(200, 100, 200, 2000.0, 300.0)
    assert np.isclose(temp, 1150.0)

    # After cooling phase
    temp = get_target_temperature(350, 100, 200, 2000.0, 300.0)
    assert temp == 300.0


def test_min_distance():
    """Test min_distance calculation."""
    # Create a simple structure with two atoms
    lattice = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]
    species = ["Si", "Si"]
    coords = [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]  # 1.0 Angstrom apart
    structure = Structure(lattice, species, coords)

    min_dist = min_distance(structure)
    assert np.isclose(min_dist, 1.0, atol=0.01)


def test_valid_subcell():
    """Test subcell validation logic."""
    # Create dummy atoms with a cell
    atoms = Atoms("Si2", positions=[[0, 0, 0], [2.0, 0, 0]], cell=[5, 5, 5], pbc=True)

    # Valid case: energy decreased (0.0 → -1.0), within bounds, no fusion
    is_valid = valid_subcell(atoms, initial_energy=0.0, final_energy=-1.0, fusion_distance=1.5)
    assert is_valid

    # Invalid: energy increased during relaxation (-1.0 → 0.0 is uphill)
    is_valid = valid_subcell(atoms, initial_energy=-1.0, final_energy=0.0, fusion_distance=1.5)
    assert not is_valid

    # Invalid: final energy is unphysically low (< -5.0 eV/atom threshold)
    is_valid = valid_subcell(atoms, initial_energy=0.0, final_energy=-10.0, fusion_distance=1.5)
    assert not is_valid


def test_default_subcell_filter():
    """Test default subcell filtering."""
    # Create a cubic subcell
    indices = np.array([0, 1, 2])
    lower_bound = np.array([0.0, 0.0, 0.0])
    upper_bound = np.array([1.0, 1.0, 1.0])
    cubic_subcell = (indices, lower_bound, upper_bound)

    # Should pass cubic filter
    assert default_subcell_filter(cubic_subcell, cubic_only=True)

    # Non-cubic subcell
    non_cubic_upper = np.array([1.0, 2.0, 1.0])
    non_cubic_subcell = (indices, lower_bound, non_cubic_upper)

    # Should fail cubic filter
    assert not default_subcell_filter(non_cubic_subcell, cubic_only=True)

    # Test atom count filter
    assert default_subcell_filter(cubic_subcell, cubic_only=False, allowed_atom_counts=[3])
    assert not default_subcell_filter(cubic_subcell, cubic_only=False, allowed_atom_counts=[4, 5])
