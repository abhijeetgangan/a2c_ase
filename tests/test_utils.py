"""Tests for a2c_ase.utils module."""

import tempfile

import numpy as np
import pytest
from ase import Atoms
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure

from a2c_ase.utils import (
    default_subcell_filter,
    extract_crystallizable_subcells,
    get_diameter,
    get_subcells_to_crystallize,
    get_target_temperature,
    min_distance,
    random_packed_structure,
    subcells_to_structures,
    valid_subcell,
)


@pytest.fixture
def simple_atoms():
    """Create simple test atoms."""
    return Atoms("Si2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[5, 5, 5], pbc=True)


@pytest.fixture
def simple_cell():
    """Create simple cubic cell."""
    return np.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]])


@pytest.mark.parametrize(
    "composition,expected_range",
    [
        ("Si", (2.0, 3.0)),
        ("Fe", (2.0, 3.0)),
        ("Fe2O3", (2.0, 4.0)),
    ],
)
def test_get_diameter(composition, expected_range):
    """Test diameter calculation for various compositions."""
    diameter = get_diameter(Composition(composition))
    assert expected_range[0] < diameter < expected_range[1]


@pytest.mark.parametrize(
    "step,equi_steps,cool_steps,T_high,T_low,expected",
    [
        (50, 100, 200, 2000.0, 300.0, 2000.0),  # High-temp phase
        (200, 100, 200, 2000.0, 300.0, 1150.0),  # Cooling phase
        (350, 100, 200, 2000.0, 300.0, 300.0),  # Low-temp phase
    ],
)
def test_get_target_temperature(step, equi_steps, cool_steps, T_high, T_low, expected):
    """Test temperature schedule calculation."""
    temp = get_target_temperature(step, equi_steps, cool_steps, T_high, T_low)
    assert np.isclose(temp, expected)


def test_min_distance():
    """Test minimum distance calculation."""
    structure = Structure(
        [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
    )
    assert np.isclose(min_distance(structure), 1.0, atol=0.01)


@pytest.mark.parametrize(
    "initial_e,final_e,fe_lower,fe_upper,fusion_dist,expected",
    [
        (0.0, -1.0, -5.0, 0.0, 1.5, True),  # Valid
        (-1.0, 0.0, -5.0, 0.0, 1.5, False),  # Energy increased
        (0.0, -10.0, -5.0, 0.0, 1.5, False),  # Unphysically low
        (2.0, 1.0, -5.0, 0.5, 1.5, False),  # Above upper limit
    ],
)
def test_valid_subcell(simple_atoms, initial_e, final_e, fe_lower, fe_upper, fusion_dist, expected):
    """Test subcell validation with various energy scenarios."""
    is_valid = valid_subcell(
        simple_atoms,
        initial_energy=initial_e,
        final_energy=final_e,
        fe_lower_limit=fe_lower,
        fe_upper_limit=fe_upper,
        fusion_distance=fusion_dist,
    )
    assert is_valid == expected


def test_valid_subcell_fusion():
    """Test atomic fusion detection."""
    atoms = Atoms("Si2", positions=[[0, 0, 0], [0.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    assert not valid_subcell(atoms, 0.0, -1.0, fusion_distance=1.5)


def test_default_subcell_filter():
    """Test subcell filtering logic."""
    indices = np.array([0, 1, 2])
    cubic = (indices, np.zeros(3), np.ones(3))
    non_cubic = (indices, np.zeros(3), np.array([1.0, 2.0, 1.0]))

    assert default_subcell_filter(cubic, cubic_only=True)
    assert not default_subcell_filter(non_cubic, cubic_only=True)
    assert default_subcell_filter(cubic, cubic_only=False, allowed_atom_counts=[3])
    assert not default_subcell_filter(cubic, cubic_only=False, allowed_atom_counts=[4, 5])


@pytest.mark.parametrize(
    "comp,cell,kwargs,n_atoms,has_log",
    [
        ("Si4", [[10, 0, 0], [0, 10, 0], [0, 0, 10]], {"verbose": False}, 4, False),
        (
            "Si2",
            [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
            {"diameter": 2.0, "max_iter": 5, "verbose": False},
            2,
            False,
        ),
        (
            "Si2",
            [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
            {"auto_diameter": True, "max_iter": 5, "verbose": True},
            2,
            True,
        ),
    ],
)
def test_random_packed_structure(comp, cell, kwargs, n_atoms, has_log):
    """Test random packed structure generation with various parameters."""
    atoms, log = random_packed_structure(Composition(comp), np.array(cell), seed=42, **kwargs)
    assert isinstance(atoms, Atoms)
    assert len(atoms) == n_atoms
    assert (log is not None) == has_log


def test_subcells_to_structures(simple_cell):
    """Test subcell structure conversion."""
    frac_pos = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.6, 0.6, 0.6]])
    candidates = [(np.array([0, 1]), np.zeros(3), 0.5 * np.ones(3))]

    structures = subcells_to_structures(candidates, frac_pos, simple_cell, ["Si", "Si", "O"])

    assert len(structures) == 1
    frac_pos, subcell, species = structures[0]
    assert species == ["Si", "Si"]
    assert frac_pos.shape == (2, 3)


def test_get_subcells_to_crystallize():
    """Test subcell extraction."""
    frac_pos = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.6, 0.6, 0.6], [0.8, 0.8, 0.8]])
    subcells = get_subcells_to_crystallize(
        frac_pos, ["Si", "Si", "O", "O"], d_frac=0.5, n_min=1, n_max=4
    )

    assert isinstance(subcells, list)
    for ids, _, _ in subcells:
        assert 1 <= len(ids) <= 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"restrict_to_compositions": ["SiO"]},
        {"max_coeff": 2, "elements": ["Si", "O"]},
    ],
)
def test_get_subcells_with_filters(kwargs):
    """Test subcell extraction with composition filters."""
    frac_pos = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]])
    subcells = get_subcells_to_crystallize(
        frac_pos, ["Si", "Si", "O"], d_frac=0.5, n_min=1, n_max=3, **kwargs
    )
    assert isinstance(subcells, list)


def test_extract_crystallizable_subcells():
    """Test subcell extraction from atoms."""
    atoms = Atoms(
        "Si2O2", positions=[[0, 0, 0], [2, 2, 2], [5, 5, 5], [7, 7, 7]], cell=[10, 10, 10], pbc=True
    )
    subcells = extract_crystallizable_subcells(
        atoms, d_frac=0.5, n_min=1, n_max=4, cubic_only=False
    )

    assert all(isinstance(s, Atoms) for s in subcells)


def test_extract_crystallizable_subcells_with_custom_filter():
    """Test subcell extraction with custom filter."""
    atoms = Atoms(
        "Si2O2", positions=[[0, 0, 0], [2, 2, 2], [5, 5, 5], [7, 7, 7]], cell=[10, 10, 10], pbc=True
    )
    subcells = extract_crystallizable_subcells(
        atoms, d_frac=0.5, n_min=2, n_max=3, filter_function=lambda x: True
    )
    assert isinstance(subcells, list)


def test_random_packed_structure_with_trajectory():
    """Test random packing with trajectory output."""
    with tempfile.NamedTemporaryFile(suffix=".traj") as f:
        atoms, _ = random_packed_structure(
            Composition("Si2"),
            np.diag([5, 5, 5]),
            seed=42,
            diameter=2.0,
            max_iter=3,
            trajectory_file=f.name,
            verbose=False,
        )
        assert len(atoms) == 2


def test_random_packed_structure_verbose_optimization():
    """Test verbose logging during optimization."""
    atoms, log = random_packed_structure(
        Composition("Si8"),
        np.diag([6, 6, 6]),
        seed=42,
        diameter=3.0,
        max_iter=10,
        fmax=0.1,
        verbose=True,
    )
    assert len(atoms) == 8
    assert log is not None and len(log) > 0


def test_get_subcells_error_on_missing_elements():
    """Test error when max_coeff provided without elements."""
    with pytest.raises(ValueError, match="elements must be provided"):
        get_subcells_to_crystallize(
            np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]),
            ["Si", "Si"],
            d_frac=0.5,
            n_min=1,
            n_max=3,
            max_coeff=2,
            elements=None,
        )
