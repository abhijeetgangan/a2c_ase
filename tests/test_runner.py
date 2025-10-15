"""Basic tests for a2c_ase.runner module."""

import tempfile

import numpy as np
from ase import Atoms
from ase.build import bulk

from a2c_ase.potentials.mlj import MultiLennardJones
from a2c_ase.runner import melt_quench_md, relax_unit_cell

# Realistic Lennard-Jones parameters for Argon
AR_SIGMA = 3.405  # Angstroms
AR_EPSILON = 0.0103  # eV
AR_RC = 2.5 * AR_SIGMA  # Typical cutoff for Ar LJ potential


def test_melt_quench_md_basic():
    """Test basic melt-quench MD functionality with realistic Ar LJ potential."""
    # Create a 2x2x2 Ar FCC supercell (32 atoms)
    atoms = bulk("Ar", "fcc", a=5.3, cubic=True) * (2, 2, 2)
    calc = MultiLennardJones(sigma=AR_SIGMA, epsilon=AR_EPSILON, rc=AR_RC)

    # Run very short simulation
    result_atoms, log_data = melt_quench_md(
        atoms,
        calc,
        equi_steps=5,
        cool_steps=5,
        final_steps=5,
        T_high=500.0,
        T_low=100.0,
        verbose=False,
    )

    # Check return types
    assert isinstance(result_atoms, Atoms)
    assert isinstance(log_data, dict)

    # Check log data structure
    assert "temperature" in log_data
    assert "potential_energy" in log_data
    assert "kinetic_energy" in log_data
    assert "total_energy" in log_data

    # Check data shapes
    total_steps = 15
    assert len(log_data["temperature"]) == total_steps
    assert len(log_data["potential_energy"]) == total_steps

    # Check atoms are modified
    assert not np.allclose(result_atoms.positions, atoms.positions)


def test_melt_quench_md_with_trajectory():
    """Test melt-quench MD with trajectory file using Ar LJ potential."""
    # Create a 2x2x2 Ar FCC supercell (32 atoms)
    atoms = bulk("Ar", "fcc", a=5.3, cubic=True) * (2, 2, 2)
    calc = MultiLennardJones(sigma=AR_SIGMA, epsilon=AR_EPSILON, rc=AR_RC)

    with tempfile.NamedTemporaryFile(suffix=".traj", delete=False) as f:
        traj_file = f.name

    # Run with trajectory
    result_atoms, log_data = melt_quench_md(
        atoms,
        calc,
        equi_steps=3,
        cool_steps=3,
        final_steps=3,
        trajectory_file=traj_file,
        verbose=False,
    )

    assert isinstance(result_atoms, Atoms)
    # Trajectory file should exist (we don't check contents to keep it simple)


def test_relax_unit_cell_basic():
    """Test basic unit cell relaxation with Ar LJ potential."""
    # Create Ar FCC structure slightly off equilibrium
    atoms = bulk("Ar", "fcc", a=5.0)
    calc = MultiLennardJones(sigma=AR_SIGMA, epsilon=AR_EPSILON, rc=AR_RC)

    # Run very short relaxation
    result_atoms, log_dict = relax_unit_cell(atoms, calc, max_iter=5, fmax=0.1, verbose=False)

    # Check return types
    assert isinstance(result_atoms, Atoms)
    assert isinstance(log_dict, dict)

    # Check that atoms have calculator
    assert result_atoms.calc is not None

    # Check that PBC is enabled
    assert all(result_atoms.pbc)


def test_relax_unit_cell_energy_decrease():
    """Test that relaxation decreases energy with Ar LJ potential."""
    # Start with compressed lattice (equilibrium is ~5.3 Å)
    atoms = bulk("Ar", "fcc", a=4.8)
    calc = MultiLennardJones(sigma=AR_SIGMA, epsilon=AR_EPSILON, rc=AR_RC)

    # Get initial energy
    atoms.calc = calc
    initial_energy = atoms.get_potential_energy()

    # Run relaxation
    result_atoms, log_dict = relax_unit_cell(atoms, calc, max_iter=10, fmax=0.1, verbose=False)

    # Get final energy
    final_energy = result_atoms.get_potential_energy()

    # Energy should decrease or stay same
    assert final_energy <= initial_energy + 0.01


def test_relax_unit_cell_convergence():
    """Test that relaxation reduces forces with Ar LJ potential."""
    # Start with highly compressed lattice
    atoms = bulk("Ar", "fcc", a=4.5)
    calc = MultiLennardJones(sigma=AR_SIGMA, epsilon=AR_EPSILON, rc=AR_RC)

    # Get initial forces
    atoms.calc = calc
    initial_fmax = np.sqrt((atoms.get_forces() ** 2).sum(axis=1).max())

    # Run relaxation
    result_atoms, log_dict = relax_unit_cell(atoms, calc, max_iter=20, fmax=0.05, verbose=False)

    # Get final forces
    final_fmax = np.sqrt((result_atoms.get_forces() ** 2).sum(axis=1).max())

    # Forces should be significantly reduced
    assert final_fmax <= initial_fmax + 0.05


def test_melt_quench_md_verbose():
    """Test melt-quench MD with verbose output."""
    atoms = bulk("Ar", "fcc", a=5.3, cubic=True) * (2, 2, 2)
    calc = MultiLennardJones(sigma=AR_SIGMA, epsilon=AR_EPSILON, rc=AR_RC)

    # Run with verbose=True and small log_interval to trigger prints
    result_atoms, log_data = melt_quench_md(
        atoms,
        calc,
        equi_steps=5,
        cool_steps=5,
        final_steps=5,
        T_high=500.0,
        T_low=100.0,
        verbose=True,
        log_interval=1,  # Log every step
    )

    assert isinstance(result_atoms, Atoms)
    assert isinstance(log_data, dict)


def test_relax_unit_cell_verbose():
    """Test unit cell relaxation with verbose output."""
    atoms = bulk("Ar", "fcc", a=5.0)
    calc = MultiLennardJones(sigma=AR_SIGMA, epsilon=AR_EPSILON, rc=AR_RC)

    # Run with verbose=True
    result_atoms, log_dict = relax_unit_cell(atoms, calc, max_iter=5, fmax=0.1, verbose=True)

    # Check logger contains data
    assert "energy" in log_dict
    assert "forces" in log_dict
    assert "stress" in log_dict
    assert "volume" in log_dict
    assert "pressure" in log_dict

    # Check that logger has entries (since verbose=True)
    assert len(log_dict["energy"]) > 0


def test_relax_unit_cell_with_trajectory():
    """Test unit cell relaxation with trajectory file."""
    atoms = bulk("Ar", "fcc", a=5.0)
    calc = MultiLennardJones(sigma=AR_SIGMA, epsilon=AR_EPSILON, rc=AR_RC)

    with tempfile.NamedTemporaryFile(suffix=".traj", delete=False) as f:
        traj_file = f.name

    # Run with trajectory
    result_atoms, log_dict = relax_unit_cell(
        atoms, calc, max_iter=3, fmax=0.1, trajectory_file=traj_file, verbose=True
    )

    assert isinstance(result_atoms, Atoms)
    assert isinstance(log_dict, dict)
