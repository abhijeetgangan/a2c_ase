"""Streamlined tests for potential calculators."""

import numpy as np
from ase import Atoms

from a2c_ase.potentials.mlj import MultiLennardJones
from a2c_ase.potentials.soft_sphere import SoftSphere


def test_soft_sphere_energy_values():
    """Test soft sphere energy at various distances with reference values.

    Reference values computed following JAX MD's soft_sphere implementation:
    u(r) = epsilon/alpha * (1 - r/sigma)^alpha for r < sigma
    """
    sigma = 1.0
    epsilon = 1.0
    alpha = 2.0

    # Test cases: (distance, expected_energy)
    # At r=0.5*sigma: u = 1.0/2.0 * (1 - 0.5)^2 = 0.125
    # At r=0.8*sigma: u = 1.0/2.0 * (1 - 0.8)^2 = 0.02
    # At r=sigma: u = 0.0
    # At r>sigma: u = 0.0
    test_cases = [
        (0.5, 0.125),
        (0.8, 0.02),
        (1.0, 0.0),
        (1.5, 0.0),
    ]

    for distance, expected in test_cases:
        atoms = Atoms("Ar2", positions=[[0, 0, 0], [distance, 0, 0]])
        calc = SoftSphere(sigma=sigma, epsilon=epsilon, alpha=alpha)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        assert np.isclose(energy, expected, atol=1e-6), (
            f"Energy at r={distance} should be {expected}, got {energy}"
        )


def test_soft_sphere_forces_gradient():
    """Test that soft sphere forces match negative energy gradient."""
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [0.5, 0, 0]])
    calc = SoftSphere(sigma=1.0, epsilon=1.0, alpha=2)
    atoms.calc = calc

    analytical_forces = atoms.get_forces()

    # Numerical gradient via finite differences
    delta = 1e-6
    numerical_forces = np.zeros_like(analytical_forces)

    for i in range(len(atoms)):
        for j in range(3):
            atoms_plus = atoms.copy()
            atoms_plus.positions[i, j] += delta
            atoms_plus.calc = SoftSphere(sigma=1.0, epsilon=1.0, alpha=2)
            e_plus = atoms_plus.get_potential_energy()

            atoms_minus = atoms.copy()
            atoms_minus.positions[i, j] -= delta
            atoms_minus.calc = SoftSphere(sigma=1.0, epsilon=1.0, alpha=2)
            e_minus = atoms_minus.get_potential_energy()

            numerical_forces[i, j] = -(e_plus - e_minus) / (2 * delta)

    assert np.allclose(analytical_forces, numerical_forces, rtol=1e-5, atol=1e-6)


def test_soft_sphere_cutoff():
    """Test that soft sphere interactions are zero beyond sigma."""
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [1.5, 0, 0]])
    calc = SoftSphere(sigma=1.0, epsilon=1.0, alpha=2)
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert np.isclose(energy, 0.0, atol=1e-10)
    assert np.allclose(forces, 0.0, atol=1e-10)


def test_soft_sphere_energy_scaling():
    """Test that energy scales linearly with epsilon."""
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [0.5, 0, 0]])

    calc1 = SoftSphere(sigma=1.0, epsilon=1.0, alpha=2)
    atoms.calc = calc1
    energy1 = atoms.get_potential_energy()

    calc2 = SoftSphere(sigma=1.0, epsilon=2.0, alpha=2)
    atoms.calc = calc2
    energy2 = atoms.get_potential_energy()

    assert np.isclose(energy2 / energy1, 2.0, rtol=1e-5)


def test_lennard_jones_energy_values():
    """Test LJ energy at various distances with reference values.

    Reference values from JAX MD's Lennard-Jones implementation:
    u(r) = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6]

    At r = 2^(1/6)*sigma ≈ 1.122*sigma: energy minimum = -epsilon
    At r = sigma: u = 0
    At r = 1.5*sigma: u ≈ -0.0163*epsilon (with shifted potential)
    """
    sigma = 1.0
    epsilon = 1.0

    # Test at minimum: r = 2^(1/6) * sigma
    r_min = 2.0 ** (1.0 / 6.0) * sigma
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [r_min, 0, 0]])
    calc = MultiLennardJones(sigma=sigma, epsilon=epsilon, rc=3.0, smooth=False)
    atoms.calc = calc
    energy_min = atoms.get_potential_energy()

    # At minimum, energy should be -epsilon (with shift correction)
    # Shifted energy at rc=3.0: e0 = 4*eps*[(1/3)^12 - (1/3)^6] ≈ -0.0009823*epsilon
    e0_at_rc = 4 * epsilon * ((sigma / 3.0) ** 12 - (sigma / 3.0) ** 6)
    expected_min = -epsilon - e0_at_rc
    assert np.isclose(energy_min, expected_min, rtol=1e-3)

    # Test at sigma: should be close to zero (with shift)
    atoms2 = Atoms("Ar2", positions=[[0, 0, 0], [sigma, 0, 0]])
    atoms2.calc = MultiLennardJones(sigma=sigma, epsilon=epsilon, rc=3.0, smooth=False)
    energy_at_sigma = atoms2.get_potential_energy()
    expected_at_sigma = -e0_at_rc  # Just the shift
    assert np.isclose(energy_at_sigma, expected_at_sigma, atol=1e-3)


def test_lennard_jones_forces_at_minimum():
    """Test that LJ forces are zero at energy minimum."""
    sigma = 1.0
    r_min = 2.0 ** (1.0 / 6.0) * sigma

    atoms = Atoms("Ar2", positions=[[0, 0, 0], [r_min, 0, 0]])
    calc = MultiLennardJones(sigma=sigma, epsilon=1.0, smooth=False)
    atoms.calc = calc
    forces = atoms.get_forces()

    assert np.allclose(forces, 0.0, atol=1e-3)


def test_lennard_jones_forces_gradient():
    """Test that LJ forces match negative energy gradient."""
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [1.2, 0, 0]])
    calc = MultiLennardJones(sigma=1.0, epsilon=1.0, smooth=False)
    atoms.calc = calc

    analytical_forces = atoms.get_forces()

    # Numerical gradient
    delta = 1e-6
    numerical_forces = np.zeros_like(analytical_forces)

    for i in range(len(atoms)):
        for j in range(3):
            atoms_plus = atoms.copy()
            atoms_plus.positions[i, j] += delta
            atoms_plus.calc = MultiLennardJones(sigma=1.0, epsilon=1.0, smooth=False)
            e_plus = atoms_plus.get_potential_energy()

            atoms_minus = atoms.copy()
            atoms_minus.positions[i, j] -= delta
            atoms_minus.calc = MultiLennardJones(sigma=1.0, epsilon=1.0, smooth=False)
            e_minus = atoms_minus.get_potential_energy()

            numerical_forces[i, j] = -(e_plus - e_minus) / (2 * delta)

    assert np.allclose(analytical_forces, numerical_forces, rtol=1e-4, atol=1e-6)


def test_lennard_jones_smooth_cutoff():
    """Test smooth cutoff gives continuous forces."""
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [1.5, 0, 0]])
    calc = MultiLennardJones(sigma=1.0, epsilon=1.0, rc=2.5, smooth=True)
    atoms.calc = calc

    analytical_forces = atoms.get_forces()

    # Verify gradient consistency with smooth cutoff
    delta = 1e-6
    numerical_forces = np.zeros_like(analytical_forces)

    for i in range(len(atoms)):
        for j in range(3):
            atoms_plus = atoms.copy()
            atoms_plus.positions[i, j] += delta
            atoms_plus.calc = MultiLennardJones(sigma=1.0, epsilon=1.0, rc=2.5, smooth=True)
            e_plus = atoms_plus.get_potential_energy()

            atoms_minus = atoms.copy()
            atoms_minus.positions[i, j] -= delta
            atoms_minus.calc = MultiLennardJones(sigma=1.0, epsilon=1.0, rc=2.5, smooth=True)
            e_minus = atoms_minus.get_potential_energy()

            numerical_forces[i, j] = -(e_plus - e_minus) / (2 * delta)

    assert np.allclose(analytical_forces, numerical_forces, rtol=1e-4, atol=1e-6)


def test_lennard_jones_cutoff_beyond_rc():
    """Test that LJ interactions are zero beyond cutoff."""
    rc = 2.0
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [2.5, 0, 0]])
    calc = MultiLennardJones(sigma=1.0, epsilon=1.0, rc=rc)
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert np.isclose(energy, 0.0, atol=1e-10)
    assert np.allclose(forces, 0.0, atol=1e-10)


def test_lennard_jones_mixing_rules():
    """Test Lorentz-Berthelot mixing rule for multi-species."""
    atoms = Atoms("ArNe", positions=[[0, 0, 0], [1.2, 0, 0]])

    sigma = {"Ar": 1.0, "Ne": 0.8}
    epsilon = {"Ar": 1.0, "Ne": 0.64}

    calc = MultiLennardJones(sigma=sigma, epsilon=epsilon, mixing_rule="lorentz_berthelot")
    atoms.calc = calc
    calc._setup_species_parameters(atoms)

    # Expected: sigma_AB = (1.0 + 0.8)/2 = 0.9
    # Expected: epsilon_AB = sqrt(1.0 * 0.64) = 0.8
    sigma_ab = calc._species_parameters["sigma_matrix"][("Ar", "Ne")]  # type: ignore
    epsilon_ab = calc._species_parameters["epsilon_matrix"][("Ar", "Ne")]  # type: ignore

    assert np.isclose(sigma_ab, 0.9)
    assert np.isclose(epsilon_ab, 0.8)


def test_newton_third_law():
    """Test that forces obey Newton's third law for both potentials."""
    test_cases = [
        (SoftSphere, {"sigma": 1.0, "epsilon": 1.0, "alpha": 2}),
        (MultiLennardJones, {"sigma": 1.0, "epsilon": 1.0}),
    ]

    for calc_class, params in test_cases:
        atoms = Atoms("Ar2", positions=[[0, 0, 0], [1.2, 0, 0]])
        calc = calc_class(**params)
        atoms.calc = calc
        forces = atoms.get_forces()

        # Forces should be equal and opposite
        assert np.allclose(forces[0], -forces[1], rtol=1e-10)


def test_translational_invariance():
    """Test that potentials are translationally invariant."""
    positions1 = np.array([[0, 0, 0], [1.2, 0, 0]])
    positions2 = positions1 + np.array([10.0, 20.0, 30.0])

    test_cases = [
        (SoftSphere, {"sigma": 1.0, "epsilon": 1.0, "alpha": 2}),
        (MultiLennardJones, {"sigma": 1.0, "epsilon": 1.0}),
    ]

    for calc_class, params in test_cases:
        atoms1 = Atoms("Ar2", positions=positions1)
        atoms1.calc = calc_class(**params)
        energy1 = atoms1.get_potential_energy()

        atoms2 = Atoms("Ar2", positions=positions2)
        atoms2.calc = calc_class(**params)
        energy2 = atoms2.get_potential_energy()

        assert np.isclose(energy1, energy2, rtol=1e-10)


def test_rotational_invariance():
    """Test that potentials are rotationally invariant."""
    positions1 = np.array([[0, 0, 0], [1.2, 0, 0], [0, 1.2, 0]])

    # Rotate 90 degrees around z-axis
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    positions2 = positions1 @ rotation.T

    test_cases = [
        (SoftSphere, {"sigma": 1.0, "epsilon": 1.0, "alpha": 2}),
        (MultiLennardJones, {"sigma": 1.0, "epsilon": 1.0}),
    ]

    for calc_class, params in test_cases:
        atoms1 = Atoms("Ar3", positions=positions1)
        atoms1.calc = calc_class(**params)
        energy1 = atoms1.get_potential_energy()

        atoms2 = Atoms("Ar3", positions=positions2)
        atoms2.calc = calc_class(**params)
        energy2 = atoms2.get_potential_energy()

        assert np.isclose(energy1, energy2, rtol=1e-10)


def test_per_atom_energy_sum():
    """Test that per-atom energies sum to total energy."""
    test_cases = [
        (SoftSphere, {"sigma": 1.0, "epsilon": 1.0, "alpha": 2}),
        (MultiLennardJones, {"sigma": 1.0, "epsilon": 1.0}),
    ]

    for calc_class, params in test_cases:
        atoms = Atoms("Ar3", positions=[[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0]])
        calc = calc_class(**params)
        atoms.calc = calc

        total_energy = atoms.get_potential_energy()
        per_atom_energies = calc.results["energies"]

        assert np.isclose(total_energy, per_atom_energies.sum(), rtol=1e-10)
