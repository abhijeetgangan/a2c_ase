"""Soft Sphere Potential Calculator Implementation.

This module provides a calculator for soft sphere potentials in atomic simulations.
The soft sphere potential is a simple repulsive interaction model commonly used
in molecular dynamics simulations of granular materials, colloids, and other
particulate systems.

The potential is defined pairwise between atoms and is characterized by three
key parameters:
    - sigma: Particle diameter (distance at which potential becomes zero)
    - epsilon: Energy scale of the interaction (strength of the potential)
    - alpha: Stiffness exponent (controls how quickly the potential increases)

When particles overlap (r < sigma), they experience a repulsive force that
increases as they move closer together. The potential is zero when particles
are at or beyond the cutoff distance (r ≥ sigma).
"""

from typing import Any, ClassVar, Optional

import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import NeighborList
from ase.stress import full_3x3_to_voigt_6_stress


class SoftSphere(Calculator):
    """Soft sphere potential calculator for purely repulsive interactions.

    Implements a soft repulsive potential used for structure generation and packing
    optimization. The potential is zero when r >= sigma and purely repulsive for r < sigma.

    Attributes
    ----------
    implemented_properties : list of str
        Calculable properties: energy, energies, forces, free_energy, stress, stresses
    default_parameters : dict
        Default values: sigma=1.0, epsilon=1.0, alpha=2, skin=0.2

    Notes
    -----
    The pairwise energy is \\(u_{ij} = \\frac{\\epsilon}{\\alpha}\\) times
    \\(\\left(1 - \\frac{r_{ij}}{\\sigma}\\right)^{\\alpha}\\) for \\(r_{ij} < \\sigma\\).
    Energy partitioning uses symmetric approach with bothways=True neighbor list.
    Implementation based on JAX-MD (https://github.com/google/jax-md).
    """

    # Properties that this calculator can compute
    implemented_properties: ClassVar[list[str]] = [
        "energy",  # Total energy of the system
        "energies",  # Per-atom energy contributions
        "forces",  # Forces on each atom
        "free_energy",  # Free energy (same as energy for this potential)
    ]
    # Bulk properties
    implemented_properties += ["stress", "stresses"]

    # Default parameters for the potential
    default_parameters: ClassVar[dict[str, float]] = {
        "sigma": 1.0,  # Particle diameter
        "epsilon": 1.0,  # Interaction energy scale
        "alpha": 2,  # Exponent for interaction stiffness
        "skin": 0.2,  # Skin parameter for neighbor list
    }
    nolabel = True

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the SoftSphere calculator with the given parameters.

        Parameters
        ----------
        **kwargs : dict
            Calculator parameters. Supported parameters:

            - sigma (float): Particle diameter, default: 1.0
            - epsilon (float): Interaction energy scale, default: 1.0
            - alpha (float): Exponent specifying interaction stiffness, default: 2
            - skin (float): Skin parameter for neighbor list, default: 0.2
        """
        Calculator.__init__(self, **kwargs)
        self.nl = None  # Neighbor list, initialized in calculate()

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: Optional[list[str]] = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        """Calculate energy, forces and stress using soft sphere potential.

        Parameters
        ----------
        atoms : Atoms, optional
            ASE Atoms object containing atomic positions and cell
        properties : list of str, optional
            Properties to calculate. If None, calculates all implemented properties
        system_changes : list of str, optional
            List of changes since last calculation
        """
        # Use all implemented properties if none specified
        if properties is None:
            properties = self.implemented_properties

        # Call parent calculator to set up atoms
        Calculator.calculate(self, atoms, properties, system_changes)

        n_atoms = len(self.atoms)

        # Extract parameters
        sigma = self.parameters.sigma
        epsilon = self.parameters.epsilon
        alpha = self.parameters.alpha
        skin = self.parameters.skin

        # Initialize or update neighbor list if needed
        if self.nl is None or "numbers" in system_changes:
            self.nl = NeighborList(
                [np.max(sigma)] * n_atoms,
                self_interaction=False,
                bothways=True,
                skin=skin,
            )

        self.nl.update(self.atoms)

        # Get atomic positions and cell
        positions = self.atoms.positions
        cell = self.atoms.cell

        # Initialize arrays for results
        energies = np.zeros(n_atoms)
        forces = np.zeros((n_atoms, 3))
        stresses = np.zeros((n_atoms, 3, 3))

        # Loop over all atoms
        for ii in range(n_atoms):
            # Get neighbors and their offsets
            neighbors, offsets = self.nl.get_neighbors(ii)
            cells = np.dot(offsets, cell)

            # Calculate distance vectors (pointing *towards* neighbors)
            distance_vectors = positions[neighbors] + cells - positions[ii]

            # Calculate distances
            r2 = (distance_vectors**2).sum(1)
            dr = np.sqrt(r2)
            dr_by_sigma = dr / sigma

            # Define energy and force calculation functions
            def energy_pair(dr_by_sigma: float) -> float:
                """Calculate pairwise energy for a given normalized distance."""
                return epsilon / alpha * (1.0 - dr_by_sigma) ** alpha

            def force_pair(dr_by_sigma: float, dr_val: float) -> float:
                """Calculate magnitude of pairwise force for a given normalized distance."""
                return ((-epsilon / sigma) * (1.0 - dr_by_sigma) ** (alpha - 1)) / dr_val

            # Apply cutoff at sigma (dr_by_sigma = 1.0)
            pairwise_energies = np.where(dr_by_sigma <= 1.0, energy_pair(dr_by_sigma), 0.0)
            pairwise_forces = np.where(dr_by_sigma <= 1.0, force_pair(dr_by_sigma, dr), 0.0)

            # Convert scalar forces to vectors
            pairwise_forces = pairwise_forces[:, np.newaxis] * distance_vectors

            # Accumulate energies and forces
            energies[ii] += 0.5 * pairwise_energies.sum()  # atomic energies
            forces[ii] += pairwise_forces.sum(axis=0)

            # Calculate stress contributions (equivalent to outer product)
            stresses[ii] += 0.5 * np.dot(pairwise_forces.T, distance_vectors)

        # Calculate stress only if we have a 3D periodic cell
        if self.atoms.cell.rank == 3:
            # Convert to Voigt notation and normalize by volume
            stresses = full_3x3_to_voigt_6_stress(stresses)
            volume = self.atoms.get_volume()
            self.results["stress"] = stresses.sum(axis=0) / volume
            self.results["stresses"] = stresses / volume

        # Store results
        energy = energies.sum()
        self.results["energy"] = energy
        self.results["energies"] = energies
        self.results["free_energy"] = energy
        self.results["forces"] = forces
