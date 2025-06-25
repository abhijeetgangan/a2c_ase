import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import NeighborList
from ase.stress import full_3x3_to_voigt_6_stress


class MultiLennardJones(Calculator):
    """Multi-species Lennard Jones potential calculator

    see https://en.wikipedia.org/wiki/Lennard-Jones_potential
    """

    implemented_properties = ["energy", "energies", "forces", "free_energy"]
    implemented_properties += ["stress", "stresses"]
    default_parameters = {
        "epsilon": 1.0,
        "sigma": 1.0,
        "rc": None,
        "ro": None,
        "smooth": False,
        "mixing_rule": "lorentz_berthelot",
        "cross_interactions": None,
    }
    nolabel = True

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        sigma: float, dict, or array-like
          If float: uniform sigma for all species, default 1.0
          If dict: mapping from atomic symbols to sigma values
          If array-like: sigma values indexed by atomic numbers
        epsilon: float, dict, or array-like
          If float: uniform epsilon for all species, default 1.0
          If dict: mapping from atomic symbols to epsilon values
          If array-like: epsilon values indexed by atomic numbers
        rc: float, None
          Cut-off for the NeighborList. If None, set to 3 * max(sigma).
          Default None
        ro: float, None
          Onset of cutoff function in 'smooth' mode. Defaults to 0.66 * rc.
        smooth: bool, False
          Cutoff mode. False means that the pairwise energy is simply shifted
          to be 0 at r = rc, leading to the energy going to 0 continuously,
          but the forces jumping to zero discontinuously at the cutoff.
          True means that a smooth cutoff function is multiplied to the pairwise
          energy that smoothly goes to 0 between ro and rc.
        mixing_rule: str, 'lorentz_berthelot'
          Mixing rule for cross-species interactions:
          - 'lorentz_berthelot': sigma_ij = (sigma_i + sigma_j)/2,
                                epsilon_ij = sqrt(epsilon_i * epsilon_j)
          - 'geometric': sigma_ij = sqrt(sigma_i * sigma_j),
                        epsilon_ij = sqrt(epsilon_i * epsilon_j)
        cross_interactions: dict, None
          Dictionary specifying explicit cross-interaction parameters.
          Format: {('A', 'B'): {'sigma': value, 'epsilon': value}}
          If provided, these override mixing rules for specified pairs.
          Example for Kob-Andersen:
          cross_interactions={('A', 'B'): {'sigma': 0.8, 'epsilon': 1.5}}

        """

        Calculator.__init__(self, **kwargs)
        self.nl = None
        self._species_parameters = None

    def _setup_species_parameters(self, atoms):
        """Setup species-specific parameters and mixing rules."""
        sigma = self.parameters.sigma
        epsilon = self.parameters.epsilon
        mixing_rule = self.parameters.mixing_rule
        cross_interactions = self.parameters.cross_interactions

        # Get unique symbols
        symbols = atoms.get_chemical_symbols()
        unique_symbols = list(set(symbols))

        # Convert parameters to per-species format
        if isinstance(sigma, dict):
            sigma_dict = sigma
        elif hasattr(sigma, "__len__") and not isinstance(sigma, str):
            # Array-like, assume indexed by atomic number
            sigma_dict = {atoms[i].symbol: sigma[atoms[i].number] for i in range(len(atoms))}
            sigma_dict = {s: sigma_dict[s] for s in unique_symbols}
        else:
            # Single value for all species
            sigma_dict = {s: sigma for s in unique_symbols}

        if isinstance(epsilon, dict):
            epsilon_dict = epsilon
        elif hasattr(epsilon, "__len__") and not isinstance(epsilon, str):
            # Array-like, assume indexed by atomic number
            epsilon_dict = {atoms[i].symbol: epsilon[atoms[i].number] for i in range(len(atoms))}
            epsilon_dict = {s: epsilon_dict[s] for s in unique_symbols}
        else:
            # Single value for all species
            epsilon_dict = {s: epsilon for s in unique_symbols}

        # Create mixing rule lookup tables
        sigma_matrix = {}
        epsilon_matrix = {}

        for sym1 in unique_symbols:
            for sym2 in unique_symbols:
                # Check for explicit cross-interactions first
                pair_key = tuple(sorted([sym1, sym2]))
                cross_found = False

                if cross_interactions is not None:
                    # Check both orderings
                    for key in [(sym1, sym2), (sym2, sym1), pair_key]:
                        if key in cross_interactions:
                            sigma_ij = cross_interactions[key]["sigma"]
                            epsilon_ij = cross_interactions[key]["epsilon"]
                            cross_found = True
                            break

                if not cross_found:
                    # Use mixing rules
                    if mixing_rule == "lorentz_berthelot":
                        sigma_ij = (sigma_dict[sym1] + sigma_dict[sym2]) / 2.0
                        epsilon_ij = np.sqrt(epsilon_dict[sym1] * epsilon_dict[sym2])
                    elif mixing_rule == "geometric":
                        sigma_ij = np.sqrt(sigma_dict[sym1] * sigma_dict[sym2])
                        epsilon_ij = np.sqrt(epsilon_dict[sym1] * epsilon_dict[sym2])
                    else:
                        raise ValueError(f"Unknown mixing rule: {mixing_rule}")

                sigma_matrix[(sym1, sym2)] = sigma_ij
                epsilon_matrix[(sym1, sym2)] = epsilon_ij

        # Set cutoff if not specified
        if self.parameters.rc is None:
            max_sigma = max(sigma_dict.values())
            self.parameters.rc = 3 * max_sigma

        if self.parameters.ro is None:
            self.parameters.ro = 0.66 * self.parameters.rc

        self._species_parameters = {
            "sigma_matrix": sigma_matrix,
            "epsilon_matrix": epsilon_matrix,
            "sigma_dict": sigma_dict,
            "epsilon_dict": epsilon_dict,
            "unique_symbols": unique_symbols,
        }

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)

        natoms = len(self.atoms)

        # Setup species parameters if needed
        if (
            self._species_parameters is None
            or "numbers" in system_changes
            or "initial_charges" in system_changes
        ):
            self._setup_species_parameters(self.atoms)

        rc = self.parameters.rc
        ro = self.parameters.ro
        smooth = self.parameters.smooth

        sigma_matrix = self._species_parameters["sigma_matrix"]
        epsilon_matrix = self._species_parameters["epsilon_matrix"]

        if self.nl is None or "numbers" in system_changes:
            self.nl = NeighborList([rc / 2] * natoms, self_interaction=False, bothways=True)

        self.nl.update(self.atoms)

        positions = self.atoms.positions
        cell = self.atoms.cell
        symbols = self.atoms.get_chemical_symbols()

        energies = np.zeros(natoms)
        forces = np.zeros((natoms, 3))
        stresses = np.zeros((natoms, 3, 3))

        for ii in range(natoms):
            neighbors, offsets = self.nl.get_neighbors(ii)
            if len(neighbors) == 0:
                continue

            cells = np.dot(offsets, cell)
            sym_i = symbols[ii]

            # pointing *towards* neighbours
            distance_vectors = positions[neighbors] + cells - positions[ii]
            r2 = (distance_vectors**2).sum(1)

            # Get species-specific parameters for each neighbor pair
            neighbor_symbols = [symbols[j] for j in neighbors]
            sigma_ij = np.array([sigma_matrix[(sym_i, sym_j)] for sym_j in neighbor_symbols])
            epsilon_ij = np.array([epsilon_matrix[(sym_i, sym_j)] for sym_j in neighbor_symbols])

            # Calculate LJ terms
            c6 = (sigma_ij**2 / r2) ** 3
            c6[r2 > rc**2] = 0.0
            c12 = c6**2

            if smooth:
                cutoff_fn = cutoff_function(r2, rc**2, ro**2)
                d_cutoff_fn = d_cutoff_function(r2, rc**2, ro**2)

            pairwise_energies = 4 * epsilon_ij * (c12 - c6)
            pairwise_forces = -24 * epsilon_ij * (2 * c12 - c6) / r2  # du_ij

            if smooth:
                # order matters, otherwise the pairwise energy is already modified
                pairwise_forces = cutoff_fn * pairwise_forces + 2 * d_cutoff_fn * pairwise_energies
                pairwise_energies *= cutoff_fn
            else:
                # potential value at rc for each pair
                e0 = 4 * epsilon_ij * ((sigma_ij / rc) ** 12 - (sigma_ij / rc) ** 6)
                pairwise_energies -= e0 * (c6 != 0.0)

            pairwise_forces = pairwise_forces[:, np.newaxis] * distance_vectors

            energies[ii] += 0.5 * pairwise_energies.sum()  # atomic energies
            forces[ii] += pairwise_forces.sum(axis=0)

            stresses[ii] += 0.5 * np.dot(
                pairwise_forces.T, distance_vectors
            )  # equivalent to outer product

        # no lattice, no stress
        if self.atoms.cell.rank == 3:
            stresses = full_3x3_to_voigt_6_stress(stresses)
            self.results["stress"] = stresses.sum(axis=0) / self.atoms.get_volume()
            self.results["stresses"] = stresses / self.atoms.get_volume()

        energy = energies.sum()
        self.results["energy"] = energy
        self.results["energies"] = energies

        self.results["free_energy"] = energy

        self.results["forces"] = forces


def cutoff_function(r, rc, ro):
    """Smooth cutoff function.

    Goes from 1 to 0 between ro and rc, ensuring
    that u(r) = lj(r) * cutoff_function(r) is C^1.

    Defined as 1 below ro, 0 above rc.

    Note that r, rc, ro are all expected to be squared,
    i.e. `r = r_ij^2`, etc.

    Taken from https://github.com/google/jax-md.

    """

    return np.where(
        r < ro,
        1.0,
        np.where(r < rc, (rc - r) ** 2 * (rc + 2 * r - 3 * ro) / (rc - ro) ** 3, 0.0),
    )


def d_cutoff_function(r, rc, ro):
    """Derivative of smooth cutoff function wrt r.

    Note that `r = r_ij^2`, so for the derivative wrt to `r_ij`,
    we need to multiply `2*r_ij`. This gives rise to the factor 2
    above, the `r_ij` is cancelled out by the remaining derivative
    `d r_ij / d d_ij`, i.e. going from scalar distance to distance vector.
    """

    return np.where(
        r < ro,
        0.0,
        np.where(r < rc, 6 * (rc - r) * (ro - r) / (rc - ro) ** 3, 0.0),
    )
