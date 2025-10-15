"""Utilities for the a2c workflow."""

import itertools
from typing import Callable, Optional, Sequence

import numpy as np
from ase.atoms import Atoms
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure

from a2c_ase.potentials.soft_sphere import SoftSphere


def get_diameter(composition: Composition) -> float:
    """Calculate characteristic atomic diameter for a chemical composition.

    Parameters
    ----------
    composition : Composition
        Pymatgen Composition object specifying the chemical formula.

    Returns
    -------
    float
        Estimated minimum atomic diameter in Angstroms.

    Notes
    -----
    For multi-element compositions, calculates minimum possible interatomic distance by:
    - Using ionic radii for each element
    - Finding minimum sum of radii across all element pairs

    For single elements:
    - Uses metallic radius for metals
    - Uses atomic radius for non-metals, falling back to ionic radius if needed
    - Returns twice the radius as diameter
    """
    elements = composition.elements

    # Check if we have multiple elements
    if len(elements) > 1:
        # For multi-element systems, find minimum separation using ionic radii
        element_radii = [element.average_ionic_radius for element in elements]
        radius_pairs = list(itertools.combinations(element_radii, 2))
        pair_sums = np.array([sum(pair) for pair in radius_pairs])
        diameter = float(np.min(pair_sums))
    else:
        # For single-element systems
        element = elements[0]
        if element.is_metal:
            # Metals use metallic radius
            radius = float(element.metallic_radius)
        else:
            # Non-metals prefer atomic radius with fallback to ionic
            radius = float(
                element.atomic_radius if element.atomic_radius else element.average_ionic_radius
            )
        diameter = 2 * radius

    return diameter


def min_distance(
    structure: Structure,
    distance_tolerance: float = 0.0001,
) -> float:
    """Calculate minimum interatomic distance in a periodic structure.

    Computes the smallest non-zero distance between any pair of atoms, accounting for
    periodic boundary conditions. Self-interactions are excluded via a distance tolerance.

    Parameters
    ----------
    structure : Structure
        Pymatgen Structure object containing atomic positions and cell information
    distance_tolerance : float, default=0.0001
        Distances below this value (in Å) are considered self-interactions and ignored

    Returns
    -------
    float
        Minimum distance between any two different atoms in Å
    """
    # Get the distance matrix from pymatgen
    distance_matrix = structure.distance_matrix

    # Create a mask for distances below tolerance to exclude self-interactions
    # These very small distances occur when an atom is compared with itself
    mask = distance_matrix < distance_tolerance

    # Replace masked distances with a large value so they won't be selected as the minimum
    masked_distances = np.where(mask, np.inf, distance_matrix)

    # Return the smallest non-masked distance
    return float(np.min(masked_distances))


def random_packed_structure(
    composition: Composition,
    cell: np.ndarray,
    *,
    seed: int = 42,
    diameter: Optional[float] = None,
    auto_diameter: bool = False,
    max_iter: int = 100,
    fmax: float = 0.01,
    distance_tolerance: float = 0.0001,
    trajectory_file: Optional[str] = None,
    verbose: bool = True,
) -> tuple[Atoms, Optional[list[dict]]]:
    """Generate a random packed atomic structure with minimal atomic overlaps.

    Parameters
    ----------
    composition : Composition
        Pymatgen Composition object specifying atomic composition (e.g. Fe80B20).
        Numbers indicate actual atom counts.
    cell : np.ndarray
        3x3 array defining triclinic simulation box in Angstroms.
    seed : int, optional
        Random seed for reproducible generation, by default 42.
        If None, uses random initialization.
    diameter : float, optional
        Minimum allowed interatomic distance for overlap detection.
        Used for soft-sphere potential, by default None.
    auto_diameter : bool, optional
        If True, automatically calculate diameter from atomic radii, by default False.
    max_iter : int, optional
        Maximum FIRE optimization steps to minimize overlaps, by default 100.
    fmax : float, optional
        Maximum force criterion for convergence in eV/Å, by default 0.01.
    distance_tolerance : float, optional
        Distance threshold for considering atoms at same position, by default 0.0001.
    trajectory_file : str, optional
        Path to save optimization trajectory (.traj format), by default None.
    verbose : bool, optional
        Print progress information during optimization, by default True.

    Returns
    -------
    tuple[Atoms, Optional[list[dict]]]
        - ASE Atoms object with optimized positions
        - Optimization log if verbose=True, otherwise None
    """
    # Extract number of atoms for each element from composition
    element_counts = [int(i) for i in composition.as_dict().values()]
    elements = list(composition.as_dict().keys())

    # Set up reproducible random number generator
    if seed is not None:
        np.random.seed(seed)

    # Generate initial random positions in fractional coordinates
    N_atoms = sum(element_counts)
    positions = np.random.random((N_atoms, 3))

    # Calculate appropriate diameter if auto_diameter is enabled
    if auto_diameter:
        diameter = get_diameter(composition)
        if verbose:
            print(f"Using random pack diameter of {diameter}")

    # Create ASE Atoms object
    # Assign atomic numbers based on composition
    atomic_numbers = []
    for element, count in zip(elements, element_counts, strict=False):
        # Get atomic number from pymatgen Element
        element_obj = Composition(element).elements[0]
        atomic_numbers.extend([element_obj.Z] * count)

    # Convert fractional to cartesian coordinates
    positions_cart = np.dot(positions, cell)

    # Create ASE Atoms object
    atoms = Atoms(numbers=atomic_numbers, positions=positions_cart, cell=cell, pbc=True)

    # Initialize log list if verbose
    log_data = [] if verbose else None

    # Perform overlap minimization if diameter is specified
    if diameter is not None:
        if verbose:
            print("Reduce atom overlap using the soft_sphere potential")

        # Initialize soft sphere potential calculator
        calculator = SoftSphere(sigma=diameter, epsilon=1.0, alpha=2, skin=0.2)
        atoms.calc = calculator

        # Set up trajectory file if requested
        if trajectory_file is not None:
            traj = Trajectory(trajectory_file, "w", atoms)
            # Create optimizer with trajectory
            optimizer = FIRE(atoms, trajectory=traj, logfile=None)
        else:
            optimizer = FIRE(atoms, logfile=None)

        if verbose:
            initial_energy = atoms.get_potential_energy()
            print(f"Initial energy: {initial_energy:.4f}")

            # Set up logger for detailed output if verbose
            def log_progress():
                """Log optimization progress."""
                assert log_data is not None
                e = atoms.get_potential_energy()
                fmax = np.sqrt((atoms.get_forces() ** 2).sum(axis=1).max())

                # Get minimum distance
                min_dist = min_distance(
                    Structure(
                        lattice=cell,
                        species=atomic_numbers,
                        coords=atoms.positions,
                        coords_are_cartesian=True,
                    ),
                    distance_tolerance=distance_tolerance,
                )

                log_entry = {
                    "step": optimizer.nsteps,
                    "energy": float(e),
                    "fmax": float(fmax),
                    "min_dist": float(min_dist),
                }
                log_data.append(log_entry)
                opt_step = optimizer.nsteps
                print(f"Step: {opt_step}, E: {e:.4f}, Fmax: {fmax:.4f}, Min dist: {min_dist:.4f}")

            # Attach the function without parameters - it will access optimizer directly
            optimizer.attach(log_progress, 1)

        # Run FIRE optimization until convergence or max iterations
        for _ in range(max_iter):
            # Check if minimum distance criterion is met (95% of target diameter)
            min_dist = min_distance(
                Structure(
                    lattice=cell,
                    species=atomic_numbers,
                    coords=atoms.positions,
                    coords_are_cartesian=True,
                ),
                distance_tolerance=distance_tolerance,
            )

            if min_dist > diameter * 0.95:
                if verbose:
                    print(f"Minimum distance criterion met: {min_dist:.4f} > {diameter * 0.95:.4f}")
                break

            optimizer.run(fmax=fmax, steps=1)

        if verbose:
            final_energy = atoms.get_potential_energy()
            print(f"Final energy: {final_energy:.4f}")

    # Return the atoms object and log data
    return atoms, log_data


def valid_subcell(
    atoms: Atoms,
    initial_energy: float,
    final_energy: float,
    *,
    e_tol: float = 0.001,
    fe_lower_limit: float = -5.0,
    fe_upper_limit: float = 0.0,
    fusion_distance: float = 1.5,
    distance_tolerance: float = 0.0001,
) -> bool:
    """Validate a relaxed subcell structure.

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object with atomic positions, species, and cell information.
    initial_energy : float
        Total energy before relaxation in eV.
    final_energy : float
        Total energy after relaxation in eV.
    e_tol : float, default=0.001
        Energy tolerance (eV) for comparing initial and final energies.
    fe_lower_limit : float, default=-5.0
        Lower limit for formation energy (eV/atom). More negative values are unphysical.
    fe_upper_limit : float, default=0.0
        Upper limit for formation energy (eV/atom). Higher values indicate poor convergence.
    fusion_distance : float, default=1.5
        Minimum allowed interatomic distance (Å). Shorter distances indicate atomic fusion.
    distance_tolerance : float, default=0.0001
        Distance tolerance (Å) for considering atoms at same position.

    Returns
    -------
    bool
        True if structure passes all validation checks:
        - Formation energy is physically reasonable
        - Energy decreased during relaxation
        - Final energy indicates good convergence
        - No atomic fusion detected
    """
    # Check if formation energy is unphysically negative
    if final_energy < fe_lower_limit:
        return False

    # Check if optimization properly reduced the energy
    # A small tolerance accounts for numerical noise
    if not (final_energy <= initial_energy - e_tol):
        return False

    # Check if final energy is low enough to indicate good convergence
    if not (final_energy <= fe_upper_limit + e_tol):
        return False

    # Check minimum interatomic distances to detect atomic fusion
    # Convert ASE Atoms to pymatgen Structure for min_distance calculation
    structure = Structure(
        lattice=atoms.cell,
        species=atoms.get_atomic_numbers(),
        coords=atoms.get_positions(),
        coords_are_cartesian=True,
    )

    min_dist = min_distance(structure, distance_tolerance)
    if min_dist < fusion_distance:
        print("Bad structure! Fusion found.")
        return False

    # Structure passed all validation checks
    return True


def subcells_to_structures(
    candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    fractional_positions: np.ndarray,
    cell: np.ndarray,
    species: list[str],
) -> list[tuple[np.ndarray, np.ndarray, list[str]]]:
    """Convert subcell candidates to structure tuples.

    Parameters
    ----------
    candidates : list[tuple[np.ndarray, np.ndarray, np.ndarray]]
        List of (atom_indices, lower_bound, upper_bound) tuples defining subcell regions
    fractional_positions : np.ndarray
        Fractional coordinates of all atoms in parent structure, shape [n_atoms, 3]
    cell : np.ndarray
        3x3 array representing parent unit cell
    species : list[str]
        Chemical symbols for all atoms in parent structure

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray, list[str]]]
        List of (frac_coords, cell, species) tuples for each subcell:
        - frac_coords: Normalized [0,1] fractional coordinates
        - cell: Scaled 3x3 subcell
        - species: Chemical symbols for subcell atoms
    """
    list_subcells = []
    for ids, l, h in candidates:  # noqa: E741
        # Get positions of atoms in this subcell
        pos = fractional_positions[ids]

        # Shift positions to start from origin
        new_frac_pos = pos - l

        # Scale positions to [0,1] range
        new_frac_pos = new_frac_pos / (h - l).reshape(1, 3)

        # Calculate new cell parameters - scale cell by the subcell dimensions
        new_cell = cell * (h - l).reshape(1, 3)

        # Get species for the atoms in this subcell
        subcell_species = [species[int(i)] for i in ids]

        list_subcells.append((new_frac_pos, new_cell, subcell_species))

    return list_subcells


def get_target_temperature(
    step: int, equi_steps: int, cool_steps: int, T_high: float, T_low: float
) -> float:
    """Calculate target temperature for a melt-quench-equilibrate simulation step.

    Parameters
    ----------
    step : int
        Current simulation step number (0-indexed)
    equi_steps : int
        Number of steps for initial high-temperature equilibration
    cool_steps : int
        Number of steps for linear cooling
    T_high : float
        Initial high temperature in Kelvin
    T_low : float
        Final low temperature in Kelvin

    Returns
    -------
    float
        Target temperature in Kelvin for the current step

    Notes
    -----
    The temperature profile consists of three phases:
    1. Initial equilibration at T_high for equi_steps
    2. Linear cooling from T_high to T_low over cool_steps
    3. Final equilibration at T_low for remaining steps

    Examples
    --------
    >>> get_target_temperature(10, 100, 200, 2000.0, 300.0)  # During equilibration
    2000.0
    >>> get_target_temperature(200, 100, 200, 2000.0, 300.0)  # During cooling
    1150.0
    >>> get_target_temperature(350, 100, 200, 2000.0, 300.0)  # After cooling
    300.0
    """
    # Initial high-temperature equilibration phase
    if step < equi_steps:
        return T_high

    # Linear cooling phase
    if step < equi_steps + cool_steps:
        # More efficient calculation with fewer operations
        cooling_progress = (step - equi_steps) / cool_steps
        return T_high - (T_high - T_low) * cooling_progress

    # Final low-temperature equilibration phase
    return T_low


def get_subcells_to_crystallize(
    fractional_positions: np.ndarray,
    species: list[str],
    d_frac: float = 0.05,
    n_min: int = 1,
    n_max: int = 48,
    restrict_to_compositions: Optional[Sequence[str]] = None,
    max_coeff: Optional[int] = None,
    elements: Optional[Sequence[str]] = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Extract subcell structures from a larger structure for crystallization.

    Parameters
    ----------
    fractional_positions : np.ndarray
        Fractional coordinates of atoms, shape [n_atoms, 3].
    species : list[str]
        Chemical element symbols for each atom.
    d_frac : float, default=0.05
        Grid spacing in fractional coordinates. Smaller values create more overlap.
    n_min : int, default=1
        Minimum atoms per subcell.
    n_max : int, default=48
        Maximum atoms per subcell.
    restrict_to_compositions : Sequence[str], optional
        Chemical formulas to filter subcells by (e.g. ["AB", "AB2"]). Only matching compositions are
        returned.
    max_coeff : int, optional
        Maximum stoichiometric coefficient for auto-generating restrictions. E.g. max_coeff=2
        allows AB2 but not AB3.
    elements : Sequence[str], optional
        Elements for generating stoichiometries. Required if max_coeff provided.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray, np.ndarray]]
        List of (indices, lower_bounds, upper_bounds) where:
        - indices: Atom indices in subcell
        - lower_bounds: Lower bounds in fractional coords [3]
        - upper_bounds: Upper bounds in fractional coords [3]

    Notes
    -----
    Divides a structure into overlapping subcells that can be relaxed to find stable crystal
    structures. Uses a grid in fractional coordinates to identify atom groups meeting size and
    composition criteria.
    """
    # Convert species list to numpy array for easier composition handling
    species_array = np.array(species)

    # Generate allowed stoichiometries if max_coeff is specified
    if max_coeff:
        if elements is None:
            raise ValueError("elements must be provided when max_coeff is specified")
        # Generate all possible stoichiometry combinations up to max_coeff
        stoichs = list(itertools.product(range(max_coeff + 1), repeat=len(elements)))
        stoichs.pop(0)  # Remove the empty composition (0,0,...)
        # Convert stoichiometries to composition formulas
        comps = []
        for stoich in stoichs:
            comp = dict(zip(elements, stoich, strict=False))
            comps.append(Composition.from_dict(comp).reduced_formula)
        restrict_to_compositions = list(set(comps))

    # Ensure compositions are in reduced formula form if provided
    if restrict_to_compositions:
        restrict_to_compositions = [
            Composition(comp).reduced_formula for comp in restrict_to_compositions
        ]

    # Create orthorhombic grid for systematic subcell generation
    bins = int(1 / d_frac)
    grid = np.linspace(0, 1, bins + 1)

    # Generate lower and upper bounds for all possible subcells
    # Use np.meshgrid with indexing='ij' for cartesian indexing
    xx, yy, zz = np.meshgrid(grid[:-1], grid[:-1], grid[:-1], indexing="ij")
    l_bound = np.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T

    xx, yy, zz = np.meshgrid(grid[1:], grid[1:], grid[1:], indexing="ij")
    u_bound = np.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T

    candidates = []
    # Iterate through all possible subcell boundary combinations
    for lb, ub in itertools.product(l_bound, u_bound):
        if np.all(ub > lb):  # Ensure valid subcell dimensions
            # Find atoms within the subcell bounds
            in_upper = np.all(fractional_positions <= ub, axis=1)
            in_lower = np.all(fractional_positions >= lb, axis=1)
            mask = np.logical_and(in_upper, in_lower)
            ids = np.where(mask)[0]

            # Check if number of atoms meets size constraints
            if n_min <= len(ids) <= n_max:
                # Apply composition restrictions if specified
                if restrict_to_compositions:
                    subcell_comp = Composition("".join(species_array[ids])).reduced_formula
                    if subcell_comp not in restrict_to_compositions:
                        continue
                candidates.append((ids, lb, ub))

    return candidates


def default_subcell_filter(
    subcell: tuple[np.ndarray, np.ndarray, np.ndarray],
    cubic_only: bool = True,
    allowed_atom_counts: Optional[list[int]] = None,
) -> bool:
    """Filter subcells based on shape and size criteria.

    Parameters
    ----------
    subcell : tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing (atom_indices, lower_bound, upper_bound) arrays that define the subcell
    cubic_only : bool, default=True
        If True, only accept subcells with equal dimensions in x, y, and z
    allowed_atom_counts : list[int], optional
        List of allowed atom counts. If provided, only accept subcells with specified atom counts

    Returns
    -------
    bool
        True if subcell meets all criteria, False otherwise
    """
    indices, lower_bound, upper_bound = subcell
    cell_dimensions = upper_bound - lower_bound

    # Skip if cubic_only is True and cell is not cubic
    if cubic_only and not np.allclose(cell_dimensions, cell_dimensions[0], rtol=1e-3):
        return False

    # Skip if atom count is not in allowed list
    if allowed_atom_counts and len(indices) not in allowed_atom_counts:
        return False

    return True


def extract_crystallizable_subcells(
    atoms: Atoms,
    *,
    d_frac: float = 0.2,
    n_min: int = 2,
    n_max: int = 8,
    cubic_only: bool = True,
    allowed_atom_counts: Optional[list[int]] = None,
    filter_function: Optional[Callable] = None,
    restrict_to_compositions: Optional[Sequence[str]] = None,
    max_coeff: Optional[int] = None,
    elements: Optional[Sequence[str]] = None,
) -> list[Atoms]:
    """Extract and filter subcells from an amorphous structure for crystallization analysis.

    Divides an amorphous structure into overlapping subcells and filters them based on shape
    criteria and atom count to identify potential crystalline structural motifs.

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object representing the amorphous structure
    d_frac : float, optional
        Grid spacing in fractional coordinates for subcell division, by default 0.2
    n_min : int, optional
        Minimum number of atoms required in a subcell, by default 2
    n_max : int, optional
        Maximum number of atoms allowed in a subcell, by default 8
    cubic_only : bool, optional
        If True, only keep subcells where all dimensions are equal, by default True
    allowed_atom_counts : list[int], optional
        If provided, only keep subcells with these atom counts, by default None
    filter_function : callable, optional
        Custom filter function that takes a subcell tuple (indices, lower_bound, upper_bound)
        and returns a boolean. If provided, used instead of default filter, by default None
    restrict_to_compositions : Sequence[str], optional
        Chemical formulas to filter subcells by (e.g. ["AB", "AB2"]). Only matching compositions are
        returned.
    max_coeff : int, optional
        Maximum stoichiometric coefficient for auto-generating restrictions. E.g. max_coeff=2
        allows AB2 but not AB3.
    elements : Sequence[str], optional
        Elements for generating stoichiometries. Required if max_coeff provided.
    Returns
    -------
    list[Atoms]
        List of ASE Atoms objects representing the filtered subcells
    """
    # Get fractional positions from the Atoms object
    cell = atoms.get_cell()
    scaled_positions = atoms.get_scaled_positions()
    chemical_symbols = atoms.get_chemical_symbols()

    # Extract all subcells
    subcells = get_subcells_to_crystallize(
        fractional_positions=scaled_positions,
        species=chemical_symbols,
        d_frac=d_frac,
        n_min=n_min,
        n_max=n_max,
        restrict_to_compositions=restrict_to_compositions,
        max_coeff=max_coeff,
        elements=elements,
    )
    print(f"Created {len(subcells)} subcells from amorphous structure")

    # Use default filter if no filter function is provided
    if filter_function is None:

        def filter_function(subcell):
            return default_subcell_filter(
                subcell, cubic_only=cubic_only, allowed_atom_counts=allowed_atom_counts
            )

    # Filter subcells using the filter function
    filtered_subcells = [subcell for subcell in subcells if filter_function(subcell)]

    print(f"Subcells kept after filtering: {len(filtered_subcells)}")

    # Convert subcells to ASE Atoms objects
    candidate_structures = subcells_to_structures(
        candidates=filtered_subcells,
        fractional_positions=scaled_positions,
        cell=cell,
        species=chemical_symbols,
    )

    # Convert to Atoms objects
    atoms_list = []
    for frac_pos, subcell, species in candidate_structures:
        atoms_obj = Atoms(symbols=species, scaled_positions=frac_pos, cell=subcell, pbc=True)
        atoms_list.append(atoms_obj)

    return atoms_list
