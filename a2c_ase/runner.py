"""Runner module for the a2c workflow."""

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.filters import FrechetCellFilter
from ase.io import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import FIRE

from a2c_ase.utils import get_target_temperature


def melt_quench_md(
    atoms,
    calculator: Calculator,
    *,
    equi_steps: int = 2500,
    cool_steps: int = 2500,
    final_steps: int = 2500,
    T_high: float = 2000.0,
    T_low: float = 300.0,
    time_step: float = 2.0,
    friction: float = 0.01,
    trajectory_file: str | None = None,
    seed: int = 42,
    verbose: bool = True,
    log_interval: int = 100,
):
    """
    Run melt-quench molecular dynamics to generate amorphous structures.

    Performs a three-stage MD simulation:
    1. High-temperature equilibration to melt the structure
    2. Controlled cooling to quench the liquid
    3. Low-temperature equilibration to relax the structure

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object with initial atomic structure
    calculator : Calculator
        ASE calculator for computing forces and energies
    equi_steps : int, default=2500
        Number of steps for high-temperature equilibration
    cool_steps : int, default=2500
        Number of steps for cooling phase
    final_steps : int, default=2500
        Number of steps for final low-temperature equilibration
    T_high : float, default=2000.0
        High temperature for melting phase (K)
    T_low : float, default=300.0
        Low temperature for final phase (K)
    time_step : float, default=2.0
        MD time step (fs)
    friction : float, default=0.01
        Langevin friction parameter (atomic units)
    trajectory_file : str, optional
        Path to save MD trajectory
    seed : int, default=42
        Random seed for reproducibility
    verbose : bool, default=True
        Whether to print progress information
    log_interval : int, default=100
        Number of steps between progress logs

    Returns
    -------
    atoms : Atoms
        ASE Atoms with final amorphous structure
    log_data : dict
        Dictionary with temperature and energy trajectories
    """
    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Make a copy of the atoms object to avoid modifying the original
    atoms = atoms.copy()

    # Set the calculator
    atoms.calc = calculator

    # Initialize velocities according to Maxwell-Boltzmann distribution at high temperature
    MaxwellBoltzmannDistribution(atoms, temperature_K=T_high, force_temp=True)

    # Set up total simulation steps
    total_steps = equi_steps + cool_steps + final_steps

    # Initialize logger to track temperature and energy
    log_data = {
        "temperature": np.zeros(total_steps),
        "potential_energy": np.zeros(total_steps),
        "kinetic_energy": np.zeros(total_steps),
        "total_energy": np.zeros(total_steps),
    }

    # Set up trajectory file if requested
    if trajectory_file is not None:
        traj = Trajectory(trajectory_file, "w", atoms)
        traj_callback = traj.write
    else:
        traj_callback = None

    # Set up Langevin dynamics with initial temperature
    # Convert time_step from fs to ASE time units, friction to ASE units
    dyn = Langevin(
        atoms, timestep=time_step * units.fs, temperature_K=T_high, friction=friction, logfile=None
    )

    # Define the callback function to update temperature and log data
    def update_temp_and_log(step):
        # Get current target temperature
        temp = get_target_temperature(step, equi_steps, cool_steps, T_high, T_low)

        # Update thermostat temperature
        dyn.set_temperature(temperature_K=temp)

        # Log current state
        current_temp = atoms.get_temperature()
        pot_energy = atoms.get_potential_energy()
        kin_energy = atoms.get_kinetic_energy()

        log_data["temperature"][step] = current_temp
        log_data["potential_energy"][step] = pot_energy
        log_data["kinetic_energy"][step] = kin_energy
        log_data["total_energy"][step] = pot_energy + kin_energy

        # Print progress if verbose
        if verbose and step % log_interval == 0:
            print(
                f"Step {step}/{total_steps}: T = {current_temp:.1f} K, "
                f"E_pot = {pot_energy:.3f} eV, E_kin = {kin_energy:.3f} eV"
            )

        # Write trajectory if callback exists
        if traj_callback is not None:
            traj_callback()

    # Run the simulation
    for step in range(total_steps):
        update_temp_and_log(step)
        dyn.run(1)

    if verbose:
        final_temp = atoms.get_temperature()
        print("\nMelt-quench simulation completed:")
        print(f"Final temperature: {final_temp:.1f} K")
        print(f"Final energy: {atoms.get_potential_energy():.3f} eV")

    # Close trajectory file if it was opened
    if trajectory_file is not None:
        traj.close()

    return atoms, log_data


def relax_unit_cell(
    atoms: Atoms,
    calculator: Calculator,
    *,
    max_iter: int = 200,
    fmax: float = 0.01,
    trajectory_file: str | None = None,
    verbose: bool = True,
) -> tuple[Atoms, dict]:
    """Relax atomic positions and cell parameters using FIRE optimization.

    Performs simultaneous optimization of atomic positions and unit cell parameters using
    the Fast Inertial Relaxation Engine (FIRE) algorithm. The cell optimization is handled
    through ASE's FrechetCellFilter.

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object containing atomic positions, species and cell information
    calculator : Calculator
        ASE calculator for computing energies, forces and stresses
    max_iter : int, optional
        Maximum FIRE iterations, by default 200
    fmax : float, optional
        Force convergence criterion in eV/Å, by default 0.01
    trajectory_file : str, optional
        Path to save optimization trajectory, by default None
    verbose : bool, optional
        Print optimization progress, by default True

    Returns
    -------
    tuple[Atoms, dict]
        - Relaxed Atoms object with optimized positions and cell
        - Dictionary containing energy, forces, stress and volume trajectories

    Notes
    -----
    The calculator must support stress tensor calculations.
    Periodic boundary conditions must be enabled on the Atoms object.
    """
    # Ensure atoms has a calculator
    atoms.calc = calculator

    # Make sure all periodic boundary conditions are enabled
    atoms.pbc = True

    # Set up trajectory file if requested
    if trajectory_file is not None:
        traj = Trajectory(trajectory_file, "w", atoms)
    else:
        traj = None

    # Create logger dictionary
    logger = {"energy": [], "forces": [], "stress": [], "volume": [], "pressure": []}

    # Set up cell filter for combined atom and cell relaxation
    cell_filter = FrechetCellFilter(atoms)

    # Create optimizer
    optimizer = FIRE(cell_filter, logfile=None, trajectory=traj)

    if verbose:
        # Get initial values
        initial_energy = atoms.get_potential_energy()
        initial_volume = atoms.get_volume()
        initial_stress = atoms.get_stress(voigt=False)  # Full 3x3 stress tensor
        initial_pressure = -np.trace(initial_stress) / 3.0

        print(f"Initial energy: {initial_energy:.6f} eV")
        print(f"Initial volume: {initial_volume:.3f} Å³")
        print(f"Initial pressure: {initial_pressure:.6f} eV/Å³")

        # Add attach function to log data during optimization
        def log_progress():
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            stress = atoms.get_stress(voigt=False)
            volume = atoms.get_volume()
            pressure = -np.trace(stress) / 3.0

            # Store in logger
            logger["energy"].append(float(energy))
            logger["forces"].append(forces.copy())
            logger["stress"].append(stress.copy())
            logger["volume"].append(float(volume))
            logger["pressure"].append(float(pressure))

            # Print progress
            fmax_val = np.sqrt((forces**2).sum(axis=1).max())
            print(
                f"Step {optimizer.nsteps}: E = {energy:.6f} eV, "
                f"Fmax = {fmax_val:.6f} eV/Å, "
                f"P = {pressure:.6f} eV/Å³, "
                f"V = {volume:.3f} Å³"
            )

        optimizer.attach(log_progress, interval=1)

    # Run optimization
    optimizer.run(fmax=fmax, steps=max_iter)

    if verbose:
        # Get final values for reporting
        final_energy = atoms.get_potential_energy()
        final_volume = atoms.get_volume()
        final_stress = atoms.get_stress(voigt=False)
        final_pressure = -np.trace(final_stress) / 3.0

        print("\nOptimization completed:")
        print(f"Final energy: {final_energy:.6f} eV")
        print(f"Final volume: {final_volume:.3f} Å³")
        print(f"Final pressure: {final_pressure:.6f} eV/Å³")
        print(f"Steps taken: {optimizer.nsteps}")

    # Convert logger arrays for consistency
    if logger["energy"]:
        logger["energy"] = np.array(logger["energy"])
        logger["forces"] = np.array(logger["forces"])
        logger["stress"] = np.array(logger["stress"])
        logger["volume"] = np.array(logger["volume"])
        logger["pressure"] = np.array(logger["pressure"])

    return atoms, logger
