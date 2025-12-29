"""
RAO (Response Amplitude Operator) computation using Capytaine.
"""

import numpy as np
import xarray as xr
import capytaine as cpt
from pathlib import Path

from hull import create_fpso_hull, get_vessel_mass_properties, DEFAULT_FPSO


def compute_raos(body: cpt.FloatingBody,
                 periods: np.ndarray,
                 wave_directions: np.ndarray = None,
                 water_depth: float = np.inf,
                 mass_props: dict = None) -> xr.Dataset:
    """
    Compute RAOs for a floating body using Capytaine.

    Parameters
    ----------
    body : cpt.FloatingBody
        The vessel mesh with DOFs defined
    periods : np.ndarray
        Wave periods to compute (s)
    wave_directions : np.ndarray, optional
        Wave directions in degrees (0=following, 90=beam, 180=head)
        Default: [0, 45, 90, 135, 180]
    water_depth : float
        Water depth (m), default infinite
    mass_props : dict, optional
        Mass properties from get_vessel_mass_properties()

    Returns
    -------
    xr.Dataset
        Dataset containing RAOs for all DOFs, periods, and directions
    """
    if wave_directions is None:
        wave_directions = np.array([0, 45, 90, 135, 180])

    # Convert directions to radians for Capytaine
    wave_dirs_rad = np.deg2rad(wave_directions)

    # Convert periods to angular frequencies
    omegas = 2 * np.pi / periods

    # Set up the BEM solver
    solver = cpt.BEMSolver()

    # Create the problems for radiation (body moving in still water)
    # and diffraction (waves hitting fixed body)
    problems = []

    for omega in omegas:
        for direction in wave_dirs_rad:
            problems.append(
                cpt.DiffractionProblem(
                    body=body,
                    wave_direction=direction,
                    omega=omega,
                    water_depth=water_depth
                )
            )
        for dof in body.dofs:
            problems.append(
                cpt.RadiationProblem(
                    body=body,
                    radiating_dof=dof,
                    omega=omega,
                    water_depth=water_depth
                )
            )

    # Solve all problems
    results = solver.solve_all(problems, keep_details=False)

    # Combine results into a dataset
    dataset = cpt.assemble_dataset(results)

    # Compute RAOs if mass properties provided
    if mass_props is not None:
        # Build mass and stiffness matrices
        dataset = add_rigid_body_dynamics(dataset, body, mass_props)

    return dataset


def add_rigid_body_dynamics(dataset: xr.Dataset,
                            body: cpt.FloatingBody,
                            mass_props: dict) -> xr.Dataset:
    """
    Add mass matrix and hydrostatic stiffness to compute RAOs.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset from Capytaine with added mass and radiation damping
    body : cpt.FloatingBody
        The floating body
    mass_props : dict
        Mass properties dictionary

    Returns
    -------
    xr.Dataset
        Dataset with RAO values added
    """
    mass = mass_props['mass']
    Ixx = mass_props['Ixx']
    Iyy = mass_props['Iyy']
    Izz = mass_props['Izz']

    # Mass matrix (6x6)
    mass_matrix = np.diag([mass, mass, mass, Ixx, Iyy, Izz])

    # Hydrostatic stiffness from Capytaine
    stiffness = body.compute_hydrostatics()['hydrostatic_stiffness']

    # Store in dataset
    dataset['mass_matrix'] = xr.DataArray(
        mass_matrix,
        dims=['influenced_dof', 'radiating_dof'],
        coords={
            'influenced_dof': list(body.dofs.keys()),
            'radiating_dof': list(body.dofs.keys())
        }
    )

    dataset['hydrostatic_stiffness'] = xr.DataArray(
        stiffness,
        dims=['influenced_dof', 'radiating_dof'],
        coords={
            'influenced_dof': list(body.dofs.keys()),
            'radiating_dof': list(body.dofs.keys())
        }
    )

    return dataset


def solve_motion_raos(dataset: xr.Dataset) -> xr.Dataset:
    """
    Solve the equation of motion to get RAOs.

    [-ω²(M + A) + iωB + K] * ξ = F

    Where:
    - M = mass matrix
    - A = added mass
    - B = radiation damping
    - K = hydrostatic stiffness
    - F = wave excitation force
    - ξ = motion response (RAO)

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with all hydrodynamic data

    Returns
    -------
    xr.Dataset
        Dataset with RAO values added
    """
    omegas = dataset['omega'].values
    directions = dataset['wave_direction'].values
    dofs = list(dataset['radiating_dof'].values)
    ndof = len(dofs)

    # Get matrices
    M = dataset['mass_matrix'].values
    K = dataset['hydrostatic_stiffness'].values

    # Initialize RAO array
    raos = np.zeros((len(omegas), len(directions), ndof), dtype=complex)

    for i, omega in enumerate(omegas):
        # Added mass and damping at this frequency
        A = dataset['added_mass'].sel(omega=omega).values
        B = dataset['radiation_damping'].sel(omega=omega).values

        # System matrix: [-ω²(M + A) + iωB + K]
        system_matrix = -omega**2 * (M + A) + 1j * omega * B + K

        for j, direction in enumerate(directions):
            # Excitation force at this frequency and direction
            F = dataset['Froude_Krylov_force'].sel(
                omega=omega, wave_direction=direction
            ).values + dataset['diffraction_force'].sel(
                omega=omega, wave_direction=direction
            ).values

            # Solve for motion
            try:
                xi = np.linalg.solve(system_matrix, F)
                raos[i, j, :] = xi
            except np.linalg.LinAlgError:
                raos[i, j, :] = np.nan

    # Store RAOs in dataset
    dataset['RAO'] = xr.DataArray(
        raos,
        dims=['omega', 'wave_direction', 'influenced_dof'],
        coords={
            'omega': omegas,
            'wave_direction': directions,
            'influenced_dof': dofs
        }
    )

    # Also store amplitude and phase
    dataset['RAO_amplitude'] = np.abs(dataset['RAO'])
    dataset['RAO_phase'] = xr.DataArray(
        np.angle(raos, deg=True),
        dims=['omega', 'wave_direction', 'influenced_dof'],
        coords={
            'omega': omegas,
            'wave_direction': directions,
            'influenced_dof': dofs
        }
    )

    # Add period coordinate for convenience
    dataset = dataset.assign_coords(period=('omega', 2 * np.pi / omegas))

    return dataset


def compute_fpso_raos(length: float = None, beam: float = None, draft: float = None,
                      periods: np.ndarray = None,
                      wave_directions: np.ndarray = None,
                      cache_dir: str = None) -> xr.Dataset:
    """
    Convenience function to compute RAOs for an FPSO.

    Parameters
    ----------
    length, beam, draft : float, optional
        Vessel dimensions (default: 300m class FPSO)
    periods : np.ndarray, optional
        Wave periods (default: 4-25s range)
    wave_directions : np.ndarray, optional
        Wave directions in degrees
    cache_dir : str, optional
        Directory to cache results

    Returns
    -------
    xr.Dataset
        Complete RAO dataset
    """
    # Use defaults if not provided
    if length is None:
        length = DEFAULT_FPSO['length']
    if beam is None:
        beam = DEFAULT_FPSO['beam']
    if draft is None:
        draft = DEFAULT_FPSO['draft']
    if periods is None:
        periods = np.linspace(4, 25, 22)  # 4s to 25s
    if wave_directions is None:
        wave_directions = np.array([0, 45, 90, 135, 180])

    # Check cache
    if cache_dir:
        cache_path = Path(cache_dir) / f"rao_L{length}_B{beam}_D{draft}.nc"
        if cache_path.exists():
            return xr.open_dataset(cache_path)

    # Create hull and compute properties
    body = create_fpso_hull(length, beam, draft)
    mass_props = get_vessel_mass_properties(length, beam, draft)

    # Compute hydrodynamic coefficients
    dataset = compute_raos(body, periods, wave_directions, mass_props=mass_props)

    # Solve for motion RAOs
    dataset = solve_motion_raos(dataset)

    # Add vessel info as attributes
    dataset.attrs['vessel_length'] = length
    dataset.attrs['vessel_beam'] = beam
    dataset.attrs['vessel_draft'] = draft
    dataset.attrs['vessel_mass'] = mass_props['mass']
    dataset.attrs['vessel_displacement'] = mass_props['displacement']

    # Cache if requested
    if cache_dir:
        Path(cache_dir).mkdir(exist_ok=True)
        dataset.to_netcdf(cache_path)

    return dataset


if __name__ == "__main__":
    # Test computation with a small example
    print("Computing RAOs for default FPSO...")
    print(f"Dimensions: {DEFAULT_FPSO}")

    # Use coarse mesh for quick test
    body = create_fpso_hull(
        DEFAULT_FPSO['length'],
        DEFAULT_FPSO['beam'],
        DEFAULT_FPSO['draft'],
        nx=10, ny=4, nz=3  # Coarse mesh for testing
    )

    print(f"Mesh has {body.mesh.nb_faces} panels")

    # Compute for a few periods
    test_periods = np.array([6, 10, 15, 20])
    test_directions = np.array([0, 90, 180])  # Following, beam, head

    mass_props = get_vessel_mass_properties(
        DEFAULT_FPSO['length'],
        DEFAULT_FPSO['beam'],
        DEFAULT_FPSO['draft']
    )
    print(f"Mass: {mass_props['mass']/1e6:.1f} kton")

    dataset = compute_raos(body, test_periods, test_directions, mass_props=mass_props)
    dataset = solve_motion_raos(dataset)

    print("\nRAO Amplitudes (Heave) [m/m]:")
    print(dataset['RAO_amplitude'].sel(influenced_dof='Heave').values)

    print("\nRAO Amplitudes (Roll) [deg/m]:")
    print(np.rad2deg(dataset['RAO_amplitude'].sel(influenced_dof='Roll').values))

    print("\nDone!")
