"""
Hull geometry generation for hydrodynamic analysis with Capytaine.
"""

import numpy as np
import capytaine as cpt


def create_box_barge(length: float, beam: float, draft: float,
                     nx: int = 30, ny: int = 10, nz: int = 8,
                     name: str = "barge") -> cpt.FloatingBody:
    """
    Create a simple box barge (rectangular hull) mesh.

    Parameters
    ----------
    length : float
        Length of the barge (m)
    beam : float
        Beam/width of the barge (m)
    draft : float
        Draft of the barge (m) - how deep it sits in water
    nx, ny, nz : int
        Number of panels in x, y, z directions
    name : str
        Name of the floating body

    Returns
    -------
    cpt.FloatingBody
        Capytaine floating body with mesh and DOFs defined
    """
    # Create a rectangular parallelepiped (box) mesh
    # Only the underwater part is meshed (from z=-draft to z=0)
    mesh = cpt.meshes.predefined.mesh_parallelepiped(
        size=(length, beam, draft),
        center=(0, 0, -draft/2),
        resolution=(nx, ny, nz),
        name=f"{name}_mesh"
    )

    # Keep only the submerged part (z <= 0)
    mesh = mesh.immersed_part()

    # Create floating body
    body = cpt.FloatingBody(mesh=mesh, name=name)

    # Set center of mass (VCG typically ~40% of draft below waterline for loaded FPSO)
    body.center_of_mass = (0, 0, -draft * 0.4)

    # Set rotation center to center of mass
    body.rotation_center = body.center_of_mass

    # Add all 6 degrees of freedom
    body.add_translation_dof(name="Surge")
    body.add_translation_dof(name="Sway")
    body.add_translation_dof(name="Heave")
    # Rotations are around rotation_center
    body.add_rotation_dof(name="Roll")
    body.add_rotation_dof(name="Pitch")
    body.add_rotation_dof(name="Yaw")

    return body


def create_fpso_hull(length: float, beam: float, draft: float,
                     bow_length: float = None, stern_length: float = None,
                     nx: int = 40, ny: int = 12, nz: int = 8,
                     name: str = "fpso") -> cpt.FloatingBody:
    """
    Create a simplified FPSO hull with tapered bow and stern.

    Hull shape (plan view):

              bow (pointed)
                 /\\
                /  \\
               /    \\
              /      \\
             |        |   <- parallel mid-body
             |        |
             |        |
              \\      /
               \\____/     <- stern (rounded/tapered)

    Parameters
    ----------
    length : float
        Length overall (m)
    beam : float
        Maximum beam (m)
    draft : float
        Design draft (m)
    bow_length : float, optional
        Length of bow taper section (default: 15% of length)
    stern_length : float, optional
        Length of stern taper section (default: 10% of length)
    nx, ny, nz : int
        Mesh resolution (nx along length, ny along beam, nz along draft)
    name : str
        Name of the vessel

    Returns
    -------
    cpt.FloatingBody
        Capytaine floating body
    """
    # Default bow/stern lengths
    if bow_length is None:
        bow_length = 0.15 * length  # 15% of length for bow
    if stern_length is None:
        stern_length = 0.10 * length  # 10% of length for stern

    mid_length = length - bow_length - stern_length

    # Generate hull vertices
    # x: along length (bow at +x, stern at -x)
    # y: along beam (port at +y, starboard at -y)
    # z: vertical (waterline at 0, keel at -draft)

    n_bow = max(4, nx // 4)
    n_mid = max(4, nx // 2)
    n_stern = max(3, nx // 4)
    n_y = max(4, ny)
    n_z = max(3, nz)

    vertices = []
    faces = []

    def half_beam_at_x(x):
        """Return half-beam at longitudinal position x."""
        bow_start = length/2 - bow_length
        stern_end = -length/2 + stern_length

        if x > bow_start:
            # Bow taper (parabolic for smoother shape)
            t = (x - bow_start) / bow_length
            return (beam/2) * (1 - t**2)
        elif x < stern_end:
            # Stern taper (gentler, more rounded)
            t = (stern_end - x) / stern_length
            return (beam/2) * (1 - 0.5*t**2)
        else:
            # Parallel mid-body
            return beam/2

    # Create grid of points for each section
    x_coords = np.concatenate([
        np.linspace(-length/2, -length/2 + stern_length, n_stern),
        np.linspace(-length/2 + stern_length, length/2 - bow_length, n_mid)[1:],
        np.linspace(length/2 - bow_length, length/2, n_bow)[1:]
    ])

    z_coords = np.linspace(-draft, 0, n_z)

    # Build mesh vertices and faces
    # We'll create the hull as connected quadrilateral panels

    def add_quad(v1, v2, v3, v4):
        """Add a quadrilateral face (4 vertex indices)."""
        faces.append([v1, v2, v3, v4])

    # Store vertex indices for each (x_idx, side, z_idx)
    # side: 0=port (+y), 1=starboard (-y), 2=bottom
    vertex_map = {}

    for i, x in enumerate(x_coords):
        hb = half_beam_at_x(x)

        for k, z in enumerate(z_coords):
            # Port side
            idx_port = len(vertices)
            vertices.append([x, hb, z])
            vertex_map[(i, 'port', k)] = idx_port

            # Starboard side
            idx_stbd = len(vertices)
            vertices.append([x, -hb, z])
            vertex_map[(i, 'stbd', k)] = idx_stbd

    # Add bottom centerline vertices
    for i, x in enumerate(x_coords):
        idx_bottom = len(vertices)
        vertices.append([x, 0, -draft])
        vertex_map[(i, 'keel', 0)] = idx_bottom

    # Create faces
    for i in range(len(x_coords) - 1):
        for k in range(len(z_coords) - 1):
            # Port side panels
            add_quad(
                vertex_map[(i, 'port', k)],
                vertex_map[(i+1, 'port', k)],
                vertex_map[(i+1, 'port', k+1)],
                vertex_map[(i, 'port', k+1)]
            )
            # Starboard side panels
            add_quad(
                vertex_map[(i, 'stbd', k+1)],
                vertex_map[(i+1, 'stbd', k+1)],
                vertex_map[(i+1, 'stbd', k)],
                vertex_map[(i, 'stbd', k)]
            )

        # Bottom panels (from port to starboard at z=-draft)
        add_quad(
            vertex_map[(i, 'port', 0)],
            vertex_map[(i, 'stbd', 0)],
            vertex_map[(i+1, 'stbd', 0)],
            vertex_map[(i+1, 'port', 0)]
        )

    # Bow transom (close the front)
    i = len(x_coords) - 1
    for k in range(len(z_coords) - 1):
        add_quad(
            vertex_map[(i, 'stbd', k)],
            vertex_map[(i, 'port', k)],
            vertex_map[(i, 'port', k+1)],
            vertex_map[(i, 'stbd', k+1)]
        )

    # Stern transom (close the back)
    i = 0
    for k in range(len(z_coords) - 1):
        add_quad(
            vertex_map[(i, 'port', k)],
            vertex_map[(i, 'stbd', k)],
            vertex_map[(i, 'stbd', k+1)],
            vertex_map[(i, 'port', k+1)]
        )

    vertices = np.array(vertices)
    faces = np.array(faces)

    # Create Capytaine mesh
    mesh = cpt.Mesh(vertices=vertices, faces=faces, name=f"{name}_mesh")
    mesh = mesh.immersed_part()

    # Create floating body
    body = cpt.FloatingBody(mesh=mesh, name=name)

    # Set center of mass
    body.center_of_mass = (0, 0, -draft * 0.4)
    body.rotation_center = body.center_of_mass

    # Add DOFs
    body.add_translation_dof(name="Surge")
    body.add_translation_dof(name="Sway")
    body.add_translation_dof(name="Heave")
    body.add_rotation_dof(name="Roll")
    body.add_rotation_dof(name="Pitch")
    body.add_rotation_dof(name="Yaw")

    return body


def get_vessel_mass_properties(length: float, beam: float, draft: float,
                               block_coefficient: float = 0.85) -> dict:
    """
    Estimate mass properties for a vessel.

    Parameters
    ----------
    length : float
        Length (m)
    beam : float
        Beam (m)
    draft : float
        Draft (m)
    block_coefficient : float
        Block coefficient (typical FPSO: 0.80-0.90)

    Returns
    -------
    dict
        Dictionary with mass properties:
        - displacement: volume displacement (m³)
        - mass: mass assuming seawater density (kg)
        - Ixx, Iyy, Izz: moments of inertia (kg·m²)
        - center_of_gravity: (x, y, z) position (m)
    """
    rho_water = 1025  # kg/m³ seawater density

    # Volume and mass
    volume = length * beam * draft * block_coefficient
    mass = volume * rho_water

    # Approximate radii of gyration (typical values)
    # kxx ≈ 0.35-0.40 * B (roll)
    # kyy ≈ 0.25 * L (pitch)
    # kzz ≈ 0.25 * L (yaw)
    kxx = 0.38 * beam
    kyy = 0.25 * length
    kzz = 0.25 * length

    Ixx = mass * kxx**2
    Iyy = mass * kyy**2
    Izz = mass * kzz**2

    # Center of gravity (assume at waterline level, centered)
    # VCG typically slightly below waterline for loaded FPSO
    cog = (0, 0, -draft * 0.4)

    return {
        'displacement': volume,
        'mass': mass,
        'Ixx': Ixx,
        'Iyy': Iyy,
        'Izz': Izz,
        'center_of_gravity': cog,
        'kxx': kxx,
        'kyy': kyy,
        'kzz': kzz
    }


# Default FPSO dimensions (approximately 300m class)
DEFAULT_FPSO = {
    'length': 300.0,  # m
    'beam': 54.0,     # m
    'draft': 22.0,    # m
    'name': 'FPSO_300m'
}
