#-GEOMETRY GENERATION PIPELINE FOR TPMS STRUCTURES
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

# ============================================================
# USER PARAMETERS
# ============================================================

# Unit-cell dimensions
Lx = 1.0
Ly = 1.0
Lz = 1.0

# Number of sampling points in each direction
Nx = 60
Ny = 60
Nz = 60

# Level set to extract
LEVEL = 0.0

# Visualization
SHOW_SURFACE = True
SHOW_NORMALS = False

# Normal visualization density
NORMAL_SKIP = 100

# Normal vector visualization scale
NORMAL_SCALE = 0.05

# Export options
EXPORT_STL = True
EXPORT_NODES = True
EXPORT_ELEMENTS = True
EXPORT_NORMALS = True

# Output file names
STL_FILENAME = "primitive_tpms.stl"
NODE_FILENAME = "primitive_nodes.csv"
ELEMENT_FILENAME = "primitive_elements.csv"
NORMAL_FILENAME = "primitive_normals.csv"

# Numerical tolerance for detecting boundary nodes
BOUNDARY_TOL = 1.0e-6


# ============================================================
# TPMS DEFINITION
# ============================================================

def primitive_level_set(x, y, z, Lx=1.0, Ly=1.0, Lz=1.0):
    """
    Primitive TPMS level-set function.

    F(x,y,z) =
        cos(2*pi*x/Lx)
      + cos(2*pi*y/Ly)
      + cos(2*pi*z/Lz)

    Zero-level surface:
        F(x,y,z) = 0
    """

    kx = 2.0 * np.pi / Lx
    ky = 2.0 * np.pi / Ly
    kz = 2.0 * np.pi / Lz

    return (
        np.cos(kx * x)
        + np.cos(ky * y)
        + np.cos(kz * z)
    )


# ============================================================
# ANALYTICAL GRADIENT
# ============================================================

def primitive_gradient(x, y, z, Lx=1.0, Ly=1.0, Lz=1.0):
    """
    Analytical gradient of Primitive TPMS function.

    grad(F) =
    [
        -kx*sin(kx*x),
        -ky*sin(ky*y),
        -kz*sin(kz*z)
    ]
    """

    kx = 2.0 * np.pi / Lx
    ky = 2.0 * np.pi / Ly
    kz = 2.0 * np.pi / Lz

    dFdx = -kx * np.sin(kx * x)
    dFdy = -ky * np.sin(ky * y)
    dFdz = -kz * np.sin(kz * z)

    return np.array([dFdx, dFdy, dFdz])


# ============================================================
# ANALYTICAL NORMALS
# ============================================================

def compute_analytical_normals(nodes, Lx, Ly, Lz):
    """
    Compute analytical unit normals from grad(F).
    """

    normals = np.zeros_like(nodes)

    for i, node in enumerate(nodes):

        x, y, z = node

        grad = primitive_gradient(
            x,
            y,
            z,
            Lx,
            Ly,
            Lz
        )

        norm_grad = np.linalg.norm(grad)

        if norm_grad > 1.0e-12:
            normals[i, :] = grad / norm_grad
        else:
            normals[i, :] = 0.0

    return normals


# ============================================================
# STRUCTURED GRID
# ============================================================

def generate_grid(Lx, Ly, Lz, Nx, Ny, Nz):

    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    z = np.linspace(0.0, Lz, Nz)

    X, Y, Z = np.meshgrid(
        x,
        y,
        z,
        indexing="ij"
    )

    return x, y, z, X, Y, Z


# ============================================================
# LEVEL-SET SURFACE EXTRACTION
# ============================================================

def extract_surface(F, Lx, Ly, Lz, Nx, Ny, Nz, level=0.0):
    """
    Extract triangulated isosurface using marching cubes.
    """

    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dz = Lz / (Nz - 1)

    vertices, faces, normals_mc, values = measure.marching_cubes(
        F,
        level=level,
        spacing=(dx, dy, dz)
    )

    return vertices, faces, normals_mc, values


# ============================================================
# BOUNDARY NODE DETECTION
# ============================================================

def find_boundary_nodes(nodes, Lx, Ly, Lz, tol=1.0e-6):

    boundaries = {}

    boundaries["xmin"] = np.where(
        np.abs(nodes[:, 0] - 0.0) <= tol
    )[0]

    boundaries["xmax"] = np.where(
        np.abs(nodes[:, 0] - Lx) <= tol
    )[0]

    boundaries["ymin"] = np.where(
        np.abs(nodes[:, 1] - 0.0) <= tol
    )[0]

    boundaries["ymax"] = np.where(
        np.abs(nodes[:, 1] - Ly) <= tol
    )[0]

    boundaries["zmin"] = np.where(
        np.abs(nodes[:, 2] - 0.0) <= tol
    )[0]

    boundaries["zmax"] = np.where(
        np.abs(nodes[:, 2] - Lz) <= tol
    )[0]

    return boundaries


# ============================================================
# BOUNDARY INFORMATION
# ============================================================

def print_boundary_information(boundaries):

    print("\nBoundary-node information")
    print("--------------------------------")

    for boundary_name, node_ids in boundaries.items():

        print(
            f"{boundary_name:5s}: "
            f"{len(node_ids):6d} nodes"
        )


# ============================================================
# CHECK PERIODIC BOUNDARY COUNTS
# ============================================================

def check_periodic_boundary_counts(boundaries):

    print("\nPeriodic boundary count check")
    print("--------------------------------")

    pairs = [
        ("xmin", "xmax"),
        ("ymin", "ymax"),
        ("zmin", "zmax")
    ]

    for b1, b2 in pairs:

        n1 = len(boundaries[b1])
        n2 = len(boundaries[b2])

        if n1 == n2:
            status = "OK"
        else:
            status = "NOT MATCHING"

        print(
            f"{b1} <-> {b2}: "
            f"{n1} / {n2}  --> {status}"
        )


# ============================================================
# SURFACE VISUALIZATION
# ============================================================

def plot_tpms_surface(
        nodes,
        elements,
        Lx,
        Ly,
        Lz,
        normals=None,
        show_normals=False,
        normal_skip=100,
        normal_scale=0.05):

    fig = plt.figure(figsize=(10, 9))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    triangle_coordinates = nodes[elements]

    surface = Poly3DCollection(
        triangle_coordinates,
        alpha=0.85,
        edgecolor="k",
        linewidth=0.05
    )

    ax.add_collection3d(surface)

    ax.set_xlim(0.0, Lx)
    ax.set_ylim(0.0, Ly)
    ax.set_zlim(0.0, Lz)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_box_aspect(
        (Lx, Ly, Lz)
    )

    ax.set_title(
        "Primitive TPMS - Zero Level Set"
    )

    # --------------------------------------------------------
    # NORMAL VECTORS
    # --------------------------------------------------------

    if show_normals and normals is not None:

        ids = np.arange(
            0,
            len(nodes),
            normal_skip
        )

        ax.quiver(
            nodes[ids, 0],
            nodes[ids, 1],
            nodes[ids, 2],
            normals[ids, 0],
            normals[ids, 1],
            normals[ids, 2],
            length=normal_scale,
            normalize=True
        )

    plt.tight_layout()
    plt.show()


# ============================================================
# ASCII STL EXPORT
# ============================================================

def export_ascii_stl(filename, nodes, elements):
    """
    Export triangular surface mesh to ASCII STL.
    """

    with open(filename, "w") as file:

        file.write("solid primitive_tpms\n")

        for element in elements:

            p1 = nodes[element[0]]
            p2 = nodes[element[1]]
            p3 = nodes[element[2]]

            v1 = p2 - p1
            v2 = p3 - p1

            normal = np.cross(v1, v2)

            norm = np.linalg.norm(normal)

            if norm > 1.0e-14:
                normal = normal / norm
            else:
                normal[:] = 0.0

            file.write(
                "  facet normal "
                f"{normal[0]:.12e} "
                f"{normal[1]:.12e} "
                f"{normal[2]:.12e}\n"
            )

            file.write(
                "    outer loop\n"
            )

            file.write(
                "      vertex "
                f"{p1[0]:.12e} "
                f"{p1[1]:.12e} "
                f"{p1[2]:.12e}\n"
            )

            file.write(
                "      vertex "
                f"{p2[0]:.12e} "
                f"{p2[1]:.12e} "
                f"{p2[2]:.12e}\n"
            )

            file.write(
                "      vertex "
                f"{p3[0]:.12e} "
                f"{p3[1]:.12e} "
                f"{p3[2]:.12e}\n"
            )

            file.write(
                "    endloop\n"
            )

            file.write(
                "  endfacet\n"
            )

        file.write(
            "endsolid primitive_tpms\n"
        )

    print(
        f"\nSTL exported: {filename}"
    )


# ============================================================
# CSV EXPORT - NODES
# ============================================================

def export_nodes_csv(filename, nodes):

    node_ids = np.arange(
        1,
        len(nodes) + 1
    ).reshape(-1, 1)

    data = np.hstack(
        (node_ids, nodes)
    )

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header="NodeID,X,Y,Z",
        comments="",
        fmt=[
            "%d",
            "%.12e",
            "%.12e",
            "%.12e"
        ]
    )

    print(
        f"Nodes exported: {filename}"
    )


# ============================================================
# CSV EXPORT - ELEMENTS
# ============================================================

def export_elements_csv(filename, elements):

    element_ids = np.arange(
        1,
        len(elements) + 1
    ).reshape(-1, 1)

    # Convert Python zero-based indices to FEM one-based indexing
    element_connectivity = elements + 1

    data = np.hstack(
        (
            element_ids,
            element_connectivity
        )
    )

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header="ElementID,Node1,Node2,Node3",
        comments="",
        fmt="%d"
    )

    print(
        f"Elements exported: {filename}"
    )


# ============================================================
# CSV EXPORT - NORMALS
# ============================================================

def export_normals_csv(filename, normals):

    node_ids = np.arange(
        1,
        len(normals) + 1
    ).reshape(-1, 1)

    data = np.hstack(
        (
            node_ids,
            normals
        )
    )

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header="NodeID,Nx,Ny,Nz",
        comments="",
        fmt=[
            "%d",
            "%.12e",
            "%.12e",
            "%.12e"
        ]
    )

    print(
        f"Normals exported: {filename}"
    )


# ============================================================
# BASIC MESH STATISTICS
# ============================================================

def print_mesh_statistics(nodes, elements):

    print("\nMesh statistics")
    print("--------------------------------")

    print(
        f"Number of nodes    : {len(nodes)}"
    )

    print(
        f"Number of triangles: {len(elements)}"
    )

    xmin = np.min(nodes[:, 0])
    xmax = np.max(nodes[:, 0])

    ymin = np.min(nodes[:, 1])
    ymax = np.max(nodes[:, 1])

    zmin = np.min(nodes[:, 2])
    zmax = np.max(nodes[:, 2])

    print("\nBounding box")
    print("--------------------------------")

    print(
        f"x = [{xmin:.6f}, {xmax:.6f}]"
    )

    print(
        f"y = [{ymin:.6f}, {ymax:.6f}]"
    )

    print(
        f"z = [{zmin:.6f}, {zmax:.6f}]"
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("========================================")
    print(" Primitive TPMS Geometry Generator")
    print("========================================")

    print("\nGenerating structured sampling grid...")

    x, y, z, X, Y, Z = generate_grid(
        Lx,
        Ly,
        Lz,
        Nx,
        Ny,
        Nz
    )

    print(
        f"Sampling grid: "
        f"{Nx} x {Ny} x {Nz}"
    )

    print(
        f"Total scalar-field points: "
        f"{Nx * Ny * Nz}"
    )

    # --------------------------------------------------------
    # LEVEL-SET EVALUATION
    # --------------------------------------------------------

    print("\nEvaluating Primitive level-set function...")

    F = primitive_level_set(
        X,
        Y,
        Z,
        Lx,
        Ly,
        Lz
    )

    print(
        f"Minimum F: {np.min(F):.6f}"
    )

    print(
        f"Maximum F: {np.max(F):.6f}"
    )

    # --------------------------------------------------------
    # ISO-SURFACE EXTRACTION
    # --------------------------------------------------------

    print(
        f"\nExtracting level set F = {LEVEL}..."
    )

    nodes, elements, normals_mc, values = extract_surface(
        F,
        Lx,
        Ly,
        Lz,
        Nx,
        Ny,
        Nz,
        LEVEL
    )

    print_mesh_statistics(
        nodes,
        elements
    )

    # --------------------------------------------------------
    # ANALYTICAL NORMALS
    # --------------------------------------------------------

    print(
        "\nComputing analytical surface normals..."
    )

    analytical_normals = compute_analytical_normals(
        nodes,
        Lx,
        Ly,
        Lz
    )

    # --------------------------------------------------------
    # BOUNDARY NODE DETECTION
    # --------------------------------------------------------

    boundaries = find_boundary_nodes(
        nodes,
        Lx,
        Ly,
        Lz,
        BOUNDARY_TOL
    )

    print_boundary_information(
        boundaries
    )

    check_periodic_boundary_counts(
        boundaries
    )

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    if EXPORT_STL:

        export_ascii_stl(
            STL_FILENAME,
            nodes,
            elements
        )

    if EXPORT_NODES:

        export_nodes_csv(
            NODE_FILENAME,
            nodes
        )

    if EXPORT_ELEMENTS:

        export_elements_csv(
            ELEMENT_FILENAME,
            elements
        )

    if EXPORT_NORMALS:

        export_normals_csv(
            NORMAL_FILENAME,
            analytical_normals
        )

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    if SHOW_SURFACE:

        print(
            "\nDisplaying TPMS surface..."
        )

        plot_tpms_surface(
            nodes,
            elements,
            Lx,
            Ly,
            Lz,
            analytical_normals,
            SHOW_NORMALS,
            NORMAL_SKIP,
            NORMAL_SCALE
        )

    print("\nFinished.")


# ============================================================
# EXECUTE
# ============================================================

if __name__ == "__main__":
    main()