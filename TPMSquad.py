# ============================================================
# TPMSquad.py
#
# TPMS triangular surface -> Gmsh remeshing ->
# quad-dominant mixed shell FEM mesh
#
# Output:
#     quadtri_nodes.csv
#     quad_elements.csv
#     triangle_elements.csv
#     quad_element_data.csv
#     triangle_element_data.csv
#     mixed_mesh.msh
#     boundary_xmin.csv
#     boundary_xmax.csv
#     boundary_ymin.csv
#     boundary_ymax.csv
#     boundary_zmin.csv
#     boundary_zmax.csv
#
# ============================================================

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import gmsh

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ============================================================
# USER PARAMETERS
# ============================================================

# Input files produced by TPMSgen.py
NODE_FILE = "primitive_nodes.csv"
ELEMENT_FILE = "primitive_elements.csv"

# Unit-cell dimensions
Lx = 1.0
Ly = 1.0
Lz = 1.0


# ============================================================
# FEM MESH DENSITY
# ============================================================

# Approximate desired edge size
TARGET_ELEMENT_SIZE = 0.04


# ============================================================
# GMSH SURFACE CLASSIFICATION
# ============================================================

CLASSIFY_ANGLE_DEG = 40.0

INCLUDE_BOUNDARY = True

FORCE_PARAMETRIZABLE_PATCHES = True

CURVE_ANGLE_DEG = 180.0


# ============================================================
# GMSH MESH SETTINGS
# ============================================================

# 1 = Blossom recombination
#
# IMPORTANT:
# This is intentionally NOT a full-quad algorithm.
#
# Good quads are generated where possible.
# Triangles are retained where quad formation is unsuitable.
RECOMBINATION_ALGORITHM = 1

# Frontal-Delaunay triangular precursor mesh
SURFACE_MESH_ALGORITHM = 6

# Mesh smoothing
SMOOTHING_STEPS = 10


# ============================================================
# MATERIAL / SHELL PARAMETERS
# ============================================================

THICKNESS = 0.02

E = 210.0e9
NU = 0.30
RHO = 7850.0


# ============================================================
# NUMERICAL TOLERANCES
# ============================================================

AREA_TOL = 1.0e-12

BOUNDARY_TOL = 1.0e-6 * max(
    Lx,
    Ly,
    Lz
)


# ============================================================
# VISUALIZATION
# ============================================================

SHOW_MESH = True

SHOW_NODES = False

SHOW_ELEMENT_EDGES = True

MESH_ALPHA = 0.80

EDGE_LINEWIDTH = 0.35

NODE_SIZE = 3.0

VIEW_ELEVATION = 25.0

VIEW_AZIMUTH = 45.0


# ============================================================
# OUTPUT FILES
# ============================================================

TEMP_STL_FILE = "_tpms_temp_surface.stl"

OUTPUT_NODE_FILE = "quadtri_nodes.csv"

OUTPUT_QUAD_FILE = "quad_elements.csv"

OUTPUT_TRIANGLE_FILE = "triangle_elements.csv"

OUTPUT_QUAD_DATA_FILE = "quad_element_data.csv"

OUTPUT_TRIANGLE_DATA_FILE = "triangle_element_data.csv"

OUTPUT_MSH_FILE = "mixed_mesh.msh"

OUTPUT_XMIN = "boundary_xmin.csv"
OUTPUT_XMAX = "boundary_xmax.csv"

OUTPUT_YMIN = "boundary_ymin.csv"
OUTPUT_YMAX = "boundary_ymax.csv"

OUTPUT_ZMIN = "boundary_zmin.csv"
OUTPUT_ZMAX = "boundary_zmax.csv"


# ============================================================
# READ ORIGINAL NODES
# ============================================================

def read_nodes(filename):

    data = np.loadtxt(
        filename,
        delimiter=",",
        skiprows=1
    )

    node_ids = data[:, 0].astype(int)

    nodes = data[:, 1:4]

    return node_ids, nodes


# ============================================================
# READ ORIGINAL TRIANGLES
# ============================================================

def read_elements(filename):

    data = np.loadtxt(
        filename,
        delimiter=",",
        skiprows=1
    )

    element_ids = data[:, 0].astype(int)

    elements = data[:, 1:4].astype(int)

    return element_ids, elements


# ============================================================
# CHECK ORIGINAL INPUT
# ============================================================

def check_input_data(
        node_ids,
        nodes,
        elements):

    print("\nChecking original surface data")
    print("----------------------------------------")

    if len(node_ids) != len(nodes):

        raise ValueError(
            "Node ID count does not match coordinate count."
        )

    if np.min(elements) < 1:

        raise ValueError(
            "Element connectivity contains node IDs < 1."
        )

    if np.max(elements) > len(nodes):

        raise ValueError(
            "Element connectivity refers to undefined nodes."
        )

    print(
        f"Original nodes       : {len(nodes)}"
    )

    print(
        f"Original triangles   : {len(elements)}"
    )

    print(
        "Input connectivity   : OK"
    )


# ============================================================
# WRITE TEMPORARY ASCII STL
# ============================================================

def write_ascii_stl(
        filename,
        nodes,
        elements):

    print(
        "\nCreating temporary STL surface..."
    )

    with open(
            filename,
            "w") as file:

        file.write(
            "solid tpms_surface\n"
        )

        for element in elements:

            n1 = element[0] - 1
            n2 = element[1] - 1
            n3 = element[2] - 1

            p1 = nodes[n1]
            p2 = nodes[n2]
            p3 = nodes[n3]

            v1 = p2 - p1
            v2 = p3 - p1

            normal = np.cross(
                v1,
                v2
            )

            normal_norm = np.linalg.norm(
                normal
            )

            if normal_norm > AREA_TOL:

                normal = (
                    normal /
                    normal_norm
                )

            else:

                normal = np.zeros(
                    3
                )

            file.write(
                "  facet normal "
                f"{normal[0]:.15e} "
                f"{normal[1]:.15e} "
                f"{normal[2]:.15e}\n"
            )

            file.write(
                "    outer loop\n"
            )

            file.write(
                "      vertex "
                f"{p1[0]:.15e} "
                f"{p1[1]:.15e} "
                f"{p1[2]:.15e}\n"
            )

            file.write(
                "      vertex "
                f"{p2[0]:.15e} "
                f"{p2[1]:.15e} "
                f"{p2[2]:.15e}\n"
            )

            file.write(
                "      vertex "
                f"{p3[0]:.15e} "
                f"{p3[1]:.15e} "
                f"{p3[2]:.15e}\n"
            )

            file.write(
                "    endloop\n"
            )

            file.write(
                "  endfacet\n"
            )

        file.write(
            "endsolid tpms_surface\n"
        )

    print(
        f"Temporary STL created: {filename}"
    )


# ============================================================
# GENERATE QUAD-DOMINANT MESH
# ============================================================

def generate_mixed_mesh(
        stl_filename):

    print(
        "\n========================================"
    )

    print(
        " Gmsh quad-dominant remeshing"
    )

    print(
        "========================================"
    )

    gmsh.initialize()

    gmsh.option.setNumber(
        "General.Terminal",
        1
    )

    gmsh.model.add(
        "TPMS_Mixed"
    )

    # --------------------------------------------------------
    # Import STL
    # --------------------------------------------------------

    print(
        "\nImporting triangulated TPMS surface..."
    )

    gmsh.merge(
        stl_filename
    )

    # --------------------------------------------------------
    # Classify surface
    # --------------------------------------------------------

    print(
        "Classifying surface into "
        "parametrizable patches..."
    )

    angle = math.radians(
        CLASSIFY_ANGLE_DEG
    )

    curve_angle = math.radians(
        CURVE_ANGLE_DEG
    )

    gmsh.model.mesh.classifySurfaces(
        angle,
        INCLUDE_BOUNDARY,
        FORCE_PARAMETRIZABLE_PATCHES,
        curve_angle
    )

    # --------------------------------------------------------
    # Reconstruct discrete geometry
    # --------------------------------------------------------

    print(
        "Creating discrete Gmsh geometry..."
    )

    gmsh.model.mesh.createGeometry()

    surfaces = gmsh.model.getEntities(
        2
    )

    print(
        f"Number of surface patches: "
        f"{len(surfaces)}"
    )

    if len(surfaces) == 0:

        raise RuntimeError(
            "No Gmsh surface patches were created."
        )

    # --------------------------------------------------------
    # Mesh size
    # --------------------------------------------------------

    gmsh.option.setNumber(
        "Mesh.MeshSizeMin",
        TARGET_ELEMENT_SIZE
    )

    gmsh.option.setNumber(
        "Mesh.MeshSizeMax",
        TARGET_ELEMENT_SIZE
    )

    gmsh.option.setNumber(
        "Mesh.MeshSizeFromPoints",
        0
    )

    gmsh.option.setNumber(
        "Mesh.MeshSizeFromCurvature",
        0
    )

    gmsh.option.setNumber(
        "Mesh.MeshSizeExtendFromBoundary",
        0
    )

    # --------------------------------------------------------
    # Surface mesh algorithm
    # --------------------------------------------------------

    gmsh.option.setNumber(
        "Mesh.Algorithm",
        SURFACE_MESH_ALGORITHM
    )

    # --------------------------------------------------------
    # Quad-dominant recombination
    # --------------------------------------------------------

    gmsh.option.setNumber(
        "Mesh.RecombineAll",
        1
    )

    gmsh.option.setNumber(
        "Mesh.RecombinationAlgorithm",
        RECOMBINATION_ALGORITHM
    )

    gmsh.option.setNumber(
        "Mesh.Smoothing",
        SMOOTHING_STEPS
    )

    # --------------------------------------------------------
    # Apply recombination to every patch
    # --------------------------------------------------------

    for dim, tag in surfaces:

        gmsh.model.mesh.setAlgorithm(
            dim,
            tag,
            SURFACE_MESH_ALGORITHM
        )

        gmsh.model.mesh.setRecombine(
            dim,
            tag
        )

        gmsh.model.mesh.setSmoothing(
            dim,
            tag,
            SMOOTHING_STEPS
        )

    # --------------------------------------------------------
    # Generate surface mesh
    # --------------------------------------------------------

    print(
        "\nGenerating quad-dominant shell mesh..."
    )

    gmsh.model.mesh.generate(
        2
    )

    gmsh.model.mesh.renumberNodes()

    gmsh.model.mesh.renumberElements()

    # --------------------------------------------------------
    # Native Gmsh file
    # --------------------------------------------------------

    gmsh.write(
        OUTPUT_MSH_FILE
    )

    print(
        f"Gmsh mesh written: "
        f"{OUTPUT_MSH_FILE}"
    )

    # --------------------------------------------------------
    # Extract numpy data
    # --------------------------------------------------------

    nodes, quads, triangles = (
        extract_gmsh_mesh()
    )

    return (
        nodes,
        quads,
        triangles
    )


# ============================================================
# EXTRACT GMSH MESH
# ============================================================

def extract_gmsh_mesh():

    print(
        "\nExtracting Gmsh FEM data..."
    )

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    (
        node_tags,
        coordinates,
        _
    ) = gmsh.model.mesh.getNodes()

    node_tags = np.asarray(
        node_tags,
        dtype=np.int64
    )

    coordinates = np.asarray(
        coordinates,
        dtype=float
    ).reshape(
        (-1, 3)
    )

    # --------------------------------------------------------
    # Gmsh tag -> continuous FEM numbering
    # --------------------------------------------------------

    tag_to_index = {}

    for i, tag in enumerate(
            node_tags):

        tag_to_index[
            int(tag)
        ] = i

    # --------------------------------------------------------
    # Elements
    # --------------------------------------------------------

    (
        element_types,
        element_tags,
        element_node_tags
    ) = gmsh.model.mesh.getElements(
        2
    )

    quads = []

    triangles = []

    for i in range(
            len(element_types)):

        element_type = (
            element_types[i]
        )

        properties = (
            gmsh.model.mesh.getElementProperties(
                element_type
            )
        )

        element_name = properties[0]

        number_of_nodes = properties[3]

        connectivity = np.asarray(
            element_node_tags[i],
            dtype=np.int64
        )

        connectivity = connectivity.reshape(
            (
                -1,
                number_of_nodes
            )
        )

        print(
            f"Element type {element_type}: "
            f"{element_name}, "
            f"{len(connectivity)} elements"
        )

        # ----------------------------------------------------
        # Linear quadrilateral
        # ----------------------------------------------------

        if number_of_nodes == 4:

            for element in connectivity:

                new_element = []

                for tag in element:

                    index = tag_to_index[
                        int(tag)
                    ]

                    new_element.append(
                        index + 1
                    )

                quads.append(
                    new_element
                )

        # ----------------------------------------------------
        # Linear triangle
        # ----------------------------------------------------

        elif number_of_nodes == 3:

            for element in connectivity:

                new_element = []

                for tag in element:

                    index = tag_to_index[
                        int(tag)
                    ]

                    new_element.append(
                        index + 1
                    )

                triangles.append(
                    new_element
                )

    quads = np.asarray(
        quads,
        dtype=int
    )

    triangles = np.asarray(
        triangles,
        dtype=int
    )

    if quads.size == 0:

        quads = np.empty(
            (0, 4),
            dtype=int
        )

    if triangles.size == 0:

        triangles = np.empty(
            (0, 3),
            dtype=int
        )

    return (
        coordinates,
        quads,
        triangles
    )


# ============================================================
# MESH COMPOSITION
# ============================================================

def print_mesh_composition(
        nodes,
        quads,
        triangles):

    number_of_quads = len(
        quads
    )

    number_of_triangles = len(
        triangles
    )

    total_elements = (
        number_of_quads +
        number_of_triangles
    )

    if total_elements > 0:

        quad_fraction = (
            100.0 *
            number_of_quads /
            total_elements
        )

        triangle_fraction = (
            100.0 *
            number_of_triangles /
            total_elements
        )

    else:

        quad_fraction = 0.0
        triangle_fraction = 0.0

    print(
        "\n========================================"
    )

    print(
        " Mixed shell mesh composition"
    )

    print(
        "========================================"
    )

    print(
        f"Nodes          : "
        f"{len(nodes)}"
    )

    print(
        f"Quadrilaterals : "
        f"{number_of_quads}"
    )

    print(
        f"Triangles      : "
        f"{number_of_triangles}"
    )

    print(
        f"Quad fraction  : "
        f"{quad_fraction:.2f} %"
    )

    print(
        f"Triangle frac. : "
        f"{triangle_fraction:.2f} %"
    )

    if total_elements == 0:

        raise RuntimeError(
            "Gmsh produced no 2D elements."
        )

    print(
        "\nMixed quad/triangle mesh: OK"
    )


# ============================================================
# QUAD ELEMENT GEOMETRY
# ============================================================

def compute_quad_geometry(
        nodes,
        quads):

    n_elements = len(
        quads
    )

    centroids = np.zeros(
        (
            n_elements,
            3
        )
    )

    normals = np.zeros(
        (
            n_elements,
            3
        )
    )

    areas = np.zeros(
        n_elements
    )

    edge_lengths = np.zeros(
        (
            n_elements,
            4
        )
    )

    for e in range(
            n_elements):

        n1, n2, n3, n4 = (
            quads[e] - 1
        )

        p1 = nodes[n1]
        p2 = nodes[n2]
        p3 = nodes[n3]
        p4 = nodes[n4]

        centroids[e] = (
            p1 +
            p2 +
            p3 +
            p4
        ) / 4.0

        cross1 = np.cross(
            p2 - p1,
            p3 - p1
        )

        cross2 = np.cross(
            p3 - p1,
            p4 - p1
        )

        area1 = (
            0.5 *
            np.linalg.norm(
                cross1
            )
        )

        area2 = (
            0.5 *
            np.linalg.norm(
                cross2
            )
        )

        areas[e] = (
            area1 +
            area2
        )

        normal = (
            cross1 +
            cross2
        )

        normal_norm = np.linalg.norm(
            normal
        )

        if normal_norm > AREA_TOL:

            normals[e] = (
                normal /
                normal_norm
            )

        edge_lengths[e, 0] = (
            np.linalg.norm(
                p2 - p1
            )
        )

        edge_lengths[e, 1] = (
            np.linalg.norm(
                p3 - p2
            )
        )

        edge_lengths[e, 2] = (
            np.linalg.norm(
                p4 - p3
            )
        )

        edge_lengths[e, 3] = (
            np.linalg.norm(
                p1 - p4
            )
        )

    return (
        centroids,
        normals,
        areas,
        edge_lengths
    )


# ============================================================
# TRIANGLE ELEMENT GEOMETRY
# ============================================================

def compute_triangle_geometry(
        nodes,
        triangles):

    n_elements = len(
        triangles
    )

    centroids = np.zeros(
        (
            n_elements,
            3
        )
    )

    normals = np.zeros(
        (
            n_elements,
            3
        )
    )

    areas = np.zeros(
        n_elements
    )

    edge_lengths = np.zeros(
        (
            n_elements,
            3
        )
    )

    for e in range(
            n_elements):

        n1, n2, n3 = (
            triangles[e] - 1
        )

        p1 = nodes[n1]
        p2 = nodes[n2]
        p3 = nodes[n3]

        centroids[e] = (
            p1 +
            p2 +
            p3
        ) / 3.0

        cross = np.cross(
            p2 - p1,
            p3 - p1
        )

        cross_norm = np.linalg.norm(
            cross
        )

        areas[e] = (
            0.5 *
            cross_norm
        )

        if cross_norm > AREA_TOL:

            normals[e] = (
                cross /
                cross_norm
            )

        edge_lengths[e, 0] = (
            np.linalg.norm(
                p2 - p1
            )
        )

        edge_lengths[e, 1] = (
            np.linalg.norm(
                p3 - p2
            )
        )

        edge_lengths[e, 2] = (
            np.linalg.norm(
                p1 - p3
            )
        )

    return (
        centroids,
        normals,
        areas,
        edge_lengths
    )


# ============================================================
# QUAD ANGLE QUALITY
# ============================================================

def compute_quad_quality(
        nodes,
        quads):

    quality = np.zeros(
        len(quads)
    )

    minimum_angles = np.zeros(
        len(quads)
    )

    maximum_angles = np.zeros(
        len(quads)
    )

    for e, element in enumerate(
            quads):

        p = nodes[
            element - 1
        ]

        angles = []

        for i in range(
                4):

            previous_point = p[
                (i - 1) % 4
            ]

            current_point = p[i]

            next_point = p[
                (i + 1) % 4
            ]

            v1 = (
                previous_point -
                current_point
            )

            v2 = (
                next_point -
                current_point
            )

            n1 = np.linalg.norm(
                v1
            )

            n2 = np.linalg.norm(
                v2
            )

            if (
                n1 < AREA_TOL or
                n2 < AREA_TOL
            ):

                angle = 0.0

            else:

                cosine = np.dot(
                    v1,
                    v2
                ) / (
                    n1 *
                    n2
                )

                cosine = np.clip(
                    cosine,
                    -1.0,
                    1.0
                )

                angle = math.degrees(
                    math.acos(
                        cosine
                    )
                )

            angles.append(
                angle
            )

        angles = np.asarray(
            angles
        )

        minimum_angles[e] = np.min(
            angles
        )

        maximum_angles[e] = np.max(
            angles
        )

        deviation = np.max(
            np.abs(
                angles - 90.0
            )
        )

        quality[e] = max(
            0.0,
            1.0 -
            deviation / 90.0
        )

    return (
        quality,
        minimum_angles,
        maximum_angles
    )


# ============================================================
# TRIANGLE QUALITY
# ============================================================

def compute_triangle_quality(
        nodes,
        triangles):

    """
    Equilateral triangle:
        quality = 1

    Degenerate triangle:
        quality -> 0
    """

    quality = np.zeros(
        len(triangles)
    )

    for e, element in enumerate(
            triangles):

        p1, p2, p3 = nodes[
            element - 1
        ]

        a = np.linalg.norm(
            p2 - p1
        )

        b = np.linalg.norm(
            p3 - p2
        )

        c = np.linalg.norm(
            p1 - p3
        )

        cross = np.cross(
            p2 - p1,
            p3 - p1
        )

        area = (
            0.5 *
            np.linalg.norm(
                cross
            )
        )

        denominator = (
            a * a +
            b * b +
            c * c
        )

        if denominator > AREA_TOL:

            quality[e] = (
                4.0 *
                math.sqrt(3.0) *
                area /
                denominator
            )

        else:

            quality[e] = 0.0

    return quality


# ============================================================
# PRINT QUALITY INFORMATION
# ============================================================

def print_quality_information(
        quad_edge_lengths,
        quad_areas,
        quad_quality,
        quad_min_angles,
        quad_max_angles,
        tri_edge_lengths,
        tri_areas,
        tri_quality):

    print(
        "\n========================================"
    )

    print(
        " Mesh-quality information"
    )

    print(
        "========================================"
    )

    print(
        f"Target element size : "
        f"{TARGET_ELEMENT_SIZE:.6e}"
    )

    if len(
            quad_areas) > 0:

        print(
            "\nQuadrilateral elements"
        )

        print(
            "----------------------------------------"
        )

        print(
            f"Mean edge length    : "
            f"{np.mean(quad_edge_lengths):.6e}"
        )

        print(
            f"Min edge length     : "
            f"{np.min(quad_edge_lengths):.6e}"
        )

        print(
            f"Max edge length     : "
            f"{np.max(quad_edge_lengths):.6e}"
        )

        print(
            f"Mean area           : "
            f"{np.mean(quad_areas):.6e}"
        )

        print(
            f"Mean quality        : "
            f"{np.mean(quad_quality):.6f}"
        )

        print(
            f"Minimum quality     : "
            f"{np.min(quad_quality):.6f}"
        )

        print(
            f"Minimum angle       : "
            f"{np.min(quad_min_angles):.3f} deg"
        )

        print(
            f"Maximum angle       : "
            f"{np.max(quad_max_angles):.3f} deg"
        )

    if len(
            tri_areas) > 0:

        print(
            "\nTriangular elements"
        )

        print(
            "----------------------------------------"
        )

        print(
            f"Mean edge length    : "
            f"{np.mean(tri_edge_lengths):.6e}"
        )

        print(
            f"Min edge length     : "
            f"{np.min(tri_edge_lengths):.6e}"
        )

        print(
            f"Max edge length     : "
            f"{np.max(tri_edge_lengths):.6e}"
        )

        print(
            f"Mean area           : "
            f"{np.mean(tri_areas):.6e}"
        )

        print(
            f"Mean quality        : "
            f"{np.mean(tri_quality):.6f}"
        )

        print(
            f"Minimum quality     : "
            f"{np.min(tri_quality):.6f}"
        )


# ============================================================
# FIND BOUNDARY NODES
# ============================================================

def find_boundary_nodes(
        nodes):

    node_ids = np.arange(
        1,
        len(nodes) + 1
    )

    boundaries = {}

    boundaries["xmin"] = (
        node_ids[
            np.abs(
                nodes[:, 0]
            ) <= BOUNDARY_TOL
        ]
    )

    boundaries["xmax"] = (
        node_ids[
            np.abs(
                nodes[:, 0] - Lx
            ) <= BOUNDARY_TOL
        ]
    )

    boundaries["ymin"] = (
        node_ids[
            np.abs(
                nodes[:, 1]
            ) <= BOUNDARY_TOL
        ]
    )

    boundaries["ymax"] = (
        node_ids[
            np.abs(
                nodes[:, 1] - Ly
            ) <= BOUNDARY_TOL
        ]
    )

    boundaries["zmin"] = (
        node_ids[
            np.abs(
                nodes[:, 2]
            ) <= BOUNDARY_TOL
        ]
    )

    boundaries["zmax"] = (
        node_ids[
            np.abs(
                nodes[:, 2] - Lz
            ) <= BOUNDARY_TOL
        ]
    )

    return boundaries


# ============================================================
# PRINT BOUNDARY INFORMATION
# ============================================================

def print_boundary_information(
        boundaries):

    print(
        "\n========================================"
    )

    print(
        " Unit-cell boundary nodes"
    )

    print(
        "========================================"
    )

    for name, ids in boundaries.items():

        print(
            f"{name:5s}: "
            f"{len(ids):6d} nodes"
        )


# ============================================================
# PERIODIC COUNT CHECK
# ============================================================

def periodic_count_check(
        boundaries):

    print(
        "\nPeriodic face count check"
    )

    print(
        "----------------------------------------"
    )

    pairs = [
        ("xmin", "xmax"),
        ("ymin", "ymax"),
        ("zmin", "zmax")
    ]

    for first, second in pairs:

        n1 = len(
            boundaries[first]
        )

        n2 = len(
            boundaries[second]
        )

        status = (
            "OK"
            if n1 == n2
            else "NOT MATCHING"
        )

        print(
            f"{first} <-> {second}: "
            f"{n1} / {n2} "
            f"--> {status}"
        )


# ============================================================
# EXPORT NODES
# ============================================================

def export_nodes(
        filename,
        nodes):

    node_ids = np.arange(
        1,
        len(nodes) + 1
    )

    data = np.column_stack(
        (
            node_ids,
            nodes
        )
    )

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header=(
            "NodeID,X,Y,Z"
        ),
        comments="",
        fmt=[
            "%d",
            "%.15e",
            "%.15e",
            "%.15e"
        ]
    )


# ============================================================
# EXPORT QUADS
# ============================================================

def export_quads(
        filename,
        quads):

    if len(
            quads) == 0:

        return

    element_ids = np.arange(
        1,
        len(quads) + 1
    )

    thickness = np.full(
        len(quads),
        THICKNESS
    )

    data = np.column_stack(
        (
            element_ids,
            quads,
            thickness
        )
    )

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header=(
            "ElementID,"
            "Node1,Node2,Node3,Node4,"
            "Thickness"
        ),
        comments="",
        fmt=[
            "%d",
            "%d",
            "%d",
            "%d",
            "%d",
            "%.15e"
        ]
    )


# ============================================================
# EXPORT TRIANGLES
# ============================================================

def export_triangles(
        filename,
        triangles):

    if len(
            triangles) == 0:

        return

    element_ids = np.arange(
        1,
        len(triangles) + 1
    )

    thickness = np.full(
        len(triangles),
        THICKNESS
    )

    data = np.column_stack(
        (
            element_ids,
            triangles,
            thickness
        )
    )

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header=(
            "ElementID,"
            "Node1,Node2,Node3,"
            "Thickness"
        ),
        comments="",
        fmt=[
            "%d",
            "%d",
            "%d",
            "%d",
            "%.15e"
        ]
    )


# ============================================================
# EXPORT QUAD ELEMENT DATA
# ============================================================

def export_quad_data(
        filename,
        centroids,
        normals,
        areas,
        quality):

    if len(
            centroids) == 0:

        return

    element_ids = np.arange(
        1,
        len(centroids) + 1
    )

    data = np.column_stack(
        (
            element_ids,
            centroids,
            normals,
            areas,
            quality
        )
    )

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header=(
            "ElementID,"
            "CentroidX,CentroidY,CentroidZ,"
            "NormalX,NormalY,NormalZ,"
            "Area,Quality"
        ),
        comments="",
        fmt=[
            "%d",
            "%.15e",
            "%.15e",
            "%.15e",
            "%.15e",
            "%.15e",
            "%.15e",
            "%.15e",
            "%.15e"
        ]
    )


# ============================================================
# EXPORT TRIANGLE ELEMENT DATA
# ============================================================

def export_triangle_data(
        filename,
        centroids,
        normals,
        areas,
        quality):

    if len(
            centroids) == 0:

        return

    element_ids = np.arange(
        1,
        len(centroids) + 1
    )

    data = np.column_stack(
        (
            element_ids,
            centroids,
            normals,
            areas,
            quality
        )
    )

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header=(
            "ElementID,"
            "CentroidX,CentroidY,CentroidZ,"
            "NormalX,NormalY,NormalZ,"
            "Area,Quality"
        ),
        comments="",
        fmt=[
            "%d",
            "%.15e",
            "%.15e",
            "%.15e",
            "%.15e",
            "%.15e",
            "%.15e",
            "%.15e",
            "%.15e"
        ]
    )


# ============================================================
# EXPORT BOUNDARY NODES
# ============================================================

def export_boundary(
        filename,
        node_ids):

    np.savetxt(
        filename,
        node_ids,
        delimiter=",",
        header="NodeID",
        comments="",
        fmt="%d"
    )


# ============================================================
# VISUALIZATION
# ============================================================

def plot_mixed_mesh(
        nodes,
        quads,
        triangles):

    print(
        "\nOpening interactive mixed FEM mesh viewer..."
    )

    fig = plt.figure(
        figsize=(
            10,
            9
        )
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # --------------------------------------------------------
    # QUADRILATERALS
    # --------------------------------------------------------

    if len(
            quads) > 0:

        quad_polygons = nodes[
            quads - 1
        ]

        if SHOW_ELEMENT_EDGES:

            quad_collection = (
                Poly3DCollection(
                    quad_polygons,
                    alpha=MESH_ALPHA,
                    edgecolor="black",
                    linewidth=EDGE_LINEWIDTH
                )
            )

        else:

            quad_collection = (
                Poly3DCollection(
                    quad_polygons,
                    alpha=MESH_ALPHA
                )
            )

        ax.add_collection3d(
            quad_collection
        )

    # --------------------------------------------------------
    # TRIANGLES
    # --------------------------------------------------------

    if len(
            triangles) > 0:

        triangle_polygons = nodes[
            triangles - 1
        ]

        if SHOW_ELEMENT_EDGES:

            triangle_collection = (
                Poly3DCollection(
                    triangle_polygons,
                    alpha=MESH_ALPHA,
                    edgecolor="black",
                    linewidth=EDGE_LINEWIDTH
                )
            )

        else:

            triangle_collection = (
                Poly3DCollection(
                    triangle_polygons,
                    alpha=MESH_ALPHA
                )
            )

        ax.add_collection3d(
            triangle_collection
        )

    # --------------------------------------------------------
    # NODES
    # --------------------------------------------------------

    if SHOW_NODES:

        ax.scatter(
            nodes[:, 0],
            nodes[:, 1],
            nodes[:, 2],
            s=NODE_SIZE
        )

    # --------------------------------------------------------
    # AXES
    # --------------------------------------------------------

    ax.set_xlim(
        0.0,
        Lx
    )

    ax.set_ylim(
        0.0,
        Ly
    )

    ax.set_zlim(
        0.0,
        Lz
    )

    ax.set_box_aspect(
        (
            Lx,
            Ly,
            Lz
        )
    )

    ax.set_xlabel(
        "X"
    )

    ax.set_ylabel(
        "Y"
    )

    ax.set_zlabel(
        "Z"
    )

    ax.view_init(
        elev=VIEW_ELEVATION,
        azim=VIEW_AZIMUTH
    )

    ax.mouse_init()

    ax.set_title(
        "Primitive TPMS - "
        "Mixed Quad/Triangle Shell FEM Mesh\n"
        "Click and drag to rotate"
    )

    plt.tight_layout()

    print(
        "\nViewer controls:"
    )

    print(
        "Left mouse + drag  : Rotate"
    )

    print(
        "Scroll             : Zoom"
    )

    print(
        "Right mouse + drag : Pan"
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "============================================="
    )

    print(
        " TPMS Quad-Dominant FEM Mesh Generator"
    )

    print(
        "============================================="
    )

    print(
        f"\nTarget FEM element size = "
        f"{TARGET_ELEMENT_SIZE}"
    )

    # --------------------------------------------------------
    # READ ORIGINAL GEOMETRY
    # --------------------------------------------------------

    (
        node_ids,
        original_nodes
    ) = read_nodes(
        NODE_FILE
    )

    (
        element_ids,
        original_triangles
    ) = read_elements(
        ELEMENT_FILE
    )

    check_input_data(
        node_ids,
        original_nodes,
        original_triangles
    )

    # --------------------------------------------------------
    # TEMPORARY STL
    # --------------------------------------------------------

    write_ascii_stl(
        TEMP_STL_FILE,
        original_nodes,
        original_triangles
    )

    try:

        # ----------------------------------------------------
        # GMSH REMESH
        # ----------------------------------------------------

        (
            mesh_nodes,
            quads,
            triangles
        ) = generate_mixed_mesh(
            TEMP_STL_FILE
        )

        # ----------------------------------------------------
        # MESH COMPOSITION
        # ----------------------------------------------------

        print_mesh_composition(
            mesh_nodes,
            quads,
            triangles
        )

        # ----------------------------------------------------
        # QUAD GEOMETRY
        # ----------------------------------------------------

        if len(
                quads) > 0:

            (
                quad_centroids,
                quad_normals,
                quad_areas,
                quad_edge_lengths
            ) = compute_quad_geometry(
                mesh_nodes,
                quads
            )

            (
                quad_quality,
                quad_min_angles,
                quad_max_angles
            ) = compute_quad_quality(
                mesh_nodes,
                quads
            )

        else:

            quad_centroids = np.empty(
                (0, 3)
            )

            quad_normals = np.empty(
                (0, 3)
            )

            quad_areas = np.empty(
                0
            )

            quad_edge_lengths = np.empty(
                (0, 4)
            )

            quad_quality = np.empty(
                0
            )

            quad_min_angles = np.empty(
                0
            )

            quad_max_angles = np.empty(
                0
            )

        # ----------------------------------------------------
        # TRIANGLE GEOMETRY
        # ----------------------------------------------------

        if len(
                triangles) > 0:

            (
                tri_centroids,
                tri_normals,
                tri_areas,
                tri_edge_lengths
            ) = compute_triangle_geometry(
                mesh_nodes,
                triangles
            )

            tri_quality = (
                compute_triangle_quality(
                    mesh_nodes,
                    triangles
                )
            )

        else:

            tri_centroids = np.empty(
                (0, 3)
            )

            tri_normals = np.empty(
                (0, 3)
            )

            tri_areas = np.empty(
                0
            )

            tri_edge_lengths = np.empty(
                (0, 3)
            )

            tri_quality = np.empty(
                0
            )

        # ----------------------------------------------------
        # QUALITY INFORMATION
        # ----------------------------------------------------

        print_quality_information(
            quad_edge_lengths,
            quad_areas,
            quad_quality,
            quad_min_angles,
            quad_max_angles,
            tri_edge_lengths,
            tri_areas,
            tri_quality
        )

        # ----------------------------------------------------
        # BOUNDARY NODES
        # ----------------------------------------------------

        boundaries = (
            find_boundary_nodes(
                mesh_nodes
            )
        )

        print_boundary_information(
            boundaries
        )

        periodic_count_check(
            boundaries
        )

        # ----------------------------------------------------
        # EXPORT
        # ----------------------------------------------------

        print(
            "\nExporting mixed FEM mesh..."
        )

        export_nodes(
            OUTPUT_NODE_FILE,
            mesh_nodes
        )

        export_quads(
            OUTPUT_QUAD_FILE,
            quads
        )

        export_triangles(
            OUTPUT_TRIANGLE_FILE,
            triangles
        )

        export_quad_data(
            OUTPUT_QUAD_DATA_FILE,
            quad_centroids,
            quad_normals,
            quad_areas,
            quad_quality
        )

        export_triangle_data(
            OUTPUT_TRIANGLE_DATA_FILE,
            tri_centroids,
            tri_normals,
            tri_areas,
            tri_quality
        )

        # ----------------------------------------------------
        # BOUNDARY EXPORT
        # ----------------------------------------------------

        export_boundary(
            OUTPUT_XMIN,
            boundaries["xmin"]
        )

        export_boundary(
            OUTPUT_XMAX,
            boundaries["xmax"]
        )

        export_boundary(
            OUTPUT_YMIN,
            boundaries["ymin"]
        )

        export_boundary(
            OUTPUT_YMAX,
            boundaries["ymax"]
        )

        export_boundary(
            OUTPUT_ZMIN,
            boundaries["zmin"]
        )

        export_boundary(
            OUTPUT_ZMAX,
            boundaries["zmax"]
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        print(
            "\n========================================"
        )

        print(
            " Generated files"
        )

        print(
            "========================================"
        )

        print(
            OUTPUT_NODE_FILE
        )

        if len(
                quads) > 0:

            print(
                OUTPUT_QUAD_FILE
            )

            print(
                OUTPUT_QUAD_DATA_FILE
            )

        if len(
                triangles) > 0:

            print(
                OUTPUT_TRIANGLE_FILE
            )

            print(
                OUTPUT_TRIANGLE_DATA_FILE
            )

        print(
            OUTPUT_MSH_FILE
        )

        print(
            OUTPUT_XMIN
        )

        print(
            OUTPUT_XMAX
        )

        print(
            OUTPUT_YMIN
        )

        print(
            OUTPUT_YMAX
        )

        print(
            OUTPUT_ZMIN
        )

        print(
            OUTPUT_ZMAX
        )

        # ----------------------------------------------------
        # VISUALIZATION
        # ----------------------------------------------------

        if SHOW_MESH:

            plot_mixed_mesh(
                mesh_nodes,
                quads,
                triangles
            )

    finally:

        # ----------------------------------------------------
        # CLOSE GMSH
        # ----------------------------------------------------

        try:

            gmsh.finalize()

        except Exception:

            pass

        # ----------------------------------------------------
        # REMOVE TEMPORARY STL
        # ----------------------------------------------------

        if os.path.exists(
                TEMP_STL_FILE):

            try:

                os.remove(
                    TEMP_STL_FILE
                )

            except Exception:

                pass

    print(
        "\nFinished."
    )


# ============================================================
# EXECUTE
# ============================================================

if __name__ == "__main__":

    main()