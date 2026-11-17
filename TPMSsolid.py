# ============================================================
# TPMSsolid.py
#
# TPMS SOLID FEM MESH GENERATOR
#
# MIDSURFACE:
#
#       QUAD4 + TRI3
#
# SOLID GENERATION:
#
#       QUAD4 -> HEX8
#
#       TRI3 -> WEDGE6 -> 3 x TET4
#
#
# FINAL FEM DATABASE:
#
#       NODES + HEX8 + TET4
#
#
# VALIDATION:
#
# 1. Unit-cell boundary check
#
#       0 <= x <= Lx
#       0 <= y <= Ly
#       0 <= z <= Lz
#
# 2. Jacobian validity check
#
#       HEX8:
#           det(J) evaluated at 8 Gauss points
#
#       TET4:
#           constant det(J)
#
# Valid element:
#
#       det(J) > 0
#
#
# Visualization is ONLY used for visual inspection.
#
# ============================================================


import os
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ============================================================
# USER PARAMETERS
# ============================================================


# ------------------------------------------------------------
# INPUT FILES
# ------------------------------------------------------------

NODE_FILE = "quadtri_nodes.csv"

QUAD_FILE = "quad_elements.csv"

TRIANGLE_FILE = "triangle_elements.csv"


# ------------------------------------------------------------
# UNIT CELL
# ------------------------------------------------------------

Lx = 1.0
Ly = 1.0
Lz = 1.0


# ------------------------------------------------------------
# PHYSICAL WALL THICKNESS
# ------------------------------------------------------------

THICKNESS = 0.04


# ------------------------------------------------------------
# NUMBER OF SOLID ELEMENT LAYERS THROUGH THICKNESS
# ------------------------------------------------------------

N_THICKNESS = 1


# ------------------------------------------------------------
# OPTIONAL LEVEL-SET PROJECTION
# ------------------------------------------------------------

PROJECT_TO_LEVEL_SET = False

PROJECTION_ITERATIONS = 5

PROJECTION_TOL = 1.0e-10

GRADIENT_TOL = 1.0e-12


# ------------------------------------------------------------
# UNIT CELL TOLERANCE
# ------------------------------------------------------------

BOUNDARY_TOL = 1.0e-10


# ------------------------------------------------------------
# JACOBIAN TOLERANCE
#
# We require:
#
#       det(J) > JACOBIAN_TOL
#
# rather than simply det(J) > 0,
# to avoid floating-point zero.
# ------------------------------------------------------------

JACOBIAN_TOL = 1.0e-14


# ------------------------------------------------------------
# STOP IF NODES ARE OUTSIDE THE UNIT CELL?
# ------------------------------------------------------------

STOP_IF_OUTSIDE = False


# ------------------------------------------------------------
# STOP IF INVALID JACOBIAN IS FOUND?
#
# Recommended:
#
# False during development.
#
# Later for batch FEA:
#
# True
#
# ------------------------------------------------------------

STOP_IF_INVALID_JACOBIAN = False


# ============================================================
# VISUALIZATION
# ============================================================

SHOW_SOLID_MESH = True


SURFACE_ALPHA = 0.95

EDGE_LINEWIDTH = 0.30


VIEW_ELEVATION = 25.0

VIEW_AZIMUTH = 45.0


# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_NODE_FILE = "solid_nodes.csv"

OUTPUT_HEX_FILE = "solid_hex8.csv"

OUTPUT_TET_FILE = "solid_tet4.csv"

OUTPUT_LAYER_FILE = "solid_node_layers.csv"

OUTPUT_INTERFACE_FILE = "quad_triangle_interfaces.csv"

OUTPUT_OUTSIDE_NODE_FILE = (
    "solid_nodes_outside_unitcell.csv"
)

OUTPUT_JACOBIAN_FILE = (
    "solid_jacobian_check.csv"
)


# ============================================================
# PRIMITIVE TPMS LEVEL SET
# ============================================================

def primitive_level_set(
        x,
        y,
        z):

    kx = (
        2.0 *
        np.pi /
        Lx
    )

    ky = (
        2.0 *
        np.pi /
        Ly
    )

    kz = (
        2.0 *
        np.pi /
        Lz
    )

    return (
        np.cos(kx * x)
        +
        np.cos(ky * y)
        +
        np.cos(kz * z)
    )


# ============================================================
# PRIMITIVE TPMS GRADIENT
# ============================================================

def primitive_gradient(
        x,
        y,
        z):

    kx = (
        2.0 *
        np.pi /
        Lx
    )

    ky = (
        2.0 *
        np.pi /
        Ly
    )

    kz = (
        2.0 *
        np.pi /
        Lz
    )

    gx = (
        -kx *
        np.sin(kx * x)
    )

    gy = (
        -ky *
        np.sin(ky * y)
    )

    gz = (
        -kz *
        np.sin(kz * z)
    )

    return np.array(
        [
            gx,
            gy,
            gz
        ],
        dtype=float
    )


# ============================================================
# READ NODE FILE
# ============================================================

def read_nodes(
        filename):

    if not os.path.exists(
            filename):

        raise FileNotFoundError(
            f"Node file not found:\n"
            f"{filename}"
        )

    data = np.loadtxt(
        filename,
        delimiter=",",
        skiprows=1
    )

    data = np.atleast_2d(
        data
    )

    node_ids = (
        data[:, 0]
        .astype(int)
    )

    nodes = (
        data[:, 1:4]
        .astype(float)
    )

    id_to_index = {}

    for index, node_id in enumerate(
            node_ids):

        node_id = int(
            node_id
        )

        if node_id in id_to_index:

            raise RuntimeError(
                f"Duplicate NodeID: "
                f"{node_id}"
            )

        id_to_index[
            node_id
        ] = index

    return (
        node_ids,
        nodes,
        id_to_index
    )


# ============================================================
# READ QUAD4 ELEMENTS
# ============================================================

def read_quads(
        filename):

    if not os.path.exists(
            filename):

        return (
            np.empty(
                0,
                dtype=int
            ),
            np.empty(
                (0, 4),
                dtype=int
            )
        )

    data = np.loadtxt(
        filename,
        delimiter=",",
        skiprows=1
    )

    data = np.atleast_2d(
        data
    )

    element_ids = (
        data[:, 0]
        .astype(int)
    )

    connectivity = (
        data[:, 1:5]
        .astype(int)
    )

    return (
        element_ids,
        connectivity
    )


# ============================================================
# READ TRI3 ELEMENTS
# ============================================================

def read_triangles(
        filename):

    if not os.path.exists(
            filename):

        return (
            np.empty(
                0,
                dtype=int
            ),
            np.empty(
                (0, 3),
                dtype=int
            )
        )

    data = np.loadtxt(
        filename,
        delimiter=",",
        skiprows=1
    )

    data = np.atleast_2d(
        data
    )

    element_ids = (
        data[:, 0]
        .astype(int)
    )

    connectivity = (
        data[:, 1:4]
        .astype(int)
    )

    return (
        element_ids,
        connectivity
    )


# ============================================================
# CONNECTIVITY CHECK
# ============================================================

def check_connectivity(
        node_ids,
        quads,
        triangles):

    valid_nodes = set(
        int(value)
        for value in node_ids
    )

    for e, quad in enumerate(
            quads):

        for node in quad:

            if int(node) not in valid_nodes:

                raise RuntimeError(
                    "Undefined node in QUAD4.\n"
                    f"Element row: {e + 1}\n"
                    f"NodeID: {node}"
                )

    for e, tri in enumerate(
            triangles):

        for node in tri:

            if int(node) not in valid_nodes:

                raise RuntimeError(
                    "Undefined node in TRI3.\n"
                    f"Element row: {e + 1}\n"
                    f"NodeID: {node}"
                )

    print(
        "Connectivity check: OK"
    )


# ============================================================
# ORIGINAL NODE IDS -> NUMPY INDICES
# ============================================================

def connectivity_to_indices(
        connectivity,
        id_to_index):

    if len(connectivity) == 0:

        return np.empty_like(
            connectivity
        )

    converted = np.zeros_like(
        connectivity,
        dtype=int
    )

    for e in range(
            len(connectivity)):

        for a in range(
                connectivity.shape[1]):

            node_id = int(
                connectivity[e, a]
            )

            converted[e, a] = (
                id_to_index[
                    node_id
                ]
            )

    return converted


# ============================================================
# OPTIONAL PROJECTION TO F = 0
# ============================================================

def project_nodes_to_level_set(
        nodes):

    projected = np.array(
        nodes,
        copy=True,
        dtype=float
    )

    for i in range(
            len(projected)):

        point = (
            projected[i]
            .copy()
        )

        for _ in range(
                PROJECTION_ITERATIONS):

            F = primitive_level_set(
                point[0],
                point[1],
                point[2]
            )

            if abs(F) < PROJECTION_TOL:

                break

            gradient = (
                primitive_gradient(
                    point[0],
                    point[1],
                    point[2]
                )
            )

            gradient_squared = (
                np.dot(
                    gradient,
                    gradient
                )
            )

            if (
                gradient_squared
                <
                GRADIENT_TOL
            ):

                break

            point = (
                point
                -
                F
                /
                gradient_squared
                *
                gradient
            )

        projected[i] = point

    return projected


# ============================================================
# ANALYTICAL TPMS NORMALS
# ============================================================

def calculate_normals(
        nodes):

    print(
        "\nCalculating analytical "
        "TPMS normals..."
    )

    normals = np.zeros_like(
        nodes
    )

    for i, point in enumerate(
            nodes):

        gradient = (
            primitive_gradient(
                point[0],
                point[1],
                point[2]
            )
        )

        magnitude = (
            np.linalg.norm(
                gradient
            )
        )

        if (
            magnitude
            <
            GRADIENT_TOL
        ):

            raise RuntimeError(
                f"Near-zero TPMS gradient "
                f"at node {i + 1}."
            )

        normals[i] = (
            gradient
            /
            magnitude
        )

    print(
        "Analytical normals: OK"
    )

    return normals


# ============================================================
# CREATE SOLID NODE LAYERS
# ============================================================

def create_solid_nodes(
        midsurface_nodes,
        normals):

    number_mid_nodes = (
        len(midsurface_nodes)
    )

    number_node_layers = (
        N_THICKNESS
        +
        1
    )

    offsets = np.linspace(
        -THICKNESS / 2.0,
        +THICKNESS / 2.0,
        number_node_layers
    )

    solid_nodes = []

    layer_data = []

    node_id = 1

    for layer in range(
            number_node_layers):

        offset = (
            offsets[layer]
        )

        for mid_index in range(
                number_mid_nodes):

            point = (
                midsurface_nodes[
                    mid_index
                ]
                +
                offset
                *
                normals[
                    mid_index
                ]
            )

            solid_nodes.append(
                point
            )

            layer_data.append(
                [
                    node_id,
                    layer,
                    mid_index + 1,
                    offset
                ]
            )

            node_id += 1

    return (
        np.asarray(
            solid_nodes,
            dtype=float
        ),
        np.asarray(
            layer_data,
            dtype=float
        )
    )


# ============================================================
# UNIT CELL LIMIT CHECK
# ============================================================

def check_unit_cell_limits(
        nodes):

    print()
    print(
        "=============================================="
    )
    print(
        " UNIT CELL LIMIT CHECK"
    )
    print(
        "=============================================="
    )

    x = nodes[:, 0]
    y = nodes[:, 1]
    z = nodes[:, 2]

    xmin = np.min(x)
    xmax = np.max(x)

    ymin = np.min(y)
    ymax = np.max(y)

    zmin = np.min(z)
    zmax = np.max(z)

    outside = (
        (x < -BOUNDARY_TOL)
        |
        (x > Lx + BOUNDARY_TOL)
        |
        (y < -BOUNDARY_TOL)
        |
        (y > Ly + BOUNDARY_TOL)
        |
        (z < -BOUNDARY_TOL)
        |
        (z > Lz + BOUNDARY_TOL)
    )

    outside_indices = (
        np.where(
            outside
        )[0]
    )

    print(
        f"Required X : [0, {Lx}]"
    )

    print(
        f"Actual X   : "
        f"[{xmin:.8e}, {xmax:.8e}]"
    )

    print()

    print(
        f"Required Y : [0, {Ly}]"
    )

    print(
        f"Actual Y   : "
        f"[{ymin:.8e}, {ymax:.8e}]"
    )

    print()

    print(
        f"Required Z : [0, {Lz}]"
    )

    print(
        f"Actual Z   : "
        f"[{zmin:.8e}, {zmax:.8e}]"
    )

    print()

    print(
        f"Nodes outside unit cell: "
        f"{len(outside_indices)}"
    )

    if len(outside_indices) == 0:

        print(
            "UNIT CELL CHECK: PASSED"
        )

        return True

    print(
        "UNIT CELL CHECK: FAILED"
    )

    outside_data = np.column_stack(
        (
            outside_indices + 1,
            nodes[
                outside_indices
            ]
        )
    )

    np.savetxt(
        OUTPUT_OUTSIDE_NODE_FILE,
        outside_data,
        delimiter=",",
        header="NodeID,X,Y,Z",
        comments="",
        fmt=[
            "%d",
            "%.15e",
            "%.15e",
            "%.15e"
        ]
    )

    return False


# ============================================================
# GLOBAL SOLID NODE ID
# ============================================================

def solid_node_id(
        layer,
        mid_index,
        number_mid_nodes):

    return (
        layer
        *
        number_mid_nodes
        +
        mid_index
        +
        1
    )


# ============================================================
# QUAD4 -> HEX8
# ============================================================

def generate_hex8(
        quads,
        number_mid_nodes):

    elements = []

    for layer in range(
            N_THICKNESS):

        for quad in quads:

            i1 = int(quad[0])
            i2 = int(quad[1])
            i3 = int(quad[2])
            i4 = int(quad[3])

            elements.append(
                [
                    solid_node_id(
                        layer,
                        i1,
                        number_mid_nodes
                    ),

                    solid_node_id(
                        layer,
                        i2,
                        number_mid_nodes
                    ),

                    solid_node_id(
                        layer,
                        i3,
                        number_mid_nodes
                    ),

                    solid_node_id(
                        layer,
                        i4,
                        number_mid_nodes
                    ),

                    solid_node_id(
                        layer + 1,
                        i1,
                        number_mid_nodes
                    ),

                    solid_node_id(
                        layer + 1,
                        i2,
                        number_mid_nodes
                    ),

                    solid_node_id(
                        layer + 1,
                        i3,
                        number_mid_nodes
                    ),

                    solid_node_id(
                        layer + 1,
                        i4,
                        number_mid_nodes
                    )
                ]
            )

    if len(elements) == 0:

        return np.empty(
            (0, 8),
            dtype=int
        )

    return np.asarray(
        elements,
        dtype=int
    )


# ============================================================
# TRI3 -> WEDGE -> 3 TET4
# ============================================================

def generate_tet4(
        triangles,
        number_mid_nodes):

    elements = []

    for layer in range(
            N_THICKNESS):

        for tri in triangles:

            i1 = int(tri[0])
            i2 = int(tri[1])
            i3 = int(tri[2])

            # Lower layer

            n1 = solid_node_id(
                layer,
                i1,
                number_mid_nodes
            )

            n2 = solid_node_id(
                layer,
                i2,
                number_mid_nodes
            )

            n3 = solid_node_id(
                layer,
                i3,
                number_mid_nodes
            )

            # Upper layer

            n4 = solid_node_id(
                layer + 1,
                i1,
                number_mid_nodes
            )

            n5 = solid_node_id(
                layer + 1,
                i2,
                number_mid_nodes
            )

            n6 = solid_node_id(
                layer + 1,
                i3,
                number_mid_nodes
            )

            # TET 1

            elements.append(
                [
                    n1,
                    n2,
                    n3,
                    n4
                ]
            )

            # TET 2

            elements.append(
                [
                    n2,
                    n3,
                    n4,
                    n5
                ]
            )

            # TET 3

            elements.append(
                [
                    n3,
                    n4,
                    n5,
                    n6
                ]
            )

    if len(elements) == 0:

        return np.empty(
            (0, 4),
            dtype=int
        )

    return np.asarray(
        elements,
        dtype=int
    )


# ============================================================
# TET4 SIGNED VOLUME / JACOBIAN
#
# For a linear tetrahedron:
#
# det(J) =
#
# det[
#     x2-x1  x3-x1  x4-x1
#     y2-y1  y3-y1  y4-y1
#     z2-z1  z3-z1  z4-z1
# ]
#
# ============================================================

def tet4_det_jacobian(
        xyz):

    p1 = xyz[0]
    p2 = xyz[1]
    p3 = xyz[2]
    p4 = xyz[3]

    J = np.column_stack(
        (
            p2 - p1,
            p3 - p1,
            p4 - p1
        )
    )

    return np.linalg.det(
        J
    )


# ============================================================
# ORIENT TET4 ELEMENTS
#
# If negative only because connectivity orientation is
# reversed, swap two nodes.
#
# ============================================================

def orient_tet4_elements(
        nodes,
        elements):

    corrected = np.array(
        elements,
        copy=True,
        dtype=int
    )

    reoriented = 0

    for e in range(
            len(corrected)):

        xyz = nodes[
            corrected[e] - 1
        ]

        detJ = (
            tet4_det_jacobian(
                xyz
            )
        )

        if detJ < 0.0:

            temp = (
                corrected[e, 1]
            )

            corrected[e, 1] = (
                corrected[e, 2]
            )

            corrected[e, 2] = temp

            reoriented += 1

    print(
        f"TET4 orientation corrections: "
        f"{reoriented}"
    )

    return corrected


# ============================================================
# HEX8 SHAPE FUNCTION DERIVATIVES
# ============================================================

def hex8_shape_derivatives(
        xi,
        eta,
        zeta):

    # Natural coordinates of the 8 standard HEX nodes

    local_coordinates = np.array(
        [
            [-1.0, -1.0, -1.0],
            [+1.0, -1.0, -1.0],
            [+1.0, +1.0, -1.0],
            [-1.0, +1.0, -1.0],

            [-1.0, -1.0, +1.0],
            [+1.0, -1.0, +1.0],
            [+1.0, +1.0, +1.0],
            [-1.0, +1.0, +1.0]
        ]
    )

    derivatives = np.zeros(
        (8, 3)
    )

    for i in range(8):

        xi_i = (
            local_coordinates[
                i,
                0
            ]
        )

        eta_i = (
            local_coordinates[
                i,
                1
            ]
        )

        zeta_i = (
            local_coordinates[
                i,
                2
            ]
        )

        derivatives[
            i,
            0
        ] = (
            0.125
            *
            xi_i
            *
            (1.0 + eta_i * eta)
            *
            (1.0 + zeta_i * zeta)
        )

        derivatives[
            i,
            1
        ] = (
            0.125
            *
            eta_i
            *
            (1.0 + xi_i * xi)
            *
            (1.0 + zeta_i * zeta)
        )

        derivatives[
            i,
            2
        ] = (
            0.125
            *
            zeta_i
            *
            (1.0 + xi_i * xi)
            *
            (1.0 + eta_i * eta)
        )

    return derivatives


# ============================================================
# HEX8 JACOBIAN MATRIX
# ============================================================

def hex8_jacobian(
        xyz,
        xi,
        eta,
        zeta):

    dN = (
        hex8_shape_derivatives(
            xi,
            eta,
            zeta
        )
    )

    J = (
        xyz.T
        @
        dN
    )

    return J


# ============================================================
# HEX8 JACOBIAN CHECK
#
# Eight 2x2x2 Gauss points.
# ============================================================

def check_hex8_jacobians(
        nodes,
        elements):

    print()
    print(
        "Checking HEX8 Jacobians..."
    )

    if len(elements) == 0:

        return (
            np.empty(0),
            np.empty(0),
            np.empty(
                0,
                dtype=bool
            )
        )

    g = (
        1.0
        /
        np.sqrt(3.0)
    )

    gauss_points = [
        (-g, -g, -g),
        (+g, -g, -g),
        (+g, +g, -g),
        (-g, +g, -g),

        (-g, -g, +g),
        (+g, -g, +g),
        (+g, +g, +g),
        (-g, +g, +g)
    ]

    minimum_detJ = np.zeros(
        len(elements)
    )

    maximum_detJ = np.zeros(
        len(elements)
    )

    valid = np.zeros(
        len(elements),
        dtype=bool
    )

    for e, element in enumerate(
            elements):

        xyz = nodes[
            element - 1
        ]

        determinants = []

        for (
            xi,
            eta,
            zeta
        ) in gauss_points:

            J = hex8_jacobian(
                xyz,
                xi,
                eta,
                zeta
            )

            determinants.append(
                np.linalg.det(
                    J
                )
            )

        determinants = np.asarray(
            determinants
        )

        minimum_detJ[e] = (
            np.min(
                determinants
            )
        )

        maximum_detJ[e] = (
            np.max(
                determinants
            )
        )

        valid[e] = (
            np.all(
                determinants
                >
                JACOBIAN_TOL
            )
        )

    return (
        minimum_detJ,
        maximum_detJ,
        valid
    )


# ============================================================
# TET4 JACOBIAN CHECK
# ============================================================

def check_tet4_jacobians(
        nodes,
        elements):

    print(
        "Checking TET4 Jacobians..."
    )

    if len(elements) == 0:

        return (
            np.empty(0),
            np.empty(
                0,
                dtype=bool
            )
        )

    detJ = np.zeros(
        len(elements)
    )

    valid = np.zeros(
        len(elements),
        dtype=bool
    )

    for e, element in enumerate(
            elements):

        xyz = nodes[
            element - 1
        ]

        detJ[e] = (
            tet4_det_jacobian(
                xyz
            )
        )

        valid[e] = (
            detJ[e]
            >
            JACOBIAN_TOL
        )

    return (
        detJ,
        valid
    )


# ============================================================
# PRINT JACOBIAN REPORT
# ============================================================

def print_jacobian_report(
        hex_min,
        hex_max,
        hex_valid,
        tet_det,
        tet_valid):

    print()
    print(
        "=============================================="
    )

    print(
        " JACOBIAN VALIDITY CHECK"
    )

    print(
        "=============================================="
    )

    # --------------------------------------------------------
    # HEX8
    # --------------------------------------------------------

    if len(hex_valid) > 0:

        number_hex_valid = (
            np.count_nonzero(
                hex_valid
            )
        )

        number_hex_invalid = (
            len(hex_valid)
            -
            number_hex_valid
        )

        print()
        print(
            "HEX8"
        )

        print(
            "----------------------------------------------"
        )

        print(
            f"Total elements       : "
            f"{len(hex_valid)}"
        )

        print(
            f"Positive Jacobian    : "
            f"{number_hex_valid}"
        )

        print(
            f"Invalid Jacobian     : "
            f"{number_hex_invalid}"
        )

        print(
            f"Minimum det(J)       : "
            f"{np.min(hex_min):.8e}"
        )

        print(
            f"Maximum det(J)       : "
            f"{np.max(hex_max):.8e}"
        )

    # --------------------------------------------------------
    # TET4
    # --------------------------------------------------------

    if len(tet_valid) > 0:

        number_tet_valid = (
            np.count_nonzero(
                tet_valid
            )
        )

        number_tet_invalid = (
            len(tet_valid)
            -
            number_tet_valid
        )

        print()
        print(
            "TET4"
        )

        print(
            "----------------------------------------------"
        )

        print(
            f"Total elements       : "
            f"{len(tet_valid)}"
        )

        print(
            f"Positive Jacobian    : "
            f"{number_tet_valid}"
        )

        print(
            f"Invalid Jacobian     : "
            f"{number_tet_invalid}"
        )

        print(
            f"Minimum det(J)       : "
            f"{np.min(tet_det):.8e}"
        )

        print(
            f"Maximum det(J)       : "
            f"{np.max(tet_det):.8e}"
        )

    # --------------------------------------------------------
    # Global result
    # --------------------------------------------------------

    all_hex_valid = (
        np.all(
            hex_valid
        )
        if len(hex_valid) > 0
        else True
    )

    all_tet_valid = (
        np.all(
            tet_valid
        )
        if len(tet_valid) > 0
        else True
    )

    overall_valid = (
        all_hex_valid
        and
        all_tet_valid
    )

    print()
    print(
        "----------------------------------------------"
    )

    if overall_valid:

        print(
            "JACOBIAN CHECK: PASSED"
        )

        print(
            "All HEX8 and TET4 elements "
            "have positive Jacobian determinants."
        )

    else:

        print(
            "JACOBIAN CHECK: FAILED"
        )

        print(
            "At least one solid element has "
            "a non-positive Jacobian determinant."
        )

    return overall_valid


# ============================================================
# EXPORT JACOBIAN DATA
# ============================================================

def export_jacobian_results(
        hex_min,
        hex_max,
        hex_valid,
        tet_det,
        tet_valid):

    rows = []

    global_id = 1

    # --------------------------------------------------------
    # HEX8
    #
    # Type = 1
    # --------------------------------------------------------

    for i in range(
            len(hex_valid)):

        rows.append(
            [
                global_id,
                1,
                hex_min[i],
                hex_max[i],
                int(
                    hex_valid[i]
                )
            ]
        )

        global_id += 1

    # --------------------------------------------------------
    # TET4
    #
    # Type = 2
    # --------------------------------------------------------

    for i in range(
            len(tet_valid)):

        rows.append(
            [
                global_id,
                2,
                tet_det[i],
                tet_det[i],
                int(
                    tet_valid[i]
                )
            ]
        )

        global_id += 1

    data = np.asarray(
        rows,
        dtype=float
    )

    np.savetxt(
        OUTPUT_JACOBIAN_FILE,
        data,
        delimiter=",",
        header=(
            "GlobalElementID,"
            "ElementType,"
            "MinDetJ,"
            "MaxDetJ,"
            "JacobianPositive"
        ),
        comments="",
        fmt=[
            "%d",
            "%d",
            "%.15e",
            "%.15e",
            "%d"
        ]
    )


# ============================================================
# EXPORT FEM NODES
# ============================================================

def export_nodes(
        filename,
        nodes):

    ids = np.arange(
        1,
        len(nodes) + 1
    )

    data = np.column_stack(
        (
            ids,
            nodes
        )
    )

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header="NodeID,X,Y,Z",
        comments="",
        fmt=[
            "%d",
            "%.15e",
            "%.15e",
            "%.15e"
        ]
    )


# ============================================================
# EXPORT HEX8
# ============================================================

def export_hex8(
        filename,
        elements):

    ids = np.arange(
        1,
        len(elements) + 1
    )

    data = np.column_stack(
        (
            ids,
            elements
        )
    )

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header=(
            "ElementID,"
            "N1,N2,N3,N4,"
            "N5,N6,N7,N8"
        ),
        comments="",
        fmt="%d"
    )


# ============================================================
# EXPORT TET4
# ============================================================

def export_tet4(
        filename,
        elements):

    ids = np.arange(
        1,
        len(elements) + 1
    )

    data = np.column_stack(
        (
            ids,
            elements
        )
    )

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header=(
            "ElementID,"
            "N1,N2,N3,N4"
        ),
        comments="",
        fmt="%d"
    )


# ============================================================
# EXPORT NODE LAYER DATA
# ============================================================

def export_layers(
        filename,
        data):

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header=(
            "SolidNodeID,"
            "Layer,"
            "MidsurfaceNode,"
            "NormalOffset"
        ),
        comments="",
        fmt=[
            "%d",
            "%d",
            "%d",
            "%.15e"
        ]
    )


# ============================================================
# ELEMENT FACES FOR VISUALIZATION
# ============================================================

def hex_faces(
        n):

    return [
        [n[0], n[1], n[2], n[3]],
        [n[4], n[5], n[6], n[7]],

        [n[0], n[1], n[5], n[4]],
        [n[1], n[2], n[6], n[5]],
        [n[2], n[3], n[7], n[6]],
        [n[3], n[0], n[4], n[7]]
    ]


def tet_faces(
        n):

    return [
        [n[0], n[1], n[2]],
        [n[0], n[1], n[3]],
        [n[1], n[2], n[3]],
        [n[2], n[0], n[3]]
    ]


# ============================================================
# EXTERNAL FACE EXTRACTION
# ============================================================

def extract_external_faces(
        hex_elements,
        tet_elements):

    database = {}

    def add_face(
            face):

        key = tuple(
            sorted(
                int(node)
                for node in face
            )
        )

        if key in database:

            database[
                key
            ]["count"] += 1

        else:

            database[
                key
            ] = {
                "count": 1,
                "face": list(face)
            }

    for element in hex_elements:

        for face in hex_faces(
                element):

            add_face(
                face
            )

    for element in tet_elements:

        for face in tet_faces(
                element):

            add_face(
                face
            )

    external = []

    for data in database.values():

        if data["count"] == 1:

            external.append(
                data["face"]
            )

    return external


# ============================================================
# VISUALIZATION
# ============================================================

def plot_solid_mesh(
        nodes,
        hex_elements,
        tet_elements):

    external_faces = (
        extract_external_faces(
            hex_elements,
            tet_elements
        )
    )

    polygons = []

    for face in external_faces:

        indices = (
            np.asarray(
                face,
                dtype=int
            )
            -
            1
        )

        polygons.append(
            nodes[
                indices
            ]
        )

    fig = plt.figure(
        figsize=(
            11,
            9
        )
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    collection = (
        Poly3DCollection(
            polygons,
            alpha=SURFACE_ALPHA,
            edgecolor="black",
            linewidth=EDGE_LINEWIDTH
        )
    )

    ax.add_collection3d(
        collection
    )

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
        "Primitive TPMS\n"
        "HEX8 + TET4 Solid FEM Mesh"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=============================================="
    )

    print(
        " TPMS SOLID FEM MESH GENERATOR"
    )

    print(
        "=============================================="
    )

    print()

    print(
        f"Unit cell               : "
        f"{Lx} x {Ly} x {Lz}"
    )

    print(
        f"Wall thickness          : "
        f"{THICKNESS}"
    )

    print(
        f"Elements through wall   : "
        f"{N_THICKNESS}"
    )

    print(
        f"Thickness per element   : "
        f"{THICKNESS/N_THICKNESS}"
    )

    # ========================================================
    # READ MIDSURFACE
    # ========================================================

    (
        node_ids,
        midsurface_nodes,
        id_to_index
    ) = read_nodes(
        NODE_FILE
    )

    (
        quad_ids,
        quads
    ) = read_quads(
        QUAD_FILE
    )

    (
        triangle_ids,
        triangles
    ) = read_triangles(
        TRIANGLE_FILE
    )

    print()
    print(
        f"Midsurface nodes : "
        f"{len(midsurface_nodes)}"
    )

    print(
        f"QUAD4 elements   : "
        f"{len(quads)}"
    )

    print(
        f"TRI3 elements    : "
        f"{len(triangles)}"
    )

    check_connectivity(
        node_ids,
        quads,
        triangles
    )

    # ========================================================
    # CONVERT TO ARRAY INDICES
    # ========================================================

    quad_indices = (
        connectivity_to_indices(
            quads,
            id_to_index
        )
    )

    triangle_indices = (
        connectivity_to_indices(
            triangles,
            id_to_index
        )
    )

    # ========================================================
    # OPTIONAL PROJECTION
    # ========================================================

    if PROJECT_TO_LEVEL_SET:

        midsurface_nodes = (
            project_nodes_to_level_set(
                midsurface_nodes
            )
        )

    # ========================================================
    # NORMALS
    # ========================================================

    normals = (
        calculate_normals(
            midsurface_nodes
        )
    )

    # ========================================================
    # SOLID NODES
    # ========================================================

    (
        solid_nodes,
        layer_data
    ) = create_solid_nodes(
        midsurface_nodes,
        normals
    )

    # ========================================================
    # UNIT CELL CHECK
    # ========================================================

    unit_cell_valid = (
        check_unit_cell_limits(
            solid_nodes
        )
    )

    if (
        not unit_cell_valid
        and
        STOP_IF_OUTSIDE
    ):

        raise RuntimeError(
            "Solid mesh exceeds "
            "unit-cell limits."
        )

    # ========================================================
    # ELEMENT CONNECTIVITY
    # ========================================================

    number_mid_nodes = (
        len(midsurface_nodes)
    )

    hex_elements = (
        generate_hex8(
            quad_indices,
            number_mid_nodes
        )
    )

    tet_elements = (
        generate_tet4(
            triangle_indices,
            number_mid_nodes
        )
    )

    # --------------------------------------------------------
    # Correct TET orientation
    # --------------------------------------------------------

    if len(tet_elements) > 0:

        tet_elements = (
            orient_tet4_elements(
                solid_nodes,
                tet_elements
            )
        )

    # ========================================================
    # MESH SUMMARY
    # ========================================================

    print()
    print(
        "=============================================="
    )

    print(
        " FINAL SOLID FEM MESH"
    )

    print(
        "=============================================="
    )

    print(
        f"Nodes : "
        f"{len(solid_nodes)}"
    )

    print(
        f"HEX8  : "
        f"{len(hex_elements)}"
    )

    print(
        f"TET4  : "
        f"{len(tet_elements)}"
    )

    # ========================================================
    # JACOBIAN CHECK
    # ========================================================

    (
        hex_min,
        hex_max,
        hex_valid
    ) = check_hex8_jacobians(
        solid_nodes,
        hex_elements
    )

    (
        tet_det,
        tet_valid
    ) = check_tet4_jacobians(
        solid_nodes,
        tet_elements
    )

    jacobian_valid = (
        print_jacobian_report(
            hex_min,
            hex_max,
            hex_valid,
            tet_det,
            tet_valid
        )
    )

    # --------------------------------------------------------
    # Export detailed Jacobian information
    # --------------------------------------------------------

    export_jacobian_results(
        hex_min,
        hex_max,
        hex_valid,
        tet_det,
        tet_valid
    )

    if (
        not jacobian_valid
        and
        STOP_IF_INVALID_JACOBIAN
    ):

        raise RuntimeError(
            "Mesh contains elements "
            "with invalid Jacobians."
        )

    # ========================================================
    # EXPORT FEM DATABASE
    # ========================================================

    print()
    print(
        "=============================================="
    )

    print(
        " EXPORTING FEM DATABASE"
    )

    print(
        "=============================================="
    )

    export_nodes(
        OUTPUT_NODE_FILE,
        solid_nodes
    )

    export_hex8(
        OUTPUT_HEX_FILE,
        hex_elements
    )

    export_tet4(
        OUTPUT_TET_FILE,
        tet_elements
    )

    export_layers(
        OUTPUT_LAYER_FILE,
        layer_data
    )

    print(
        OUTPUT_NODE_FILE
    )

    print(
        OUTPUT_HEX_FILE
    )

    print(
        OUTPUT_TET_FILE
    )

    print(
        OUTPUT_JACOBIAN_FILE
    )

    # ========================================================
    # FINAL RUN STATUS
    # ========================================================

    print()
    print(
        "=============================================="
    )

    print(
        " FINAL VALIDATION STATUS"
    )

    print(
        "=============================================="
    )

    if unit_cell_valid:

        print(
            "Unit-cell limits : PASSED"
        )

    else:

        print(
            "Unit-cell limits : FAILED"
        )

    if jacobian_valid:

        print(
            "Jacobians        : PASSED"
        )

    else:

        print(
            "Jacobians        : FAILED"
        )

    print()

    if (
        unit_cell_valid
        and
        jacobian_valid
    ):

        print(
            "MESH STATUS: READY FOR FEA"
        )

    elif jacobian_valid:

        print(
            "MESH STATUS: ELEMENTS VALID, "
            "BUT UNIT-CELL TRIMMING IS REQUIRED"
        )

    else:

        print(
            "MESH STATUS: NOT READY FOR FEA"
        )

    # ========================================================
    # VISUALIZATION — CHECK ONLY
    # ========================================================

    if SHOW_SOLID_MESH:

        plot_solid_mesh(
            solid_nodes,
            hex_elements,
            tet_elements
        )

    print()
    print(
        "Finished."
    )


# ============================================================
# EXECUTE
# ============================================================

if __name__ == "__main__":

    main()