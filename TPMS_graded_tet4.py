"""Starting-point TPMS mesh: smooth thickness gradient + linear TET4 only.

Miniforge dependencies:
    conda install -c conda-forge numpy matplotlib scikit-image gmsh python-gmsh
Run:
    python TPMS_graded_tet4.py
    python TPMS_graded_tet4.py --no-show

Two Primitive TPMS cells along x. Physical thickness is prescribed by normal
offsets of a shared midsurface. The final volume contains only four-node,
linear tetrahedra. One mesh-size parameter controls the free volume mesh;
there are no thickness layers or hexahedral conversions.

Boundary geometry is faceted. Its resolution is controlled separately by
POINTS_PER_CELL. This starting point does not include a structural solver,
periodic constraints, a global self-intersection test or FE convergence study.
"""
from pathlib import Path
from datetime import datetime
import argparse
import json
import numpy as np
from skimage import measure
import gmsh

# ----------------------- USER PARAMETERS -----------------------
CELL_SIZE = (1.0, 1.0, 1.0)
CELL_COUNTS = (2, 1, 1)
THICKNESS_AT_X_BOUNDARIES = (0.02, 0.1, 0.2)
LEVEL = 0.0
POINTS_PER_CELL = 36
ELEMENT_SIZE = 0.06  # desired tetrahedral edge scale; decrease for refinement
SHOW_PLOT = True
INITIAL_CUT_FRACTION = 0.5
DETAIL_ELEMENT_COUNT = 80
QUALITY_WARNING_THRESHOLD = 0.05
# Heuristic quality gates, not a substitute for FE convergence studies.
MIN_ACCEPTABLE_QUALITY = 0.04
MAX_LOW_QUALITY_FRACTION = 0.01
MAX_BOUNDARY_VOLUME_CHANGE = 0.2
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / 'tpms_tet4_output'
# Surface refinement respects existing facets, without changing thickness.
SURFACE_MAX_EDGE_FACTOR = 1.5
MAX_SURFACE_REFINEMENTS = 40
MAX_BOUNDARY_TRIANGLES = 500_000
# ---------------------------------------------------------------


def target_size(x):
    return np.full_like(np.asarray(x, dtype=float), ELEMENT_SIZE)


def validate_parameters():
    if min(CELL_SIZE) <= 0 or any(n < 1 or int(n) != n for n in CELL_COUNTS):
        raise ValueError('Cell dimensions must be positive; counts must be positive integers.')
    if len(THICKNESS_AT_X_BOUNDARIES) != CELL_COUNTS[0] + 1:
        raise ValueError('Supply one thickness value at each x-cell boundary.')
    if min(THICKNESS_AT_X_BOUNDARIES) <= 0 or ELEMENT_SIZE <= 0:
        raise ValueError('Thickness and element size must be positive.')
    if POINTS_PER_CELL < 4 or int(POINTS_PER_CELL) != POINTS_PER_CELL:
        raise ValueError('POINTS_PER_CELL must be an integer >= 4.')
    if not 0 <= INITIAL_CUT_FRACTION <= 1 or DETAIL_ELEMENT_COUNT < 1:
        raise ValueError('Check viewer parameters.')


def thickness(x):
    x = np.asarray(x)
    values = np.asarray(THICKNESS_AT_X_BOUNDARIES, float)
    q = np.clip(x / CELL_SIZE[0], 0, CELL_COUNTS[0])
    i = np.minimum(np.floor(q).astype(int), CELL_COUNTS[0] - 1)
    u = q - i
    s = u ** 3 * (10 - 15 * u + 6 * u * u)
    return values[i] + (values[i + 1] - values[i]) * s

def field(p):
    return np.cos(np.asarray(p) * (2 * np.pi / np.asarray(CELL_SIZE))).sum(axis=-1) - LEVEL

def gradient(p):
    k = 2 * np.pi / np.asarray(CELL_SIZE)
    return -k * np.sin(np.asarray(p) * k)

def midsurface():
    counts = np.asarray(CELL_COUNTS) * (POINTS_PER_CELL - 1) + 1
    lengths = np.asarray(CELL_SIZE) * CELL_COUNTS
    axes = [np.linspace(0, L, n) for L, n in zip(lengths, counts)]
    k = 2 * np.pi / np.asarray(CELL_SIZE)
    F = np.cos(k[0] * axes[0])[:, None, None] + np.cos(k[1] * axes[1])[None, :, None] + np.cos(k[2] * axes[2])[None, None, :]
    if not F.min() < LEVEL < F.max():
        raise ValueError('LEVEL lies outside the sampled scalar-field range.')
    p, f, _, _ = measure.marching_cubes(F, level=LEVEL, spacing=tuple(lengths / (counts - 1)), allow_degenerate=False)
    p = p.astype(float)
    planes = np.rint(p / np.asarray(CELL_SIZE)) * CELL_SIZE
    fixed = np.abs(p - planes) < 1e-08 * max(CELL_SIZE)
    p[fixed] = planes[fixed]
    for _ in range(25):
        residual = field(p)
        if np.max(np.abs(residual)) < 1e-11:
            break
        g = gradient(p)
        g[fixed] = 0
        den = (g * g).sum(axis=1)
        if np.any(den < 1e-20):
            raise RuntimeError('Near-zero projection gradient.')
        p -= residual[:, None] * g / den[:, None]
        p[fixed] = planes[fixed]
    if np.max(np.abs(field(p))) > 1e-08:
        raise RuntimeError('Midsurface projection failed.')
    xyz = p[f]
    reverse = (np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]) * gradient(xyz.mean(axis=1))).sum(axis=1) < 0
    f[reverse] = f[reverse][:, ::-1]
    return (p, f.astype(int))

def closed_boundary(p, f):
    g = gradient(p)
    g /= np.linalg.norm(g, axis=1)[:, None]
    t = thickness(p[:, 0])
    n = len(p)
    nodes = np.concatenate([p - 0.5 * t[:, None] * g, p + 0.5 * t[:, None] * g])
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    _, ids, counts = np.unique(np.sort(directed, axis=1), axis=0, return_index=True, return_counts=True)
    if np.any(counts > 2):
        raise RuntimeError('Nonmanifold midsurface.')
    a, b = directed[ids[counts == 1]].T
    faces = np.concatenate([f[:, ::-1], f + n, np.column_stack([a, b, b + n]), np.column_stack([a, b + n, a + n])])
    return (nodes, faces)

def refine_boundary(nodes, faces):
    """Shared-edge red/green refinement: no hanging nodes or surface gaps.

    New nodes lie on existing facets. Refinement controls mesh spacing, not
    underlying geometric accuracy; increase POINTS_PER_CELL for that.
    """
    for step in range(MAX_SURFACE_REFINEMENTS + 1):
        edges, inverse = np.unique(np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0, return_inverse=True)
        p, q = (nodes[edges[:, 0]], nodes[edges[:, 1]])
        h = np.minimum(np.minimum(target_size(p[:, 0]), target_size(q[:, 0])), target_size((p[:, 0] + q[:, 0]) / 2))
        split = np.linalg.norm(q - p, axis=1) > SURFACE_MAX_EDGE_FACTOR * h
        if not split.any():
            return (nodes, faces)
        if step == MAX_SURFACE_REFINEMENTS:
            raise RuntimeError('Boundary sizing did not converge; increase MAX_SURFACE_REFINEMENTS.')
        mids = np.full(len(edges), -1, int)
        mids[split] = np.arange(len(nodes), len(nodes) + split.sum())
        nodes = np.concatenate([nodes, (p[split] + q[split]) / 2])
        mid = mids[inverse.reshape(-1, 3)]
        result = []
        for row, m in zip(faces, mid):
            active = np.flatnonzero(m >= 0)
            count = len(active)
            if count == 0:
                result.append(row)
                continue
            if count == 3:
                a, b, c = row
                u, v, w = m
                result.extend([[a, u, w], [u, b, v], [w, v, c], [u, v, w]])
            elif count == 1:
                i = active[0]
                a, b, c = np.roll(row, -i)
                u = m[i]
                result.extend([[a, u, c], [u, b, c]])
            else:
                start = next((i for i in range(3) if m[i] >= 0 and m[(i + 1) % 3] >= 0))
                a, b, c = np.roll(row, -start)
                u = m[start]
                v = m[(start + 1) % 3]
                result.extend([[b, v, u], [a, u, c], [u, v, c]])
        faces = np.asarray(result, int)
        print(f'Boundary refinement {step + 1}: {len(faces):,} triangles', flush=True)
        if len(faces) > MAX_BOUNDARY_TRIANGLES:
            raise RuntimeError('Boundary exceeds MAX_BOUNDARY_TRIANGLES. Review mesh sizing.')

def check_boundary(nodes, faces):
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    signs = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    winding = np.bincount(inverse, weights=signs)
    if np.any(counts != 2) or np.any(winding != 0):
        raise RuntimeError('Boundary is not a closed, consistently oriented two-manifold.')
    p = nodes[faces]
    area2 = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    if np.any(area2 < 1e-16 * min(CELL_SIZE) ** 2):
        raise RuntimeError('Degenerate boundary facets.')
    vol = np.einsum('ij,ij->i', p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6
    if vol <= 0:
        raise RuntimeError('Nonpositive boundary volume.')
    lengths = np.asarray(CELL_SIZE) * CELL_COUNTS
    if np.any(nodes < -1e-08 * max(lengths)) or np.any(nodes > lengths + 1e-08 * max(lengths)):
        raise RuntimeError('Offsets leave specimen bounds; trimming is not implemented.')
    return float(vol)

def write_stl(path, nodes, faces):
    with open(path, 'w') as out:
        out.write('solid graded_tpms\n')
        for tri in nodes[faces]:
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            n /= np.linalg.norm(n)
            out.write('facet normal ' + ' '.join((f'{v:.12e}' for v in n)) + '\nouter loop\n')
            for p in tri:
                out.write('vertex ' + ' '.join((f'{v:.12e}' for v in p)) + '\n')
            out.write('endloop\nendfacet\n')
        out.write('endsolid graded_tpms\n')

TET_FACES = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])


def exposed_faces(tetrahedra):
    faces = tetrahedra[:, TET_FACES].reshape(-1, 3)
    if len(faces) == 0:
        return faces
    _, indices, counts = np.unique(np.sort(faces, axis=1), axis=0,
                                   return_index=True, return_counts=True)
    if np.any(counts > 2):
        raise RuntimeError('Nonmanifold tetrahedral faces.')
    return faces[indices[counts == 1]]


def mesh_volume(boundary_nodes, boundary_faces, output):
    """Fill the closed boundary directly with standard linear tetrahedra."""
    gmsh.initialize()
    try:
        gmsh.model.add('graded_tpms_tet4')
        # Replace the original skinny marching-cubes facets instead of forcing
        # the volume mesher to preserve them. The STL defines the geometry;
        # classified, parametrized patches receive a NEW surface mesh.
        gmsh.merge(str(output / 'graded_boundary.stl'))
        gmsh.model.mesh.classifySurfaces(np.deg2rad(40), True, True, np.pi)
        gmsh.model.mesh.createGeometry()
        surfaces = [tag for _, tag in gmsh.model.getEntities(2)]
        if not surfaces:
            raise RuntimeError('No parametrizable boundary patches.')
        shell = gmsh.model.geo.addSurfaceLoop(surfaces)
        volume = gmsh.model.geo.addVolume([shell])
        gmsh.model.geo.synchronize()
        for dim, entities, name in [(3, [volume], 'TPMS_SOLID'), (2, surfaces, 'EXTERIOR')]:
            group = gmsh.model.addPhysicalGroup(dim, entities)
            gmsh.model.setPhysicalName(dim, group, name)
        gmsh.option.setNumber('General.NumThreads', 1)
        gmsh.option.setNumber('Mesh.Algorithm3D', 1)
        gmsh.option.setNumber('Mesh.ElementOrder', 1)
        gmsh.option.setNumber('Mesh.Algorithm', 6)
        gmsh.option.setNumber('Mesh.MeshSizeMin', 0)
        gmsh.option.setNumber('Mesh.MeshSizeMax', ELEMENT_SIZE)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
        # Thin walls need a finer local mesh than the bulk-size upper bound.
        # This is a size target, not an imposed count of thickness layers.
        gmsh.model.mesh.setSizeCallback(
            lambda dim, tag, x, y, z, lc: min(ELEMENT_SIZE, float(thickness(x)) / 2))
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize('')
        try:
            gmsh.model.mesh.optimize('Netgen')
        except Exception as exc:
            raise RuntimeError('Netgen optimization failed or is unavailable in this Gmsh build.') from exc
        types, _, _ = gmsh.model.mesh.getElements(3)
        if list(types) != [4]:
            raise RuntimeError('Final volume must contain TET4 elements only.')
        element_tags, connectivity = gmsh.model.mesh.getElementsByType(4)
        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        node_tags = np.asarray(node_tags)
        order = np.argsort(node_tags)
        nodes = np.asarray(coordinates).reshape(-1, 3)[order]
        node_tags = node_tags[order]
        tetrahedra = np.searchsorted(node_tags, np.asarray(connectivity).reshape(-1, 4))
        _, face_connectivity = gmsh.model.mesh.getElementsByType(2)
        surface_triangles = np.searchsorted(node_tags, np.asarray(face_connectivity).reshape(-1, 3))
        quality = np.asarray(gmsh.model.mesh.getElementQualities(element_tags, 'minSICN'))
        gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)
        # Export only after validation in main(). Mesh files are written below
        # after all numerical and connectivity checks have passed.
        return nodes, tetrahedra, surface_triangles, quality, gmsh.__version__
    finally:
        gmsh.finalize()


def validate_mesh(nodes, tetrahedra, surface_triangles, quality, expected_volume, output):
    xyz = nodes[tetrahedra]
    jacobian = np.stack([xyz[:, i] - xyz[:, 0] for i in (1, 2, 3)], axis=2)
    determinant = np.linalg.det(jacobian)
    np.savetxt(output / 'element_checks.csv',
               np.column_stack([np.arange(1, len(tetrahedra) + 1), determinant, quality]),
               delimiter=',', header='ElementID,DetJ,ScaledQuality', comments='',
               fmt=['%d', '%.15e', '%.15e'])
    if not np.all(np.isfinite(determinant)) or np.any(determinant <= 0):
        raise RuntimeError('Nonpositive TET4 Jacobians; inspect element_checks.csv.')
    if not np.all(np.isfinite(quality)) or np.any(quality <= 0):
        raise RuntimeError('Nonpositive tetrahedral quality; inspect element_checks.csv.')
    if (quality.min() < MIN_ACCEPTABLE_QUALITY or
            np.mean(quality < QUALITY_WARNING_THRESHOLD) > MAX_LOW_QUALITY_FRACTION):
        raise RuntimeError('Mesh fails the configured shape-quality limits; inspect element_checks.csv. '
                           'No final volume mesh was exported.')
    exterior = exposed_faces(tetrahedra)
    a = np.unique(np.sort(exterior, axis=1), axis=0)
    b = np.unique(np.sort(surface_triangles, axis=1), axis=0)
    if not np.array_equal(a, b):
        raise RuntimeError('Volume/surface connectivity mismatch or internal gaps.')
    total_volume = determinant.sum() / 6
    p = nodes[exterior]
    meshed_boundary_volume = abs(np.einsum('ij,ij->i', p[:,0], np.cross(p[:,1],p[:,2])).sum() / 6)
    if not np.isclose(total_volume, meshed_boundary_volume, rtol=1e-7, atol=1e-12):
        raise RuntimeError('Tetrahedral volume does not match the remeshed closed boundary.')
    boundary_change = abs(meshed_boundary_volume - expected_volume) / expected_volume
    if boundary_change > MAX_BOUNDARY_VOLUME_CHANGE:
        raise RuntimeError('Surface remeshing changed volume beyond the configured limit. '
                           'Refine geometry sampling and/or ELEMENT_SIZE.')
    return exterior, dict(tet4_elements=len(tetrahedra), solid_nodes=len(nodes),
                          minimum_detJ=float(determinant.min()),
                          minimum_scaled_quality=float(quality.min()),
                          mean_scaled_quality=float(quality.mean()),
                          quality_percentiles=np.percentile(quality, [1,5,50,95]).tolist(),
                          elements_below_quality_threshold=int(np.sum(quality < QUALITY_WARNING_THRESHOLD)),
                          solid_volume=float(total_volume),
                          original_boundary_volume=float(expected_volume),
                          remeshed_boundary_volume=float(meshed_boundary_volume),
                          boundary_volume_change_fraction=float(boundary_change),
                          connectivity_verified=True)


def write_outputs(output, nodes, tetrahedra, boundary):
    np.savetxt(output / 'solid_nodes.csv',
               np.column_stack([np.arange(1, len(nodes) + 1), nodes]),
               delimiter=',', header='NodeID,X,Y,Z', comments='',
               fmt=['%d', '%.15e', '%.15e', '%.15e'])
    np.savetxt(output / 'tet4_elements.csv',
               np.column_stack([np.arange(1, len(tetrahedra) + 1), tetrahedra + 1]),
               delimiter=',', header='ElementID,N1,N2,N3,N4', comments='', fmt='%d')
    # Gmsh 2.2 ASCII: surface triangles type 2, linear tetrahedra type 4.
    with open(output / 'graded_tet4.msh', 'w') as f:
        f.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n2\n'
                '2 2 "EXTERIOR"\n3 1 "TPMS_SOLID"\n$EndPhysicalNames\n')
        f.write(f'$Nodes\n{len(nodes)}\n')
        for i, p in enumerate(nodes, 1):
            f.write(f'{i} ' + ' '.join(f'{x:.15e}' for x in p) + '\n')
        f.write(f'$EndNodes\n$Elements\n{len(boundary) + len(tetrahedra)}\n')
        index = 0
        for typ, physical, elements in [(2, 2, boundary), (4, 1, tetrahedra)]:
            for element in elements:
                index += 1
                f.write(f'{index} {typ} 2 {physical} 1 ' +
                        ' '.join(str(int(n) + 1) for n in element) + '\n')
        f.write('$EndElements\n')
    with open(output / 'graded_tet4.vtk', 'w') as f:
        f.write('# vtk DataFile Version 3.0\nGraded TPMS TET4\nASCII\nDATASET UNSTRUCTURED_GRID\n')
        f.write(f'POINTS {len(nodes)} double\n')
        np.savetxt(f, nodes, fmt='%.15e')
        f.write(f'CELLS {len(tetrahedra)} {len(tetrahedra) * 5}\n')
        np.savetxt(f, np.column_stack([np.full(len(tetrahedra), 4), tetrahedra]), fmt='%d')
        f.write(f'CELL_TYPES {len(tetrahedra)}\n')
        np.savetxt(f, np.full(len(tetrahedra), 10), fmt='%d')
    lengths = np.asarray(CELL_SIZE) * CELL_COUNTS
    for axis, label in enumerate('xyz'):
        for value, side in [(0, 'min'), (lengths[axis], 'max')]:
            ids = np.flatnonzero(np.abs(nodes[:, axis] - value) < 1e-8 * max(lengths)) + 1
            np.savetxt(output / f'boundary_{label}{side}.csv', ids,
                       header='NodeID', comments='', fmt='%d')


def preview(output, nodes, tetrahedra, boundary, show):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(18, 6))
    axes = [fig.add_subplot(131 + i, projection='3d') for i in range(3)]
    lengths = np.asarray(CELL_SIZE) * CELL_COUNTS
    for ax in axes[:2]:
        ax.set(xlim=(0, lengths[0]), ylim=(0, lengths[1]), zlim=(0, lengths[2]))
        ax.set_box_aspect(lengths.copy())
    for ax in axes:
        ax.set(xlabel='x', ylabel='y', zlabel='z')
        ax.view_init(24, -60)
    def draw(ax, faces, thickness_colors=True):
        collection = Poly3DCollection(nodes[faces], edgecolors='#263746', linewidths=0.2)
        if thickness_colors:
            collection.set_array(thickness(nodes[faces].mean(axis=1)[:, 0]))
            collection.set_cmap('viridis')
            collection.set_clim(min(THICKNESS_AT_X_BOUNDARIES), max(THICKNESS_AT_X_BOUNDARIES))
        else:
            collection.set_facecolor('#6bb4db')
        ax.add_collection3d(collection)
        return collection
    full = draw(axes[0], boundary)
    axes[0].set_title(f'TET4 solid mesh: {len(tetrahedra):,} elements')
    fig.colorbar(full, ax=axes[0], shrink=0.5, label='Wall thickness')
    centres = nodes[tetrahedra].mean(axis=1)
    artist = [None]
    def update(value):
        selected = tetrahedra[centres[:, 1] <= value]
        if artist[0] is not None:
            artist[0].remove()
        artist[0] = draw(axes[1], exposed_faces(selected))
        axes[1].set_title(f'Whole-element cutaway: y <= {value:.3f}')
        fig.canvas.draw_idle()
    # A small solid-element patch makes the mesh visible even for dense models.
    distance = np.linalg.norm(centres - lengths / 2, axis=1)
    anchor = centres[np.argmin(distance)]
    ids = np.argsort(np.linalg.norm(centres - anchor, axis=1))[:DETAIL_ELEMENT_COUNT]
    detail = tetrahedra[ids]
    detail_faces = detail[:, TET_FACES].reshape(-1, 3)
    draw(axes[2], detail_faces, False)
    points = nodes[np.unique(detail)]
    lo, hi = points.min(axis=0), points.max(axis=0)
    padding = np.maximum((hi - lo) * 0.08, 1e-6)
    axes[2].set(xlim=(lo[0] - padding[0], hi[0] + padding[0]),
                ylim=(lo[1] - padding[1], hi[1] + padding[1]),
                zlim=(lo[2] - padding[2], hi[2] + padding[2]))
    axes[2].set_box_aspect((hi - lo).copy())
    axes[2].set_title(f'Local mesh detail: {len(detail)} TET4')
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.17, top=0.88, wspace=0.12)
    slider = Slider(fig.add_axes([0.43, 0.075, 0.20, 0.025]), 'Cut y', 0,
                    float(lengths[1]), valinit=INITIAL_CUT_FRACTION * lengths[1])
    slider.on_changed(update)
    update(slider.val)
    fig.savefig(output / 'tet4_mesh_preview.png', dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--output', type=Path, default=OUTPUT_DIRECTORY)
    args = parser.parse_args()
    validate_parameters()
    output = args.output.resolve() / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    print('Output:', output, flush=True)
    points, triangles = midsurface()
    boundary_nodes, boundary_faces = closed_boundary(points, triangles)
    boundary_nodes, boundary_faces = refine_boundary(boundary_nodes, boundary_faces)
    boundary_volume = check_boundary(boundary_nodes, boundary_faces)
    write_stl(output / 'graded_boundary.stl', boundary_nodes, boundary_faces)
    nodes, tetrahedra, surface, quality, version = mesh_volume(boundary_nodes, boundary_faces, output)
    exterior, report = validate_mesh(nodes, tetrahedra, surface, quality, boundary_volume, output)
    write_outputs(output, nodes, tetrahedra, exterior)
    report.update(cell_size=list(CELL_SIZE), cell_counts=list(CELL_COUNTS),
                  thickness=list(THICKNESS_AT_X_BOUNDARIES), element_size=ELEMENT_SIZE,
                  gmsh_version=version, element_type='TET4: 4-node linear tetrahedron',
                  limitations=['No global boundary self-intersection test.',
                               'No guaranteed integer count through the wall.',
                               'Starting-point mesh; inspect quality and establish FE convergence.'])
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if report['elements_below_quality_threshold']:
        print('Some elements have low shape quality; details are in element_checks.csv.', flush=True)
    preview(output, nodes, tetrahedra, exterior, SHOW_PLOT and not args.no_show)
    print('Finished: linear TET4 only.', flush=True)


if __name__ == '__main__':
    main()
