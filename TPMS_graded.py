"""Primitive TPMS: two cells, graded thickness, shell and conforming solid mesh.

Run in the Miniforge environment used by VS Code:
    conda install -c conda-forge numpy matplotlib scikit-image gmsh python-gmsh
    python TPMS_graded.py
    python TPMS_graded.py --no-show
    python TPMS_graded.py --no-remesh --no-show

Lengths use one consistent user-selected unit. Defaults are dimensionless;
material density below is kg/m^3 only if lengths are interpreted in metres.
This is a mesh generator, not a structural solver or SALOME script.

Merged from TPMSgen/TPMSmesh/TPMSquad/TPMSsolid. TRI3 patches extrude to
WEDGE6 (not TET4), preserving compatible quadrilateral interfaces with HEX8.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from skimage import measure

# ===================== USER PARAMETERS =====================
CELL_SIZE = (1.0, 1.0, 1.0)  # periodic wavelengths, NOT specimen dimensions
CELL_COUNTS = (2, 1, 1)
POINTS_PER_CELL = 60          # intervals = points - 1; shared cell planes
LEVEL = 0.0
# One value at each x-cell boundary: x=0, a, 2a for two cells.
THICKNESS_AT_X_BOUNDARIES = (0.02, 0.1, 0.2)
N_THICKNESS = 4             # solid elements through the wall
TARGET_ELEMENT_SIZE = 0.1
USE_GMSH_REMESH = True
AUTO_WEDGE_FALLBACK = True  # retry original surface if remeshed solids fail
SHOW_PLOT = True
INITIAL_CUT_FRACTION = 0.5  # viewer: retain whole elements below this y fraction
DENSITY = 7850.0            # used only for approximate shell mass
YOUNGS_MODULUS = 210e9      # metadata only; no FE solve
POISSON_RATIO = 0.30
# Distinct outputs prevent interference with previous four scripts.
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / 'tpms_graded_output'
# ===========================================================


def field(points):
    """Trigonometric approximation to the Primitive minimal surface."""
    k = 2 * np.pi / np.asarray(CELL_SIZE)
    return np.cos(np.asarray(points) * k).sum(axis=-1) - LEVEL


def gradient(points):
    k = 2 * np.pi / np.asarray(CELL_SIZE)
    return -k * np.sin(np.asarray(points) * k)


def thickness(points):
    """C2 piecewise-quintic field; zero slope/curvature at x-cell planes."""
    values = np.asarray(THICKNESS_AT_X_BOUNDARIES, dtype=float)
    q = np.clip(np.asarray(points)[..., 0] / CELL_SIZE[0], 0, CELL_COUNTS[0])
    i = np.minimum(np.floor(q).astype(int), CELL_COUNTS[0] - 1)
    u = q - i
    smooth = u**3 * (10 - 15*u + 6*u*u)
    return values[i] + (values[i + 1] - values[i]) * smooth


def parameters_check():
    if any(n < 1 or int(n) != n for n in CELL_COUNTS):
        raise ValueError('CELL_COUNTS must contain positive integers.')
    if min(CELL_SIZE) <= 0 or POINTS_PER_CELL < 4 or N_THICKNESS < 1:
        raise ValueError('Check positive cell sizes, sampling and layer count.')
    if int(N_THICKNESS) != N_THICKNESS:
        raise ValueError('N_THICKNESS must be an integer.')
    if len(THICKNESS_AT_X_BOUNDARIES) != CELL_COUNTS[0] + 1:
        raise ValueError('Supply one thickness per x-cell boundary.')
    if min(THICKNESS_AT_X_BOUNDARIES) <= 0 or TARGET_ELEMENT_SIZE <= 0:
        raise ValueError('Thickness and target mesh size must be positive.')


def extract_midsurface():
    counts = np.asarray(CELL_COUNTS) * (POINTS_PER_CELL - 1) + 1
    lengths = np.asarray(CELL_SIZE) * CELL_COUNTS
    axes = [np.linspace(0, L, n) for L, n in zip(lengths, counts)]
    # Sparse broadcasting avoids allocating three full coordinate grids.
    k = 2*np.pi / np.asarray(CELL_SIZE)
    F = (np.cos(k[0]*axes[0])[:, None, None]
         + np.cos(k[1]*axes[1])[None, :, None]
         + np.cos(k[2]*axes[2])[None, None, :])
    if not F.min() < LEVEL < F.max():
        raise ValueError('LEVEL must lie strictly inside the sampled field range.')
    nodes, tris, _, _ = measure.marching_cubes(
        F, level=LEVEL, spacing=tuple(lengths/(counts-1)),
        allow_degenerate=False)
    return nodes.astype(float), tris.astype(int)


def write_stl(path, nodes, triangles, name='tpms'):
    with open(path, 'w') as out:
        out.write(f'solid {name}\n')
        for tri in triangles:
            p = nodes[tri]
            n = np.cross(p[1]-p[0], p[2]-p[0])
            norm = np.linalg.norm(n)
            if norm == 0:
                raise RuntimeError('Degenerate STL triangle.')
            n /= norm
            out.write('  facet normal ' + ' '.join(f'{v:.15e}' for v in n) + '\n    outer loop\n')
            for v in p:
                out.write('      vertex ' + ' '.join(f'{x:.15e}' for x in v) + '\n')
            out.write('    endloop\n  endfacet\n')
        out.write(f'endsolid {name}\n')


def remesh(stl_path, output):
    try:
        import gmsh
    except (ImportError, OSError) as exc:
        raise RuntimeError('Gmsh unavailable. Install python-gmsh in Miniforge '
                           'or run with --no-remesh.') from exc
    gmsh.initialize()
    try:
        gmsh.model.add('graded_tpms_midsurface')
        gmsh.merge(str(stl_path))
        gmsh.model.mesh.classifySurfaces(np.deg2rad(40), True, True, np.pi)
        gmsh.model.mesh.createGeometry()
        for key, value in {
            'Mesh.MeshSizeMin': TARGET_ELEMENT_SIZE,
            'Mesh.MeshSizeMax': TARGET_ELEMENT_SIZE,
            'Mesh.MeshSizeFromPoints': 0,
            'Mesh.MeshSizeFromCurvature': 0,
            'Mesh.MeshSizeExtendFromBoundary': 0,
            'Mesh.Algorithm': 6, 'Mesh.RecombineAll': 1,
            'Mesh.RecombinationAlgorithm': 1, 'Mesh.Smoothing': 10,
        }.items():
            gmsh.option.setNumber(key, value)
        for dim, tag in gmsh.model.getEntities(2):
            gmsh.model.mesh.setRecombine(dim, tag)
        gmsh.model.mesh.generate(2)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        lookup = {int(tag): i for i, tag in enumerate(tags)}
        nodes = np.asarray(xyz).reshape(-1, 3)
        types, _, connections = gmsh.model.mesh.getElements(2)
        quads, triangles = [], []
        for typ, conn in zip(types, connections):
            if typ not in (2, 3):
                raise RuntimeError(f'Unsupported Gmsh surface element type {typ}.')
            width = 3 if typ == 2 else 4
            dest = triangles if typ == 2 else quads
            dest.extend([[lookup[int(t)] for t in row]
                         for row in np.asarray(conn).reshape(-1, width)])
        return nodes, np.asarray(quads, int).reshape(-1, 4), np.asarray(triangles, int).reshape(-1, 3)
    finally:
        gmsh.finalize()


def project_and_orient(nodes, quads, triangles):
    """Project to analytic surface while preserving all integer cell planes."""
    nodes = nodes.copy()
    a = np.asarray(CELL_SIZE)
    # At integer cell planes the corresponding gradient component is zero.
    planes = np.rint(nodes/a)*a
    fixed = np.abs(nodes-planes) < 1e-8*max(a)
    nodes[fixed] = planes[fixed]
    for _ in range(20):
        f = field(nodes)
        if np.max(np.abs(f)) < 1e-11:
            break
        g = gradient(nodes)
        g[fixed] = 0
        den = (g*g).sum(axis=1)
        if np.any(den < 1e-20):
            raise RuntimeError('Projection encountered a near-zero free gradient.')
        nodes -= f[:, None]*g/den[:, None]
        nodes[fixed] = planes[fixed]
    if np.max(np.abs(field(nodes))) > 1e-8:
        raise RuntimeError('Midsurface projection did not converge.')
    for elements in (quads, triangles):
        if len(elements):
            p = nodes[elements]
            n = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
            reverse = (n*gradient(p.mean(axis=1))).sum(axis=1) < 0
            elements[reverse] = elements[reverse][:, ::-1]
    return nodes, quads, triangles


def surface_statistics(nodes, quads, triangles):
    tri = np.concatenate((triangles, quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]))
    p = nodes[tri]
    areas = np.linalg.norm(np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]), axis=1)/2
    if np.any(areas < 1e-14*min(CELL_SIZE)**2):
        raise RuntimeError('Degenerate midsurface triangles.')
    edges = np.sort(np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(counts > 2):
        raise RuntimeError('Nonmanifold midsurface edges.')
    approximate_volume = np.sum(areas*thickness(p.mean(axis=1)))
    return tri, dict(surface_area=float(areas.sum()),
                     approximate_shell_volume=float(approximate_volume),
                     approximate_shell_mass=float(approximate_volume*DENSITY),
                     midsurface_open_edges=int(np.sum(counts == 1)))


def extrude(nodes, quads, triangles):
    g = gradient(nodes)
    normals = g/np.linalg.norm(g, axis=1)[:, None]
    t = thickness(nodes)
    fractions = np.linspace(-0.5, 0.5, N_THICKNESS+1)
    solid = np.concatenate([nodes+f*t[:, None]*normals for f in fractions])
    n = len(nodes)
    def stack(elements):
        width = elements.shape[1]
        return np.concatenate([np.column_stack((elements+k*n, elements+(k+1)*n))
                               for k in range(N_THICKNESS)]).reshape(-1, 2*width)
    return solid, stack(quads), stack(triangles), normals, t


def derivatives_hex(r, s, z):
    signs = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                      [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
    u, v, w = signs.T
    return np.column_stack((u*(1+v*s)*(1+w*z),
                            v*(1+u*r)*(1+w*z), w*(1+u*r)*(1+v*s)))/8


def derivatives_wedge(r, s, z):
    b = np.array([1-r-s, r, s])
    dr, ds = np.array([-1,1,0]), np.array([-1,0,1])
    return np.vstack((np.column_stack((dr*(1-z)/2, ds*(1-z)/2, -b/2)),
                      np.column_stack((dr*(1+z)/2, ds*(1+z)/2, b/2))))


class InvalidSolidMesh(RuntimeError):
    """A mesh failed the sampled solid-element Jacobian checks."""


def validate_jacobians(nodes, hexes, wedges, output):
    # Gauss points AND vertices/centre. This is sampled validation, not a
    # mathematical guarantee of positivity everywhere or no self-intersection.
    from itertools import product
    g = 1/np.sqrt(3)
    hp = list(product((-g,g), repeat=3)) + list(product((-1,1), repeat=3)) + [(0,0,0)]
    wp = [(r,s,z) for r,s in [(1/6,1/6),(2/3,1/6),(1/6,2/3),
                              (0,0),(1,0),(0,1),(1/3,1/3)] for z in [-1,-g,0,g,1]]
    rows, volumes = [], []
    tolerance = 1e-12*min(CELL_SIZE)**3
    for name, elements, points, derivative in [('HEX8',hexes,hp,derivatives_hex),
                                              ('WEDGE6',wedges,wp,derivatives_wedge)]:
        if not len(elements):
            continue
        xyz = nodes[elements]
        dets = np.array([np.linalg.det(np.einsum('eai,aj->eij', xyz, derivative(*pt)))
                         for pt in points])
        for i in range(len(elements)):
            rows.append([name, i+1, float(dets[:,i].min()), float(dets[:,i].max())])
        # Exact quadrature for these linear isoparametric element volumes.
        gauss = list(product((-g,g), repeat=3)) if name=='HEX8' else [
            (r,s,z) for r,s in [(1/6,1/6),(2/3,1/6),(1/6,2/3)] for z in [-g,g]]
        weight = 1 if name=='HEX8' else 1/6
        volumes.append(sum(np.linalg.det(np.einsum('eai,aj->eij',xyz,derivative(*p))).sum()*weight
                           for p in gauss))
    import csv
    with open(output/'solid_jacobians.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['Type','ElementID','MinDetJ','MaxDetJ']);w.writerows(rows)
    bad = [row for row in rows if not np.isfinite(row[2]) or row[2] <= tolerance]
    if not rows or bad:
        counts = {name: sum(row[0] == name for row in bad) for name in ('HEX8','WEDGE6')}
        raise InvalidSolidMesh(
            f'Jacobian check failed: {counts["HEX8"]} HEX8 and '
            f'{counts["WEDGE6"]} WEDGE6 invalid/near-degenerate. '
            f'Details: {output / "solid_jacobians.csv"}.')
    return float(min(row[2] for row in rows)), float(sum(volumes))


HEX_FACES = [(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
WEDGE_FACES = [(0,2,1),(3,4,5),(0,1,4,3),(1,2,5,4),(2,0,3,5)]


def external_faces(nodes, hexes, wedges):
    database = {}
    for elements, faces in [(hexes,HEX_FACES),(wedges,WEDGE_FACES)]:
        for element in elements:
            centre = nodes[element].mean(axis=0)
            for local in faces:
                ids = element[list(local)]
                key = tuple(sorted(ids))
                if key in database:
                    database[key][0] += 1
                else:
                    p = nodes[ids]
                    if np.dot(np.cross(p[1]-p[0], p[2]-p[0]), p.mean(axis=0)-centre) < 0:
                        ids = ids[::-1]
                    database[key] = [1, ids]
    if any(count > 2 for count, _ in database.values()):
        raise RuntimeError('Nonmanifold solid faces.')
    boundary = [ids for count, ids in database.values() if count == 1]
    triangles = []
    for face in boundary:
        triangles.append(face[[0,1,2]])
        if len(face)==4:
            triangles.append(face[[0,2,3]])
    triangles = np.asarray(triangles)
    edges = np.sort(np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]]),axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(counts != 2):
        raise RuntimeError('Solid exterior is not watertight; check mesh connectivity.')
    return boundary, triangles


def csv_array(path, array, header, integer=False):
    np.savetxt(path, array, delimiter=',', header=header, comments='',
               fmt='%d' if integer else '%.15g')


def write_msh(path, nodes, hexes, wedges):
    """Portable ASCII Gmsh 2.2 mesh; types 5=HEX8 and 6=WEDGE6."""
    with open(path,'w') as f:
        f.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n'+str(len(nodes))+'\n')
        for i,p in enumerate(nodes,1):
            f.write(f'{i} '+ ' '.join(f'{x:.15g}' for x in p)+'\n')
        f.write('$EndNodes\n$Elements\n'+str(len(hexes)+len(wedges))+'\n')
        i=0
        for typ, elements in [(5,hexes),(6,wedges)]:
            for row in elements:
                i+=1;f.write(f'{i} {typ} 2 1 1 '+' '.join(str(int(x)+1) for x in row)+'\n')
        f.write('$EndElements\n')


def preview(output, nodes, hexes, wedges, boundary, show):
    """View actual solid-element faces, plus an adjustable element cutaway.

    The right pane removes whole elements according to centroid y; it is
    not a geometrically clipped cross-section. Newly exposed faces show
    the elements and layers through the wall.
    """
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.widgets import Slider
    from matplotlib.patches import Patch
    fig=plt.figure(figsize=(15,7))
    left=fig.add_subplot(121,projection='3d')
    right=fig.add_subplot(122,projection='3d')
    lengths=np.asarray(CELL_SIZE)*CELL_COUNTS
    for ax in (left,right):
        ax.set(xlim=(0,lengths[0]),ylim=(0,lengths[1]),zlim=(0,lengths[2]),
               xlabel='x',ylabel='y',zlabel='z')
        ax.set_box_aspect(lengths.copy());ax.view_init(24,-60)
    # Physical outer faces of the actual solid mesh, including wall end faces.
    collection=Poly3DCollection([nodes[f] for f in boundary],
                                linewidths=0.25,edgecolors='#243447')
    values=np.array([thickness(nodes[f].mean(axis=0)) for f in boundary])
    collection.set_array(values);collection.set_cmap('viridis')
    collection.set_clim(min(THICKNESS_AT_X_BOUNDARIES),max(THICKNESS_AT_X_BOUNDARIES))
    left.add_collection3d(collection)
    left.set_title(f'Solid mesh: {len(hexes):,} HEX8 + {len(wedges):,} WEDGE6')
    fig.colorbar(collection,ax=left,shrink=0.55,pad=0.05,label='Local prescribed thickness')
    colors={'HEX8':'#69b4dc','WEDGE6':'#f3a34f'}
    centres=[nodes[e].mean(axis=1)[:,1] if len(e) else np.empty(0)
             for e in (hexes,wedges)]
    cut_artist=[None]

    def draw_cut(cut):
        faces={}; retained=[]
        for name,elements,local_faces,cy in [('HEX8',hexes,HEX_FACES,centres[0]),
                                           ('WEDGE6',wedges,WEDGE_FACES,centres[1])]:
            selected=elements[cy<=cut]
            retained.append(len(selected))
            for element in selected:
                for local in local_faces:
                    ids=element[list(local)];key=tuple(sorted(ids))
                    if key in faces:faces[key][0]+=1
                    else:faces[key]=[1,ids,name]
        outer=[(ids,name) for count,ids,name in faces.values() if count==1]
        if cut_artist[0] is not None:cut_artist[0].remove()
        artist=Poly3DCollection([nodes[ids] for ids,_ in outer],
                 facecolors=[colors[name] for _,name in outer],
                 edgecolors='#243447',linewidths=0.4)
        right.add_collection3d(artist);cut_artist[0]=artist
        right.set_title(f'Element cutaway: centroid y <= {cut:.3f}\n'
                        f'{retained[0]:,} HEX8 + {retained[1]:,} WEDGE6 retained')
        fig.canvas.draw_idle()

    right.legend(handles=[Patch(facecolor=c,edgecolor='#243447',label=n)
                          for n,c in colors.items()],loc='upper right')
    fig.subplots_adjust(left=0.02,right=0.98,top=0.90,bottom=0.17,wspace=0.08)
    slider_axis=fig.add_axes([0.59,0.08,0.30,0.025])
    slider=Slider(slider_axis,'Cut y',0,float(lengths[1]),
                  valinit=float(INITIAL_CUT_FRACTION*lengths[1]))
    slider.on_changed(draw_cut)
    draw_cut(slider.val)
    fig.suptitle(f'Graded Primitive TPMS — {N_THICKNESS} solid elements through thickness',fontsize=14)
    fig.text(0.5,0.02,'Drag either view to rotate. Move Cut y to expose solid elements; '
             'whole elements are removed, not clipped.',ha='center',fontsize=10)
    fig.savefig(output/'solid_mesh_preview.png',dpi=180)
    if show:
        print('Solid viewer: drag to rotate; move Cut y slider to expose wall layers.')
        plt.show()
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--no-remesh',action='store_true',help='Extrude original triangles into WEDGE6.')
    parser.add_argument('--no-show',action='store_true',help='Save preview without opening a window.')
    parser.add_argument('--output',type=Path,default=OUTPUT_DIRECTORY)
    args=parser.parse_args();parameters_check()
    # A fresh run folder prevents stale element files from earlier runs.
    from datetime import datetime
    output=args.output.resolve()/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    print('Output:',output)
    nodes, triangles=extract_midsurface()
    raw_nodes,raw_triangles=nodes.copy(),triangles.copy()
    write_stl(output/'raw_midsurface.stl',nodes,triangles)
    quads=np.empty((0,4),int)
    remeshed=USE_GMSH_REMESH and not args.no_remesh
    fallback_used=False
    rejected_mesh_reason=None
    if remeshed:
        nodes,quads,triangles=remesh(output/'raw_midsurface.stl',output)
    nodes,quads,triangles=project_and_orient(nodes,quads,triangles)
    surface_triangles,report=surface_statistics(nodes,quads,triangles)
    solid,hexes,wedges,normals,t=extrude(nodes,quads,triangles)
    lengths=np.asarray(CELL_SIZE)*CELL_COUNTS
    if np.any(solid < -1e-8*max(lengths)) or np.any(solid > lengths+1e-8*max(lengths)):
        raise RuntimeError('Offsets leave specimen bounds; trimming is not implemented.')
    try:
        min_j,volume=validate_jacobians(solid,hexes,wedges,output)
    except InvalidSolidMesh as exc:
        if not (remeshed and AUTO_WEDGE_FALLBACK):
            raise
        rejected_mesh_reason=str(exc)
        print(rejected_mesh_reason)
        (output/'solid_jacobians.csv').rename(output/'rejected_gmsh_jacobians.csv')
        print('Retrying the original marching-cubes surface with WEDGE6 solid elements. '
              'Cell dimensions and thickness values are unchanged.')
        nodes,quads,triangles=project_and_orient(
            raw_nodes,np.empty((0,4),int),raw_triangles)
        surface_triangles,report=surface_statistics(nodes,quads,triangles)
        solid,hexes,wedges,normals,t=extrude(nodes,quads,triangles)
        if np.any(solid < -1e-8*max(lengths)) or np.any(solid > lengths+1e-8*max(lengths)):
            raise RuntimeError('Fallback offsets leave specimen bounds; trimming is not implemented.')
        # Never bypass validation: the fallback must independently pass.
        min_j,volume=validate_jacobians(solid,hexes,wedges,output)
        fallback_used=True
        print('WEDGE6 fallback passed the sampled Jacobian checks.')
    boundary,outer_triangles=external_faces(solid,hexes,wedges)
    csv_array(output/'midsurface_nodes.csv',np.column_stack((np.arange(1,len(nodes)+1),nodes,normals,t)),
              'NodeID,X,Y,Z,NormalX,NormalY,NormalZ,Thickness')
    for name,e in [('shell_quad4',quads),('shell_tri3',triangles),('solid_hex8',hexes),('solid_wedge6',wedges)]:
        csv_array(output/(name+'.csv'),np.column_stack((np.arange(1,len(e)+1),e+1)),
                  'ElementID,'+','.join(f'N{i+1}' for i in range(e.shape[1])),integer=True)
    csv_array(output/'solid_nodes.csv',np.column_stack((np.arange(1,len(solid)+1),solid)),'NodeID,X,Y,Z')
    for axis,label in enumerate('xyz'):
        for value,side in [(0,'min'),(lengths[axis],'max')]:
            ids=np.flatnonzero(np.abs(solid[:,axis]-value)<1e-8*max(lengths))+1
            csv_array(output/f'solid_boundary_{label}{side}.csv',ids,'NodeID',integer=True)
    n=len(nodes)
    csv_array(output/'solid_layers.csv',np.column_stack((np.arange(1,len(solid)+1),
              np.repeat(np.arange(N_THICKNESS+1),n),np.tile(np.arange(1,n+1),N_THICKNESS+1),
              np.concatenate([f*t for f in np.linspace(-.5,.5,N_THICKNESS+1)]))),
              'SolidNodeID,Layer,MidsurfaceNodeID,NormalOffset')
    write_stl(output/'graded_solid.stl',solid,outer_triangles,'graded_solid')
    write_msh(output/'graded_solid.msh',solid,hexes,wedges)
    write_stl(output/'projected_midsurface.stl',nodes,surface_triangles,'midsurface')
    report.update(cell_size=list(CELL_SIZE),cell_counts=list(CELL_COUNTS),level=LEVEL,
                  wedge_fallback_used=fallback_used,rejected_mesh_reason=rejected_mesh_reason,
                  thickness_at_cell_boundaries=list(THICKNESS_AT_X_BOUNDARIES),
                  thickness_layers=N_THICKNESS,midsurface_nodes=len(nodes),
                  shell_quads=len(quads),shell_triangles=len(triangles),solid_nodes=len(solid),
                  hex8=len(hexes),wedge6=len(wedges),solid_volume=volume,
                  minimum_sampled_detJ=min_j,exterior_watertight=True,
                  density=DENSITY,youngs_modulus=YOUNGS_MODULUS,poisson_ratio=POISSON_RATIO,
                  limitations=['Sampled Jacobian checks do not prove global injectivity.',
                               'No global self-intersection test or periodic constraints.',
                               'Thickness is separation along the midsurface normal; '
                               'it is not necessarily closest-distance wall thickness.'])
    (output/'report.json').write_text(json.dumps(report,indent=2))
    preview(output,solid,hexes,wedges,boundary,SHOW_PLOT and not args.no_show)
    print(json.dumps(report,indent=2))
    print('Finished. Solid mesh uses compatible HEX8/WEDGE6 faces; no FE solve performed.')


if __name__=='__main__':
    main()
