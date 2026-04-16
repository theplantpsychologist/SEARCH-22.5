import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import math
import random
import sympy
from src.engine.math225_core import Fraction, Vertex4D, AplusBsqrt2
from ladybug_geometry.geometry2d.pointvector import Point2D
from ladybug_geometry.geometry2d.polygon import Polygon2D
from ladybug_geometry_polyskel.polyskel import skeleton_as_edge_list
# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

random.seed(42)

def get_dir(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    g = math.gcd(abs(dx), abs(dy))
    return (dx // g, dy // g)

def reflect_node(node):
    return (node[1], node[0])

def reflect_edge(edge):
    return (reflect_node(edge[0]), reflect_node(edge[1]))

def check_unique_directions(cycle_verts, pos):
    if len(cycle_verts) < 3: return False
    dirs = []
    for i in range(len(cycle_verts)):
        u, v = cycle_verts[i], cycle_verts[(i + 1) % len(cycle_verts)]
        d = get_dir(pos[u], pos[v])
        if not dirs or dirs[-1] != d: dirs.append(d)
    if len(dirs) > 1 and dirs[0] == dirs[-1]: dirs.pop()
    return len(dirs) == len(set(dirs))

# =============================================================================
# TOPOLOGY ENGINE (Agglomerative)
# =============================================================================

def merge_faces(f1_edges, f2_edges):
    set1, set2_rev = set(f1_edges), set((v, u) for (u, v) in f2_edges)
    all_new_edges = [e for e in f1_edges if e not in set2_rev] + \
                    [e for e in f2_edges if (e[1], e[0]) not in set1]
    adj = {}
    for u, v in all_new_edges:
        if u in adj: return None
        adj[u] = v
    if not adj: return None
    start, cycle, curr = list(adj.keys())[0], [], list(adj.keys())[0]
    while True:
        if curr not in adj: return None
        cycle.append(curr); curr = adj[curr]
        if curr == start: break
    return cycle if len(cycle) == len(all_new_edges) else None

def try_merge(u, v, faces, edge_to_face, pos):
    e, rev_e = (u, v), (v, u)
    if e not in edge_to_face or rev_e not in edge_to_face: return False
    f1_id, f2_id = edge_to_face[e], edge_to_face[rev_e]
    if f1_id == f2_id: return False
    new_cycle = merge_faces(faces[f1_id], faces[f2_id])
    if not new_cycle or not check_unique_directions(new_cycle, pos): return False
    new_edges = [(new_cycle[i], new_cycle[(i+1)%len(new_cycle)]) for i in range(len(new_cycle))]
    del faces[f1_id]; del faces[f2_id]
    new_fid = max(faces.keys()) + 1 if faces else 0
    faces[new_fid] = new_edges
    for old_e in (e for e in (edge_to_face.keys()) if edge_to_face[e] in [f1_id, f2_id]):
        pass # Dict safety: logic below handles re-mapping
    edge_to_face.update({ne: new_fid for ne in new_edges})
    # Clean up stale references in edge_to_face
    for e_key in list(edge_to_face.keys()):
        if edge_to_face[e_key] in [f1_id, f2_id] and e_key not in new_edges:
            del edge_to_face[e_key]
    return True

def get_stems(faces, pos, width, height):
    stems, n2e = set(), {}
    for edges in faces.values():
        for u, v in edges: n2e.setdefault(u, []).append(v)
    for edges in faces.values():
        cv = [e[0] for e in edges]
        for i in range(len(cv)):
            curr = cv[i]
            if curr[0] in [0, width-1] or curr[1] in [0, height-1]: continue
            if get_dir(pos[cv[i-1]], pos[curr]) == get_dir(pos[curr], pos[cv[(i+1)%len(cv)]]):
                for v in n2e[curr]:
                    d = get_dir(pos[curr], pos[v])
                    d_flat = get_dir(pos[curr], pos[cv[(i+1)%len(cv)]])
                    if d != d_flat and d != (-d_flat[0], -d_flat[1]):
                        stems.add(tuple(sorted([curr, v])))
    return list(stems)

def generate_tiling(width=8, height=8, target_faces=12, symmetric=False):
    attempts = 0
    while True:
        attempts += 1
        pos, faces, fid = {(x,y): (x,y) for x in range(width) for y in range(height)}, {}, 0
        for x in range(width-1):
            for y in range(height-1):
                p1, p2, p3, p4 = (x,y), (x+1,y), (x+1,y+1), (x,y+1)
                faces[fid] = [(p1, p2), (p2, p3), (p3, p1)]; fid += 1
                faces[fid] = [(p1, p3), (p3, p4), (p4, p1)]; fid += 1
        e2f = {e: f for f, edges in faces.items() for e in edges}
        internal = [e for e, f in e2f.items() if (e[1], e[0]) in e2f and e[0] < e[1]]
        random.shuffle(internal)

        for u, v in internal:
            if len(faces) <= target_faces: break
            if symmetric:
                re = reflect_edge((u, v))
                if (u,v) == re or (u,v) == (re[1], re[0]): try_merge(u,v, faces, e2f, pos)
                else: 
                    if try_merge(u, v, faces, e2f, pos): try_merge(re[0], re[1], faces, e2f, pos)
            else: try_merge(u, v, faces, e2f, pos)

        stems = get_stems(faces, pos, width, height)
        stuck = False
        while stems:
            merged = False
            for u, v in stems:
                if symmetric:
                    re = reflect_edge((u, v))
                    if try_merge(u, v, faces, e2f, pos): 
                        try_merge(re[0], re[1], faces, e2f, pos); merged = True; break
                elif try_merge(u, v, faces, e2f, pos): merged = True; break
            if not merged: 
                stuck = True
                break
            stems = get_stems(faces, pos, width, height)

        if not stuck:
            G = nx.Graph()
            # 1. Store the node sequences for each face BEFORE cleanup
            # We take the first element 'u' from every directed edge (u, v) in the face
            face_node_paths = [[e[0] for e in edge_list] for edge_list in faces.values()]
            
            for edge_list in faces.values(): 
                G.add_edges_from(edge_list)
            
            nx.set_node_attributes(G, pos, 'pos')
            
            # 2. Collinear cleanup (Removing degree-2 nodes on straight lines)
            removed_nodes = set()
            for u in list(G.nodes()):
                if G.degree(u) == 2:
                    v1, v2 = list(G.neighbors(u))
                    if get_dir(pos[u], pos[v1]) == (-get_dir(pos[u], pos[v2])[0], -get_dir(pos[u], pos[v2])[1]):
                        G.add_edge(v1, v2)
                        G.remove_node(u)
                        removed_nodes.add(u)
            
            # 3. Filter face paths to remove nodes deleted during cleanup
            # This prevents the 'int' object is not iterable error and 
            # ensures Polygon2D receives valid vertices.
            cleaned_face_node_paths = []
            for path in face_node_paths:
                cleaned = [n for n in path if n not in removed_nodes]
                if len(cleaned) >= 3:
                    cleaned_face_node_paths.append(cleaned)
                    
            return G, cleaned_face_node_paths

# =============================================================================
# DOF ANALYSIS & LINEAR CONSTRAINTS
# =============================================================================

def get_dof_basis(G, pos, symmetric=False):
    nodes = list(G.nodes()); n2i = {n: i for i, n in enumerate(nodes)}; n = len(nodes)
    constraints = []
    # Bounds logic
    all_x, all_y = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    min_x, max_x, min_y, max_y = min(all_x), max(all_x), min(all_y), max(all_y)
    
    for u in nodes:
        i, (x, y) = n2i[u], pos[u]
        if math.isclose(x, min_x) or math.isclose(x, max_x):
            row = [0]*(2*n); row[2*i] = 1; constraints.append(row)
        if math.isclose(y, min_y) or math.isclose(y, max_y):
            row = [0]*(2*n); row[2*i+1] = 1; constraints.append(row)
        if symmetric:
            v = reflect_node(u)
            if v in n2i:
                j = n2i[v]
                if u == v: # Diagonal node: dx = dy
                    row = [0]*(2*n); row[2*i] = 1; row[2*i+1] = -1; constraints.append(row)
                elif i < j: # Reflected pair: dx_u = dy_v, dy_u = dx_v
                    r1 = [0]*(2*n); r1[2*i] = 1; r1[2*j+1] = -1; constraints.append(r1)
                    r2 = [0]*(2*n); r2[2*i+1] = 1; r2[2*j] = -1; constraints.append(r2)

    for u, v in G.edges():
        i, j = n2i[u], n2i[v]
        dx, dy = pos[v][0]-pos[u][0], pos[v][1]-pos[u][1]
        row = [0]*(2*n)
        if math.isclose(dy, 0): row[2*i+1]=1; row[2*j+1]=-1
        elif math.isclose(dx, 0): row[2*i]=1; row[2*j]=-1
        elif math.isclose(dy, dx): row[2*i+1]=1; row[2*i]=-1; row[2*j+1]=-1; row[2*j]=1
        elif math.isclose(dy, -dx): row[2*i+1]=1; row[2*i]=1; row[2*j+1]=-1; row[2*j]=-1
        if any(row): constraints.append(row)

    M = sympy.Matrix(constraints)
    print(f"Constraint matrix M has shape {M.shape} and rank {M.rank()}.")
    ns = M.nullspace()
    basis = []
    for v in ns:
        lcm = sympy.lcm([val.q for val in v if hasattr(val, 'q')] or [1])
        basis.append(np.array(v * lcm).astype(np.int64).flatten())
    return nodes, np.array(basis).T

def get_polytope(G, nodes, pos, basis, eps=0.1):
    n2i = {n: i for i, n in enumerate(nodes)}; n_dof = basis.shape[1]
    M, b = [], []
    all_x, all_y = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    x_min, x_max, y_min, y_max = min(all_x), max(all_x), min(all_y), max(all_y)

    for i, node in enumerate(nodes):
        x0, y0 = pos[node]
        M.append(basis[2*i, :]); b.append(x_max-x0)
        M.append(-basis[2*i, :]); b.append(x0-x_min)
        M.append(basis[2*i+1, :]); b.append(y_max-y0)
        M.append(-basis[2*i+1, :]); b.append(y0-y_min)

    for u, v in G.edges():
        idx_u, idx_v = n2i[u], n2i[v]
        vec = np.array(pos[v]) - np.array(pos[u])
        L0 = np.linalg.norm(vec); unit = vec / L0
        row = np.zeros(n_dof)
        for j in range(n_dof):
            row[j] = np.dot(basis[2*idx_v:2*idx_v+2, j] - basis[2*idx_u:2*idx_u+2, j], unit)
        M.append(-row); b.append(L0 - eps)
    return np.array(M), np.array(b)


# =============================================================================
# STRAIGHT SKELETON (Using Ladybug-Geometry-Polyskel)
# =============================================================================

def compute_straight_skeleton(face_nodes, pos_dict):
    """
    Computes the straight skeleton using the ladybug-geometry-polyskel library.
    Returns a list of coordinate pairs for red-line rendering.
    """
    # Convert nodes to ladybug Point2D objects
    points = [Point2D(pos_dict[n][0], pos_dict[n][1]) for n in face_nodes]
    
    # Ensure winding order is correct for the library
    poly = Polygon2D(points)
    
    # Generate skeleton segments
    try:
        skeleton_segments = skeleton_as_edge_list(poly)
        lines = []
        for seg in skeleton_segments:
            lines.append(((seg.p1.x, seg.p1.y), (seg.p2.x, seg.p2.y)))
        return lines
    except Exception:
        # Fallback for degenerate polygons during interactive tuning
        return []
# =============================================================================
# GUI & VISUALIZATION
# =============================================================================

def run_constrained_gui(width=6, target_faces=12, symmetric=True, use_bounds=True):
    G, face_node_lists = generate_tiling(width, width, target_faces, symmetric)
    pos = nx.get_node_attributes(G, 'pos')
    nodes, basis = get_dof_basis(G, pos, symmetric)
    n_dof = basis.shape[1]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.1 + (0.04 * n_dof))
    weights, sliders = [0.0] * n_dof, []
    M, b = get_polytope(G, nodes, pos, basis)

    def compute_bounds(k):
        weights_fixed = np.array(weights); weights_fixed[k] = 0
        offsets = M @ weights_fixed
        targets = b - offsets
        upper, lower = [], []
        for i, m_ik in enumerate(M[:, k]):
            if abs(m_ik) < 1e-9: continue
            limit = targets[i] / m_ik
            if m_ik > 0: upper.append(limit)
            else: lower.append(limit)
        w_min, w_max = (max(lower) if lower else -10.0), (min(upper) if upper else 10.0)
        return min(w_min, w_max), max(w_min, w_max)
    def update(val):
        for k in range(n_dof): weights[k] = sliders[k].val
        disp = basis @ np.array(weights)
        curr_p = {n: (pos[n][0]+disp[2*i], pos[n][1]+disp[2*i+1]) for i, n in enumerate(nodes)}
        
        ax.clear()
        for f in face_node_lists:
            # 1. Tile Boundary
            f_pts = [curr_p[n] for n in f]
            ax.add_patch(plt.Polygon(f_pts, fill=False, edgecolor='black', lw=1.5))
            
            # 2. Straight Skeleton (Red Lines)
            skel_lines = compute_straight_skeleton(f, curr_p)
            for p1, p2 in skel_lines:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', lw=0.8, alpha=0.6)
                
        ax.set_aspect('equal'); ax.axis('off')
        if use_bounds:
            for k in range(n_dof):
                w_m, w_M = compute_bounds(k)
                sliders[k].valmin, sliders[k].valmax = w_m, w_M
                sliders[k].ax.set_xlim(w_m, w_M)
        fig.canvas.draw_idle()

    for i in range(n_dof):
        ax_s = plt.axes([0.2, 0.05 + (i * 0.035), 0.6, 0.025])
        w_m, w_M = compute_bounds(i) if use_bounds else (-2.0, 2.0)
        s = Slider(ax_s, f'DOF {i}', w_m, w_M, valinit=0.0)
        s.on_changed(update); sliders.append(s)

    print(f"number of dof: {n_dof},\n number of vertices: {len(nodes)}, \n \n dimensions of A: {basis.shape[0]}x{basis.shape[1]}, \nnumber of faces: {len(face_node_lists)}, \nnumber of edges: {G.number_of_edges()}. \nPredicted dof: {2*len(nodes) - len(G.edges()) - 4 + (0 if not symmetric else -1*len(nodes))}.")
    print("")

    update(0); plt.show()
if __name__ == "__main__":
    run_constrained_gui(width=7, target_faces=30, symmetric=True, use_bounds=True)