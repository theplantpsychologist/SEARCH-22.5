"""
Load planar graph from database.
Map initial positions, construct constraints in an exact 4D basis:
Basis: (1,0), (s,s), (0,1), (-s,s) where s = sqrt(2)/2.
Variables per node: x, y, z, w.
Solve exactly over Rational fractions when full rank.
Output Cp225 object.
"""

from database.tilings.build_topologies import extract_topology
from src.engine.math225_core import Vertex4D, Fraction
from src.engine.cp225 import Cp225
from py_straight_skeleton import compute_skeleton

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import random
from scipy.linalg import null_space
from itertools import combinations

# =============================================================================
# 1. GRAPH CLEANUP & METADATA
# =============================================================================

def clean_deg2_vertices(G_in, N):
    G = G_in.copy()
    pos = {n: (n[0]/N, n[1]/N) for n in G.nodes()} 
    
    while True:
        removed_any = False
        deg2_nodes = [n for n in G.nodes() if G.degree(n) == 2]
        for node in deg2_nodes:
            nbrs = list(G.neighbors(node))
            if len(nbrs) != 2: continue
            u, v = nbrs
            dx1, dy1 = pos[u][0] - pos[node][0], pos[u][1] - pos[node][1]
            dx2, dy2 = pos[v][0] - pos[node][0], pos[v][1] - pos[node][1]
            if math.isclose(dx1*dy2 - dy1*dx2, 0, abs_tol=1e-7):
                G.remove_node(node)
                G.add_edge(u, v)
                removed_any = True
                break 
        if not removed_any: break
            
    pos = {n: pos[n] for n in G.nodes()}
    nodes = list(G.nodes())
    # n2i maps node to index in variable list, used for constraint construction
    n2i = {n: i for i, n in enumerate(nodes)}
    return G, pos, nodes, n2i

def extract_oriented_faces(G, pos):
    adj = {}
    for u, v in G.edges():
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    for u in adj:
        adj[u].sort(key=lambda v: math.atan2(pos[v][1] - pos[u][1], pos[v][0] - pos[u][0]))

    unvisited_he = set((u, v) for u, v in G.edges()) | set((v, u) for u, v in G.edges())
    faces = []
    
    while unvisited_he:
        start_he = unvisited_he.pop()
        face = [start_he[0]]
        curr_he = start_he
        while True:
            u, v = curr_he
            neighbors = adj[v]
            idx = neighbors.index(u)
            w = neighbors[(idx + 1) % len(neighbors)] 
            next_he = (v, w)
            if next_he == start_he: break
            if next_he in unvisited_he: unvisited_he.remove(next_he)
            face.append(next_he[0])
            curr_he = next_he
        faces.append(face)
    return faces

# =============================================================================
# 2. BASIC 4D CONSTRAINT GENERATORS
# =============================================================================

def get_angle(nx, ny):
    """Maps float outward normal to integer code 0-15. Using 22.5 for consistency with other modules, but we only expect multiples of 45 (even)"""
    return int(math.atan2(ny, nx) / (math.pi/8)) % 16

def build_angle_constraints_4d(G, pos, n2i):
    """
    for each edge, determine its normal direction and construct constraints to hold the edge angle constant.
    For example, the total height of a vertex is z + sqrt2/2 (y+w). So for a horizontal edge, we enforce z1=z2, y1+w1 = y2+w2
    """
    M, b = [], [] # List of constraints. Each constraint is a dict in M mapping variable index to coefficient, and a corresponding constant term in b. Will be integrated into a matrix form later
    for u, v in G.edges():
        i, j = 4*n2i[u], 4*n2i[v]
        x1,y1,z1,w1 = i, i+1, i+2, i+3
        x2,y2,z2,w2 = j, j+1, j+2, j+3
        dx, dy = pos[v][0] - pos[u][0], pos[v][1] - pos[u][1]
        L = math.hypot(dx, dy)
        angle = get_angle(-dy/L, dx/L)

        if angle in {4,12}: # Vertical normal, horizontal edge
            M.extend([{z1:1,z2:-1}, {y1:1,w1:1,y2:-1,w2:-1}])

        elif angle in {6,14}: # Diag \ normal, diag / edge
            M.extend([{w1:1,w2:-1}, {z1:1,x1:-1,z2:-1,x2:1}])

        elif angle in {0, 8}: # Horizontal normal, vertical edge
            M.extend([{x1:1,x2:-1}, {y1:1,w1:-1,y2:-1,w2:1}])

        elif angle in {2,10}: # Diag / normal, diag \ edge
            M.extend([{y1:1,y2:-1}, {x1:1,z1:1,x2:-1,z2:-1}])
            
        b.extend([0, 0])
    return M, b

def build_symmetry_constraints_4d(nodes, n2i, symmetry, N):
    """
    For diagonal symmetry: x1=z2, z1=x2, y1=y2, w1=w2
    For book symmetry: x1 = 1-x2, z1=z2, y1=w2, w1=y2
    Or if its on the line of symmetry, then for diagonal: x=z, w=0. For book: x=0.5, y=w
    """
    M, b = [], []
    if symmetry == 'none': return M, b
    for u in nodes:
        i = 4*n2i[u]
        u_sym = (u[1], u[0]) if symmetry == 'diag' else (N - u[0], u[1]) #symmetric twin (could be self)
        if u_sym in n2i:
            j = 4*n2i[u_sym]
            x1,y1,z1,w1 = i, i+1, i+2, i+3
            x2,y2,z2,w2 = j, j+1, j+2, j+3
            if i<j: # if the two nodes are paired about symmetry line
                if symmetry == 'diag':
                    M.extend([{x1:1,z2:-1},{z1:1,x2:-1},{w1:1,w2:-1},{y1:1,y2:-1}])
                    b.extend([0, 0, 0, 0])
                elif symmetry == 'book':
                    M.extend([{x1:1,x2:1},{z1:1,z2:-1},{y1:1,w2:-1},{w1:1,y2:-1}])
                    b.extend([1, 0, 0, 0])
            elif i==j: # if the node is on the symmetry line
                if symmetry == 'diag':
                    M.extend([{x1:1,z1:-1},{w1:1}])
                    b.extend([0, 0])
                elif symmetry == 'book':
                    M.extend([{x1:2},{y1:1,w1:-1}])
                    b.extend([1, 0])
    return M, b

def build_boundary_constraints_4d(n2i, N):
    """
    Pin the bottom left corner to (0,0) and the top right corner to (1,1) in the original 2D space, which translates to (0,0,0,0) and (1,0,1,0) in the 4D basis. 
    """
    M, b = [], []
    i = 4*n2i[(0,0)]
    j = 4*n2i[(N,N)]
    x1,y1,z1,w1 = i, i+1, i+2, i+3
    x2,y2,z2,w2 = j, j+1, j+2, j+3

    M.extend([{x1: 1}, {y1: 1}, {z1: 1}, {w1: 1}])
    b.extend([0, 0, 0, 0])

    M.extend([{x2: 1}, {y2: 1}, {z2: 1}, {w2: 1}])
    b.extend([1, 0, 1, 0])
    return M, b

ANGLE_TO_4D = {
    0: [1, 0, 0, 0],
    2: [0, 1, 0, 0],
    4: [0, 0, 1, 0],
    6: [0, 0, 0, 1],
    8: [-1, 0, 0, 0],
    10: [0, -1, 0, 0],
    12: [0, 0, -1, 0],
    14: [0, 0, 0, -1]
}
def build_quadruplet_constraint_4d(edges, pos,n2i):
    """
    Constrain 4 edges to have an equidistant incircle center. Returns constraint in sparse form (list of dicts)
    input edges is a tuple of dicts where each dict contains a bunch of precomputed info about each edge
    """
    A = []
    for edge in edges:
        normal = ANGLE_TO_4D[edge["angle"]]
        A.append(normal + [-1])
    A = np.array(A)
    W = null_space(A.T)
    n = len(n2i)
    M_rows = []
    for col in range(W.shape[1]):
        w = W[:, col]
        #un-normalize so w is integer
        w = w / min(abs(w[w.nonzero()]))
        w = w.astype(int)
        constraint = {} #a sparse row in the constraint matrix. Should have one nonzero (+-1) element per edge
        for local_idx, edge in enumerate(edges):
            u = edge["u"]
            idx = 4*n2i[u]
            constraint[idx], constraint[idx+1], constraint[idx+2], constraint[idx+3] = w[local_idx]*np.array(ANGLE_TO_4D[edge["angle"]])
        M_rows.append(constraint)
    return M_rows, [0] * len(M_rows)

# =============================================================================
# 3. STRAIGHT-SKELETON BASED CONSTRAINT SELECTION AND GENERATORS
# =============================================================================

def get_edge_data(face, pos):
    k = len(face)
    face_edges = [(face[i], face[(i+1)%k]) for i in range(k)]
    edge_data = []
    for u, v in face_edges:
        p_u, p_v = pos[u], pos[v]
        dx, dy = p_v[0] - p_u[0], p_v[1] - p_u[1]
        L = math.hypot(dx, dy)
        if L < 1e-7: continue
        nx, ny = -dy/L, dx/L 
        eta = nx * p_u[0] + ny * p_u[1]
        angle = get_angle(nx, ny)
        edge_data.append({'e': (u, v), 'u': u, 'v': v, 'n': (nx, ny), 'eta': eta, 'pu': p_u, 'pv': p_v, 'angle': angle})
    return edge_data

def get_py_skeleton_lines(face, pos):
    if len(face) < 3: return []
    exterior = [[float(pos[n][0]), float(pos[n][1])] for n in face]
    area = sum(exterior[i][0] * exterior[(i+1)%len(exterior)][1] - exterior[(i+1)%len(exterior)][0] * exterior[i][1] for i in range(len(exterior)))
    if abs(area) / 2.0 > 0.95: return []
    if area < 0: exterior.reverse()
        
    try:
        skeleton = compute_skeleton(exterior=exterior, holes=[])
    except Exception: return []
        
    lines = []
    for skv1, skv2 in skeleton.arc_iterator():
        p1 = (float(getattr(skv1.position, 'x', skv1.position[0])), float(getattr(skv1.position, 'y', skv1.position[1])))
        p2 = (float(getattr(skv2.position, 'x', skv2.position[0])), float(getattr(skv2.position, 'y', skv2.position[1])))
        lines.append((p1, p2))
    return lines

def harvest_candidates(faces, face_indices, pos_init):
    candidates = []
    for face_idx in face_indices:
        face = faces[face_idx]
        edge_data = get_edge_data(face, pos_init)
        perimeter = sum(math.hypot(ed['pv'][0]-ed['pu'][0], ed['pv'][1]-ed['pu'][1]) for ed in edge_data)
        if perimeter < 1e-7: perimeter = 1e-7 
        
        # Fast algebraic harvest mimicking Straight Skeleton Topology
        skel_vertices = {}
        for e1, e2, e3 in combinations(edge_data, 3):
            A_sys = np.array([[e1['n'][0], e1['n'][1], -1], [e2['n'][0], e2['n'][1], -1], [e3['n'][0], e3['n'][1], -1]])
            b_sys = np.array([e1['eta'], e2['eta'], e3['eta']])
            try: P_triplet = np.linalg.solve(A_sys, b_sys)
            except np.linalg.LinAlgError: continue
            Px, Py = P_triplet[0], P_triplet[1]
            # Simple check, rely on topological veto downstream
            key = (round(Px, 4), round(Py, 4))
            if key not in skel_vertices: skel_vertices[key] = {'edges': set(), 'P': (Px, Py)}
            skel_vertices[key]['edges'].update([e1['e'], e2['e'], e3['e']])
            
        # Match back to edge dicts
        edge_lookup = {ed['e']: ed for ed in edge_data}
        deg3_verts = []
        for key, vdata in skel_vertices.items():
            edges_list = [edge_lookup[e] for e in vdata['edges']]
            if len(edges_list) >= 4:
                for combo in combinations(edges_list, 4):
                    candidates.append({'face_idx': face_idx, 'edges': combo, 'norm_dist': 0.0, 'skel_edge': (vdata['P'], vdata['P']), 'P': vdata['P']})
            elif len(edges_list) == 3:
                deg3_verts.append(vdata)
                
        for i in range(len(deg3_verts)):
            for j in range(i+1, len(deg3_verts)):
                v1, v2 = deg3_verts[i], deg3_verts[j]
                shared = [e for e in v1['edges'] if e in v2['edges']]
                if len(shared) == 2: 
                    union_edges = [edge_lookup[e] for e in list(set(v1['edges']).union(set(v2['edges'])))]
                    if len(union_edges) == 4:
                        phys_dist = math.hypot(v1['P'][0]-v2['P'][0], v1['P'][1]-v2['P'][1])
                        P_mid = ((v1['P'][0]+v2['P'][0])/2, (v1['P'][1]+v2['P'][1])/2)
                        candidates.append({'face_idx': face_idx, 'edges': tuple(union_edges), 'norm_dist': phys_dist / perimeter, 'skel_edge': (v1['P'], v2['P']), 'P': P_mid})
    return candidates

def is_valid_diag_cand(edges_info, pos_init):
    for e in edges_info:
        u, v = e['u'], e['v']
        if pos_init[u][1] < pos_init[u][0] - 1e-5 or pos_init[v][1] < pos_init[v][0] - 1e-5:
            return True
    return False


# =============================================================================
# 4. EXACT SOLVER PIPELINE
# =============================================================================

def build_dense(M_list, num_vars):
    """Convert list-of-dict sparse representation to dense numpy array for rank checks and solves."""

    arr = np.zeros((len(M_list), num_vars))
    for r, d in enumerate(M_list):
        for c, v in d.items(): arr[r, c] = v
    return arr

def exact_fraction_solve(M_list, b_list, num_vars):
    """Custom Exact Gauss-Jordan over Fraction class."""
    mat = [[Fraction(0) for _ in range(num_vars + 1)] for _ in range(len(M_list))]
    for r, row_dict in enumerate(M_list):
        for c, coef in row_dict.items(): mat[r][c] = Fraction(int(coef))
        mat[r][-1] = Fraction(int(b_list[r]) if b_list[r].is_integer() else b_list[r])
        
    row = 0
    for col in range(num_vars):
        pivot = -1
        for i in range(row, len(mat)):
            if mat[i][col] != 0:
                pivot = i; break
        if pivot == -1: continue
            
        mat[row], mat[pivot] = mat[pivot], mat[row]
        inv = Fraction(mat[row][col].den, mat[row][col].num)
        for j in range(col, num_vars + 1): mat[row][j] *= inv
            
        for i in range(len(mat)):
            if i != row and mat[i][col] != 0:
                factor = mat[i][col]
                for j in range(col, num_vars + 1): mat[i][j] -= factor * mat[row][j]
        row += 1
        
    ans = [Fraction(0)] * num_vars
    for i in range(num_vars):
        for j in range(num_vars):
            if mat[i][j] == 1:
                ans[j] = mat[i][-1]
                break
    return ans

def solve_absolute_positions(G_in, symmetry='none', N=4):
    G, pos_init, nodes, n2i = clean_deg2_vertices(G_in, N)
    n = len(nodes)
    
    # Track 4D variables
    num_vars = 4 * n
    X_init_list = []
    for u in nodes:
        X_init_list.extend([pos_init[u][0], 0, pos_init[u][1], 0])
        
    M_ang, b_ang = build_angle_constraints_4d(G, pos_init, n2i)
    M_sym, b_sym = build_symmetry_constraints_4d(nodes, n2i, symmetry, N)
    M_bnd, b_bnd = build_boundary_constraints_4d(n2i, N)
    M_list = M_ang + M_sym + M_bnd
    b_list = b_ang + b_sym + b_bnd
    
    faces = extract_oriented_faces(G, pos_init)
    
    # 1. Base matrix baseline
    M_dense = build_dense(M_list, num_vars)
    current_rank = np.linalg.matrix_rank(M_dense, tol=1e-7)
    
    # 2. Gather and sort candidates
    poly_idx = [i for i, f in enumerate(faces) if len(f) > 4]
    all_candidates = harvest_candidates(faces, poly_idx, pos_init)
    if symmetry == 'diag': 
        all_candidates = [c for c in all_candidates if is_valid_diag_cand(c['edges'], pos_init)]
        
    m_counts = {i: 0 for i in range(len(faces))}
    applied = []
    
    # 3. Process the priority queue
    while all_candidates and current_rank < num_vars:
        # Dynamic re-sort based on your priority heuristics
        all_candidates.sort(key=lambda c: (
            -(len(faces[c['face_idx']]) - 2 - m_counts[c['face_idx']]), 
            c['norm_dist']
        ))
        
        cand = all_candidates.pop(0)
        M_eq, b_eq = build_quadruplet_constraint_4d(cand['edges'], pos_init, n2i)
        if not M_eq: 
            continue
            
        cand['M_eq'] = M_eq
        cand['b_eq'] = b_eq
        
        # Tentatively apply constraints
        M_test = M_list + M_eq
        b_test = b_list + b_eq
        
        M_test_dense = build_dense(M_test, num_vars)
        new_rank = np.linalg.matrix_rank(M_test_dense, tol=1e-7)
        
        if new_rank > current_rank:
            # Check for overconstrained (inconsistent system)
            aug_matrix = np.column_stack([M_test_dense, b_test])
            if np.linalg.matrix_rank(aug_matrix, tol=1e-7) == new_rank:
                # Rank increased and system is consistent. Keep it!
                M_list = M_test
                b_list = b_test
                current_rank = new_rank
                applied.append(cand)
                m_counts[cand['face_idx']] += 1

    # 4. Fallback to Quadrilaterals
    if current_rank < num_vars:
        # print("Warning: Matrix underconstrained. Falling back to quadrilaterals...")
        quad_idx = [i for i, f in enumerate(faces) if len(f) == 4]
        q_cands = harvest_candidates(faces, quad_idx, pos_init)
        if symmetry == 'diag': 
            q_cands = [c for c in q_cands if is_valid_diag_cand(c['edges'], pos_init)]
            
        q_cands.sort(key=lambda c: c['norm_dist'])
        
        while q_cands and current_rank < num_vars:
            cand = q_cands.pop(0)
            M_eq, b_eq = build_quadruplet_constraint_4d(cand['edges'], pos_init, n2i)
            if not M_eq: 
                continue
                
            cand['M_eq'] = M_eq
            cand['b_eq'] = b_eq
            
            M_test = M_list + M_eq
            b_test = b_list + b_eq
            
            M_test_dense = build_dense(M_test, num_vars)
            new_rank = np.linalg.matrix_rank(M_test_dense, tol=1e-7)
            
            if new_rank > current_rank:
                aug_matrix = np.column_stack([M_test_dense, b_test])
                if np.linalg.matrix_rank(aug_matrix, tol=1e-7) == new_rank:
                    M_list = M_test
                    b_list = b_test
                    current_rank = new_rank
                    applied.append(cand)

    print(f"Final Matrix Rank: {current_rank} / {num_vars}")

    # 5. Output Parsing
    if current_rank == num_vars:
        print(f"Matrix Full Rank. Solving Exactly.")
        ans = exact_fraction_solve(M_list, b_list, num_vars)
        vertices = [Vertex4D(ans[4*i], ans[4*i+1], ans[4*i+2], ans[4*i+3]) for i in range(n)]
        cp_edges = [(n2i[u], n2i[v], 'm') for u, v in G.edges()]
        cp = Cp225(vertices, cp_edges)
        
        # pos_solved = {}
        # for i, u in enumerate(nodes):
        #     x, y, z, w = float(ans[4*i]), float(ans[4*i+1]), float(ans[4*i+2]), float(ans[4*i+3])
        #     pos_solved[u] = (x + 0.70710678*(y-w), z + 0.70710678*(y+w))
    else:
        print("Warning: Exhausted all candidates, still rank deficient. Float displacement solve.")
        M_dense = build_dense(M_list, num_vars)
        delta = np.linalg.lstsq(M_dense, np.array(b_list) - M_dense @ np.array(X_init_list), rcond=None)[0]
        ans = np.array(X_init_list) + delta
        cp = None
        # pos_solved = {}
        # for i, u in enumerate(nodes):
        #     x, y, z, w = ans[4*i:4*i+4]
        #     pos_solved[u] = (x + 0.70710678*(y-w), z + 0.70710678*(y+w))
            
    return G, pos_init, faces, applied, cp

# =============================================================================
# 5. PLOTTING
# =============================================================================

def draw_before_ax(ax, G, pos_init, faces, applied_constraints, title):
    ax.set_title(title, fontsize=14)
    for u, v in G.edges():
        ax.plot([pos_init[u][0], pos_init[v][0]], [pos_init[u][1], pos_init[v][1]], 'k-', lw=1.5, zorder=2, alpha=0.3)
    for face in faces:
        for p1, p2 in get_py_skeleton_lines(face, pos_init):
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', lw=1.5, alpha=0.5, zorder=1)

    colors = cm.get_cmap('tab20', max(1, len(applied_constraints)))
    epsilon = -0.008 
    for idx, cand in enumerate(applied_constraints):
        color = colors(idx)
        p1, p2 = cand['skel_edge']
        if math.isclose(p1[0], p2[0], abs_tol=1e-5) and math.isclose(p1[1], p2[1], abs_tol=1e-5):
            ax.plot(p1[0], p1[1], marker='X', color=color, markersize=10, zorder=5)
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=5.0, zorder=5)
        mid_x, mid_y = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
        for e in cand['edges']:
            u, v, code = e['u'], e['v'], e['angle']
            dx, dy = pos_init[v][0] - pos_init[u][0], pos_init[v][1] - pos_init[u][1]
            L = math.hypot(dx, dy)
            nx, ny = -dy/L, dx/L if L > 1e-7 else (0,0)
            u_shift = (pos_init[u][0] + nx * epsilon, pos_init[u][1] + ny * epsilon)
            v_shift = (pos_init[v][0] + nx * epsilon, pos_init[v][1] + ny * epsilon)
            ax.plot([u_shift[0], v_shift[0]], [u_shift[1], v_shift[1]], color=color, lw=3.0, zorder=4, solid_capstyle='round')
            ax.plot([mid_x, (pos_init[u][0]+pos_init[v][0])/2], [mid_y, (pos_init[u][1]+pos_init[v][1])/2], color=color, linestyle=':', lw=1.5, zorder=4, alpha=0.8)
    ax.set_aspect('equal')
    ax.axis('off')

def draw_after_ax(ax, G, faces, cp, title):
    ax.set_title(title, fontsize=14)    
    if cp:
        for t, x1, y1, x2, y2 in cp.render():
            ax.plot([x1, x2], [y1, y2], 'k-', lw=2, zorder=2, alpha=0.7)

    ax.set_aspect('equal')
    ax.axis('off')

def plot_multiple_before_after(results, filename=None):
    if not results: return
    n_res = len(results)
    n_rows = max(1, int(np.round(np.sqrt(n_res / 1.5))))
    n_cols = int(np.ceil(n_res / n_rows))
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(8 * n_cols, 4 * n_rows))
    axes_flat = axes.flatten() if n_res > 1 else np.array(axes).flatten()

    for i in range(n_res):
        G, pos_init, faces, applied_constraints, cp = results[i]
        draw_before_ax(axes_flat[i * 2], G, pos_init, faces, applied_constraints, f"Graph {i+1}: Before")
        draw_after_ax(axes_flat[i * 2 + 1], G, faces, cp, f"Graph {i+1}: After")
        
    for j in range(n_res * 2, len(axes_flat)): axes_flat[j].axis('off')
    plt.tight_layout()
    if filename: plt.savefig(filename, dpi=200)
    plt.show()

# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    solved_results = []
    for i in random.sample(range(1, 9000), 5):
        G_raw = extract_topology(i, db_name="topologies_4_none.db", N=4)
        G_solved, pos_init, faces, applied, cp = solve_absolute_positions(G_raw, symmetry='none', N=4)
        print(f"ID {i}: Applied {len(applied)} constraints.")
        solved_results.append((G_solved, pos_init,faces, applied, cp))
            
    plot_multiple_before_after(solved_results, filename="renders/debug_batch_none_exact.png")