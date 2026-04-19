"""
Load planar graph from database

Clean up deg 2 vertices. From this point on, vertices will be assumed to lie in the unit square with floats, rather than grid points

generate matrix and homogeneous b (column of zeros) for angle constraints. 
generate matrix and homogeneous b (column of zeros) for diag or book symmetry. 
generate matrix and homogeneous b to pin the bottom left vertex to 0,0 and the top right vertex to 1,1. 

Finally, solve the system by taking the pseudoinverse of the constraint matrix and multiplying by the homogeneous b vector. 
Output the vertex positions as a new graph with the same topology, and plot, with straight skeletons overlaid for each polygon.
"""

from database.tilings.build_topologies import extract_topology

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from scipy.linalg import null_space
import random
from itertools import combinations

from py_straight_skeleton import compute_skeleton

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
            cross = dx1*dy2 - dy1*dx2
            
            if math.isclose(cross, 0, abs_tol=1e-7):
                G.remove_node(node)
                G.add_edge(u, v)
                removed_any = True
                break 
                
        if not removed_any:
            break
            
    pos = {n: pos[n] for n in G.nodes()}
    nodes = list(G.nodes())
    n2i = {n: i for i, n in enumerate(nodes)}
    
    return G, pos, nodes, n2i

# =============================================================================
# 2. FACE EXTRACTION
# =============================================================================

def extract_oriented_faces(G, pos):
    adj = {}
    for u, v in G.edges():
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    for u in adj:
        adj[u].sort(key=lambda v: math.atan2(pos[v][1] - pos[u][1], pos[v][0] - pos[u][0]))

    unvisited_he = set()
    for u, v in G.edges():
        unvisited_he.add((u, v))
        unvisited_he.add((v, u))

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
# 3. PY_STRAIGHT_SKELETON WRAPPERS & HARVESTERS
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
        edge_data.append({'e': (u, v), 'n': (nx, ny), 'eta': eta, 'pu': p_u, 'pv': p_v})
    return edge_data

def compute_interior_skeleton_vertices(face, pos, edge_data):
    skel_vertices = {}
    exterior = [[float(pos[n][0]), float(pos[n][1])] for n in face]
    
    area = 0
    k = len(exterior)
    for i in range(k):
        area += exterior[i][0] * exterior[(i+1)%k][1] - exterior[(i+1)%k][0] * exterior[i][1]
        
    if abs(area) / 2.0 > 0.95: 
        return {}
        
    if area < 0:
        exterior.reverse() 
        
    try:
        skeleton = compute_skeleton(exterior=exterior, holes=[])
    except Exception:
        return {}

    internal_positions = []
    for skv1, skv2 in skeleton.arc_iterator():
        for skv in (skv1, skv2):
            try:
                Px, Py = float(skv.position.x), float(skv.position.y)
            except AttributeError:
                Px, Py = float(skv.position[0]), float(skv.position[1])
                
            is_internal = True
            for ex, ey in exterior:
                if math.isclose(Px, ex, abs_tol=1e-5) and math.isclose(Py, ey, abs_tol=1e-5):
                    is_internal = False
                    break
            if is_internal:
                internal_positions.append((Px, Py))
                
    unique_positions = []
    for px, py in internal_positions:
        if not any(math.isclose(px, ux, abs_tol=1e-5) and math.isclose(py, uy, abs_tol=1e-5) for ux, uy in unique_positions):
            unique_positions.append((px, py))
            
    for Px, Py in unique_positions:
        gen_edges = set()
        for e1, e2, e3 in combinations(edge_data, 3):
            A_sys = np.array([
                [e1['n'][0], e1['n'][1], -1],
                [e2['n'][0], e2['n'][1], -1],
                [e3['n'][0], e3['n'][1], -1]
            ])
            b_sys = np.array([e1['eta'], e2['eta'], e3['eta']])
            try:
                P_triplet = np.linalg.solve(A_sys, b_sys)
                if math.isclose(Px, P_triplet[0], abs_tol=1e-4) and math.isclose(Py, P_triplet[1], abs_tol=1e-4):
                    gen_edges.update([e1['e'], e2['e'], e3['e']])
            except np.linalg.LinAlgError:
                pass
                
        if len(gen_edges) >= 3:
            key = (round(Px, 5), round(Py, 5))
            skel_vertices[key] = {'edges': gen_edges, 'P': (Px, Py)}
            
    return skel_vertices

def get_py_skeleton_lines(face, pos):
    if len(face) < 3: return []
    exterior = [[float(pos[n][0]), float(pos[n][1])] for n in face]
    area = 0
    k = len(exterior)
    for i in range(k):
        area += exterior[i][0] * exterior[(i+1)%k][1] - exterior[(i+1)%k][0] * exterior[i][1]
        
    if abs(area) / 2.0 > 0.95: return []
    if area < 0: exterior.reverse()
        
    try:
        skeleton = compute_skeleton(exterior=exterior, holes=[])
    except Exception:
        return []
        
    lines = []
    for skv1, skv2 in skeleton.arc_iterator():
        try:
            p1 = (float(skv1.position.x), float(skv1.position.y))
            p2 = (float(skv2.position.x), float(skv2.position.y))
        except AttributeError:
            p1 = (float(skv1.position[0]), float(skv1.position[1]))
            p2 = (float(skv2.position[0]), float(skv2.position[1]))
        lines.append((p1, p2))
            
    return lines

def harvest_candidates(faces, face_indices, pos_init):
    """Modular function to harvest candidate quadruplets from a specific subset of faces."""
    candidates = []
    for face_idx in face_indices:
        face = faces[face_idx]
        edge_data = get_edge_data(face, pos_init)
        
        perimeter = sum(math.hypot(ed['pv'][0] - ed['pu'][0], ed['pv'][1] - ed['pu'][1]) for ed in edge_data)
        if perimeter < 1e-7: perimeter = 1e-7 
        
        skel_vertices = compute_interior_skeleton_vertices(face, pos_init, edge_data)
                
        deg3_verts = []
        for key, vdata in skel_vertices.items():
            edges_list = list(vdata['edges'])
            if len(edges_list) >= 4:
                for combo in combinations(edges_list, 4):
                    candidates.append({
                        'face_idx': face_idx, 'edges': combo, 
                        'norm_dist': 0.0, 'skel_edge': (vdata['P'], vdata['P']) 
                    })
            elif len(edges_list) == 3:
                deg3_verts.append(vdata)
                
        for i in range(len(deg3_verts)):
            for j in range(i+1, len(deg3_verts)):
                v1, v2 = deg3_verts[i], deg3_verts[j]
                shared_edges = v1['edges'].intersection(v2['edges'])
                if len(shared_edges) == 2: 
                    union_edges = list(v1['edges'].union(v2['edges']))
                    if len(union_edges) == 4:
                        phys_dist = math.hypot(v1['P'][0] - v2['P'][0], v1['P'][1] - v2['P'][1])
                        candidates.append({
                            'face_idx': face_idx, 'edges': tuple(union_edges), 
                            'norm_dist': phys_dist / perimeter, 'skel_edge': (v1['P'], v2['P'])
                        })
    return candidates

def is_valid_diag_cand(edges, pos_init):
    """Returns True if at least one vertex of the quadruplet edges is strictly below the y=x line."""
    for u, v in edges:
        if pos_init[u][1] < pos_init[u][0] - 1e-5 or pos_init[v][1] < pos_init[v][0] - 1e-5:
            return True
    return False

# =============================================================================
# 4. CONSTRAINT GENERATORS
# =============================================================================

def build_angle_constraints(G, pos, n2i):
    n = len(n2i)
    M, b = [], []
    for u, v in G.edges():
        i, j = n2i[u], n2i[v]
        dx, dy = pos[v][0] - pos[u][0], pos[v][1] - pos[u][1]
        row = [0] * (2*n)
        if math.isclose(dy, 0, abs_tol=1e-5):     row[2*i+1] = 1; row[2*j+1] = -1
        elif math.isclose(dx, 0, abs_tol=1e-5):   row[2*i] = 1; row[2*j] = -1
        elif math.isclose(dy, dx, abs_tol=1e-5):  row[2*i+1] = 1; row[2*i] = -1; row[2*j+1] = -1; row[2*j] = 1
        elif math.isclose(dy, -dx, abs_tol=1e-5): row[2*i+1] = 1; row[2*i] = 1; row[2*j+1] = -1; row[2*j] = -1
        if any(row): M.append(row); b.append(0)
    return np.array(M), np.array(b)

def build_symmetry_constraints(nodes, pos, n2i, symmetry, N):
    n = len(n2i)
    M, b = [], []
    if symmetry == 'none': return np.empty((0, 2*n)), np.empty(0)
    for u in nodes:
        i = n2i[u]
        if symmetry == 'diag': u_sym = (u[1], u[0])
        elif symmetry == 'book': u_sym = (N - u[0], u[1])
        if u_sym in n2i:
            j = n2i[u_sym]
            if i < j: 
                if symmetry == 'diag':
                    r1 = [0]*(2*n); r1[2*i] = 1; r1[2*j+1] = -1
                    r2 = [0]*(2*n); r2[2*i+1] = 1; r2[2*j] = -1
                    M.extend([r1, r2]); b.extend([0, 0])
                elif symmetry == 'book':
                    r1 = [0]*(2*n); r1[2*i] = 1; r1[2*j] = 1
                    r2 = [0]*(2*n); r2[2*i+1] = 1; r2[2*j+1] = -1
                    M.extend([r1, r2]); b.extend([1, 0])
            elif i == j: 
                if symmetry == 'diag':
                    r1 = [0]*(2*n); r1[2*i] = 1; r1[2*i+1] = -1
                    M.append(r1); b.append(0)
                elif symmetry == 'book':
                    r1 = [0]*(2*n); r1[2*i] = 1
                    M.append(r1); b.append(0.5)
    return np.array(M), np.array(b)

def build_boundary_constraints(nodes, n2i, N):
    n = len(n2i)
    M, b = [], []
    bl, tr = (0, 0), (N, N)
    if bl in n2i:
        idx = n2i[bl]
        r1 = [0]*(2*n); r1[2*idx] = 1; r2 = [0]*(2*n); r2[2*idx+1] = 1
        M.extend([r1, r2]); b.extend([0, 0])
    if tr in n2i:
        idx = n2i[tr]
        r1 = [0]*(2*n); r1[2*idx] = 1; r2 = [0]*(2*n); r2[2*idx+1] = 1
        M.extend([r1, r2]); b.extend([1, 1])
    return np.array(M), np.array(b)

def build_equidistant_for_edges(edges, pos, n2i):
    k = len(edges)
    A = []
    eta_init = []
    for u, v in edges:
        dx, dy = pos[v][0] - pos[u][0], pos[v][1] - pos[u][1]
        L = math.hypot(dx, dy)
        nx, ny = -dy/L, dx/L 
        A.append([nx, ny, -1])
        eta_init.append(nx * pos[u][0] + ny * pos[u][1])
    A = np.array(A)
    eta_init = np.array(eta_init)
    W = null_space(A.T)
    n = len(n2i)
    M_edges, b_edges = [], []
    for col in range(W.shape[1]):
        w = W[:, col]
        row = [0] * (2*n)
        for i in range(k):
            u = edges[i][0]
            idx = n2i[u]
            row[2*idx] += w[i] * A[i, 0]
            row[2*idx+1] += w[i] * A[i, 1]
        M_edges.append(row)
        b_edges.append(0)
    return np.array(M_edges), np.array(b_edges)

# =============================================================================
# 5. SOLVER PIPELINE
# =============================================================================

def solve_absolute_positions(G_in, symmetry='none', N=4):
    G, pos_init, nodes, n2i = clean_deg2_vertices(G_in, N)
    n = len(nodes)
    target_rank = 2 * n
    
    X_init_flat = np.zeros(2*n)
    for u in nodes:
        idx = n2i[u]
        X_init_flat[2*idx] = pos_init[u][0]
        X_init_flat[2*idx+1] = pos_init[u][1]
        
    M_ang, b_ang = build_angle_constraints(G, pos_init, n2i)
    M_sym, b_sym = build_symmetry_constraints(nodes, pos_init, n2i, symmetry, N)
    M_bnd, b_bnd = build_boundary_constraints(nodes, n2i, N)
    
    M = np.vstack([M_ang, M_sym, M_bnd]) if M_sym.size else np.vstack([M_ang, M_bnd])
    b = np.concatenate([b_ang, b_sym, b_bnd]) if b_sym.size else np.concatenate([b_ang, b_bnd])

    faces = extract_oriented_faces(G, pos_init)
    current_rank = np.linalg.matrix_rank(M, tol=1e-7)
    
    # -------------------------------------------------------------------------
    # PHASE 1: Complex Polygons (Min-Max Topology Heuristic)
    # -------------------------------------------------------------------------
    poly_indices = [i for i, f in enumerate(faces) if len(f) > 4]
    all_candidates = harvest_candidates(faces, poly_indices, pos_init)
    
    if symmetry == 'diag':
        all_candidates = [c for c in all_candidates if is_valid_diag_cand(c['edges'], pos_init)]

    applied_constraints = []
    m_counts = {i: 0 for i in range(len(faces))}
    
    while all_candidates and current_rank < target_rank:
        all_candidates.sort(key=lambda c: (
            -(len(faces[c['face_idx']]) - 2 - m_counts[c['face_idx']]), 
            c['norm_dist']
        ))
        
        cand = all_candidates.pop(0)
        M_eq, b_eq = build_equidistant_for_edges(cand['edges'], pos_init, n2i)
        if len(M_eq) == 0: continue
            
        cand['M_eq'] = M_eq; cand['b_eq'] = b_eq
        M_test, b_test = M.copy(), b.copy()
        test_rank = current_rank
        rank_increased = False
        
        for r, val in zip(M_eq, b_eq):
            M_temp = np.vstack([M_test, r])
            b_temp = np.append(b_test, val)
            new_rank = np.linalg.matrix_rank(M_temp, tol=1e-7)
            if new_rank > test_rank:
                if np.linalg.matrix_rank(np.column_stack([M_temp, b_temp]), tol=1e-7) == new_rank:
                    M_test, b_test, test_rank = M_temp, b_temp, new_rank
                    rank_increased = True
        
        if rank_increased:
            delta_X_test = np.linalg.pinv(M_test) @ (b_test - M_test @ X_init_flat)
            X_proposed = X_init_flat + delta_X_test
            
            is_valid_topology = True
            for u, v in G.edges():
                i, j = n2i[u], n2i[v]
                v_init = np.array([X_init_flat[2*j] - X_init_flat[2*i], X_init_flat[2*j+1] - X_init_flat[2*i+1]])
                v_prop = np.array([X_proposed[2*j] - X_proposed[2*i], X_proposed[2*j+1] - X_proposed[2*i+1]])
                if np.dot(v_init, v_prop) <= 1e-5:
                    is_valid_topology = False
                    break
            
            if is_valid_topology:
                M, b, current_rank = M_test, b_test, test_rank
                applied_constraints.append(cand)
                m_counts[cand['face_idx']] += 1

    # -------------------------------------------------------------------------
    # PHASE 2: Quadrilateral Fallback
    # -------------------------------------------------------------------------
    if current_rank < target_rank:
        print("Warning: Matrix underconstrained. Falling back to quadrilaterals...")
        quad_indices = [i for i, f in enumerate(faces) if len(f) == 4]
        quad_candidates = harvest_candidates(faces, quad_indices, pos_init)
        
        if symmetry == 'diag':
            quad_candidates = [c for c in quad_candidates if is_valid_diag_cand(c['edges'], pos_init)]
            
        quad_candidates.sort(key=lambda c: c['norm_dist'])
        
        while quad_candidates and current_rank < target_rank:
            cand = quad_candidates.pop(0)
            M_eq, b_eq = build_equidistant_for_edges(cand['edges'], pos_init, n2i)
            if len(M_eq) == 0: continue
                
            cand['M_eq'] = M_eq; cand['b_eq'] = b_eq
            M_test, b_test = M.copy(), b.copy()
            test_rank = current_rank
            rank_increased = False
            
            for r, val in zip(M_eq, b_eq):
                M_temp = np.vstack([M_test, r])
                b_temp = np.append(b_test, val)
                new_rank = np.linalg.matrix_rank(M_temp, tol=1e-7)
                if new_rank > test_rank:
                    if np.linalg.matrix_rank(np.column_stack([M_temp, b_temp]), tol=1e-7) == new_rank:
                        M_test, b_test, test_rank = M_temp, b_temp, new_rank
                        rank_increased = True
            
            if rank_increased:
                delta_X_test = np.linalg.pinv(M_test) @ (b_test - M_test @ X_init_flat)
                X_proposed = X_init_flat + delta_X_test
                
                is_valid_topology = True
                for u, v in G.edges():
                    i, j = n2i[u], n2i[v]
                    v_init = np.array([X_init_flat[2*j] - X_init_flat[2*i], X_init_flat[2*j+1] - X_init_flat[2*i+1]])
                    v_prop = np.array([X_proposed[2*j] - X_proposed[2*i], X_proposed[2*j+1] - X_proposed[2*i+1]])
                    if np.dot(v_init, v_prop) <= 1e-5:
                        is_valid_topology = False
                        break
                
                if is_valid_topology:
                    M, b, current_rank = M_test, b_test, test_rank
                    applied_constraints.append(cand)
                    m_counts[cand['face_idx']] += 1

    print(f"Final Matrix Rank: {current_rank} / {target_rank}")
    
    delta_X_final = np.linalg.pinv(M) @ (b - M @ X_init_flat)
    X_final = X_init_flat + delta_X_final
    
    pos_solved = {}
    for u in nodes:
        idx = n2i[u]
        pos_solved[u] = (X_final[2*idx], X_final[2*idx+1])
        
    return G, pos_init, pos_solved, faces, applied_constraints

# =============================================================================
# 6. PLOTTING
# =============================================================================

def draw_before_ax(ax, G, pos_init, faces, applied_constraints, title):
    ax.set_title(title, fontsize=14)
    for u, v in G.edges():
        ax.plot([pos_init[u][0], pos_init[v][0]], [pos_init[u][1], pos_init[v][1]], 
                 'k-', lw=1.5, zorder=2, alpha=0.3)
                 
    for face in faces:
        lines = get_py_skeleton_lines(face, pos_init)
        for p1, p2 in lines:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', lw=1.5, alpha=0.5, zorder=1)

    colors = cm.get_cmap('tab20', max(1, len(applied_constraints)))
    epsilon = -0.008 
    
    for idx, candidate in enumerate(applied_constraints):
        color = colors(idx)
        p1, p2 = candidate['skel_edge']
        is_point = math.isclose(p1[0], p2[0], abs_tol=1e-5) and math.isclose(p1[1], p2[1], abs_tol=1e-5)
        
        if is_point:
            ax.plot(p1[0], p1[1], marker='X', color=color, markersize=10, zorder=5)
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=5.0, zorder=5)
        
        mid_x = (p1[0] + p2[0]) / 2.0
        mid_y = (p1[1] + p2[1]) / 2.0
        
        for u, v in candidate['edges']:
            dx, dy = pos_init[v][0] - pos_init[u][0], pos_init[v][1] - pos_init[u][1]
            L = math.hypot(dx, dy)
            if L > 1e-7: nx, ny = -dy/L, dx/L
            else: nx, ny = 0, 0
            
            u_shifted = (pos_init[u][0] + nx * epsilon, pos_init[u][1] + ny * epsilon)
            v_shifted = (pos_init[v][0] + nx * epsilon, pos_init[v][1] + ny * epsilon)
            ax.plot([u_shifted[0], v_shifted[0]], [u_shifted[1], v_shifted[1]], 
                     color=color, lw=3.0, zorder=4, solid_capstyle='round')
                     
            e_mid_x = (pos_init[u][0] + pos_init[v][0]) / 2.0
            e_mid_y = (pos_init[u][1] + pos_init[v][1]) / 2.0
            ax.plot([mid_x, e_mid_x], [mid_y, e_mid_y], color=color, linestyle=':', lw=1.5, zorder=4, alpha=0.8)

    ax.set_aspect('equal')
    ax.axis('off')

def draw_after_ax(ax, G, pos_solved, faces, title):
    ax.set_title(title, fontsize=14)
    for u, v in G.edges():
        ax.plot([pos_solved[u][0], pos_solved[v][0]], [pos_solved[u][1], pos_solved[v][1]], 
                 'k-', lw=2, zorder=2, alpha=0.7)
        
    for u in G.nodes():
        ax.plot(pos_solved[u][0], pos_solved[u][1], 'ko', markersize=4, zorder=3)

    for face in faces:
        lines = get_py_skeleton_lines(face, pos_solved)
        for p1, p2 in lines:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', lw=1.5, alpha=0.8, zorder=1)
            
    ax.set_aspect('equal')
    ax.axis('off')

def plot_before_after(G, pos_init, pos_solved, faces, applied_constraints, filename=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    draw_before_ax(ax1, G, pos_init, faces, applied_constraints, "Before (pos_init) + Collapsed Skeleton Edges")
    draw_after_ax(ax2, G, pos_solved, faces, "After (pos_solved)")
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=200)
    plt.show()

def plot_multiple_before_after(results, filename=None):
    """
    Plots a grid of 'Before' and 'After' pairs for multiple solved topologies.
    Aims for a roughly 3:2 rectangular aspect ratio.
    """
    if not results: 
        return
        
    n_results = len(results)
    
    # To get a 6:4 aspect ratio, we want cols ~= 1.5 * rows
    # Since n = cols * rows, then n = 1.5 * rows^2 -> rows = sqrt(n / 1.5)
    n_rows = max(1, int(np.round(np.sqrt(n_results / 1.5))))
    n_cols = int(np.ceil(n_results / n_rows))
    
    # Each 'result' entry needs two subplots (Before and After)
    # So we create a figure with n_rows and n_cols * 2
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(8 * n_cols, 4 * n_rows))
    
    # Flatten axes for easy indexing
    if n_results == 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = axes.flatten()

    for i in range(n_results):
        G, pos_init, pos_solved, faces, applied_constraints = results[i]
        
        # Calculate the base index for the pair in the flattened grid
        idx_before = i * 2
        idx_after = i * 2 + 1
        
        ax_before = axes_flat[idx_before]
        ax_after = axes_flat[idx_after]
        
        # Use existing drawing helpers
        draw_before_ax(ax_before, G, pos_init, faces, applied_constraints, f"Graph {i+1}: Before")
        draw_after_ax(ax_after, G, pos_solved, faces, f"Graph {i+1}: After")
        
    # Hide any remaining empty axes in the grid
    for j in range(n_results * 2, len(axes_flat)):
        axes_flat[j].axis('off')
        
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=200)
    plt.show()

# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    
    solved_results = []
    for i in random.sample(range(1, 9000), 10):
        try:
            G_raw = extract_topology(i, db_name="topologies_4_diag.db", N=4)
            G_solved, pos_init, pos_solved, faces, applied_constraints = solve_absolute_positions(G_raw, symmetry='diag', N=4)
            print(f"ID {i}: Applied {len(applied_constraints)} constraints.")
            solved_results.append((G_solved, pos_init, pos_solved, faces, applied_constraints))
        except Exception as e:
            print(f"Failed to solve topology {i}: {e}")
            
    plot_multiple_before_after(solved_results, filename="renders/debug_batch_diag.png")