"""
Core Tiling Module: Topology -> Exact 4D Coordinates -> Frozen Blob
Trimmed for speed and database integration. 
Downstream processing (straight skeletons, crease patterns) will be handled in a separate module.
"""

from database.tilings.build_topologies import extract_topology
from src.engine.math225_core import Vertex4D, Fraction
from py_straight_skeleton import compute_skeleton

import numpy as np
import networkx as nx
import math
import pulp
from itertools import combinations

# =============================================================================
# CONFIGURATION & TUNING (Easily Accessible)
# =============================================================================
EPSILON = 1/4 # Minimum feature size as a fraction of 1/N.
C_scale = 10  # for big-M constraints in MILP, scaled by N. Should be larger than max possible incircle radius, but not too large to cause numerical instability.
class MILPTuning:
    """
    Adjust these parameters to tune how the MILP prioritizes quadruplet selection.
    """
    CONCAVE_WEIGHT = 5.0
    BASE_WEIGHT = 1.0
    
    @staticmethod
    def penalty_func(poly_size):
        """
        Deprioritize quadruplets from large faces, which have multiple redundant 
        quadruplets, pushing the solver to lock down smaller faces first.
        """
        return 0.5 / max(1.0, poly_size - 3.0)


# =============================================================================
# 1. GRAPH CLEANUP & TOPOLOGY
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
# 2. BASIC GEOMETRIC CONSTRAINTS (4D)
# =============================================================================

ANGLE_TO_4D = {
    0: [1, 0, 0, 0], 2: [0, 1, 0, 0], 4: [0, 0, 1, 0], 6: [0, 0, 0, 1],
    8: [-1, 0, 0, 0], 10: [0, -1, 0, 0], 12: [0, 0, -1, 0], 14: [0, 0, 0, -1]
}

def get_angle(nx, ny):
    return int(round(math.atan2(ny, nx) / (math.pi/8))) % 16

def scale_sqrt2(vec):
    x, y, z, w = vec
    return [y - w, x + z, w + y, z - x]

def build_angle_constraints_4d(G, pos, n2i):
    M, b = [], []
    for u, v in G.edges():
        i, j = 4*n2i[u], 4*n2i[v]
        x1,y1,z1,w1 = i, i+1, i+2, i+3
        x2,y2,z2,w2 = j, j+1, j+2, j+3
        dx, dy = pos[v][0] - pos[u][0], pos[v][1] - pos[u][1]
        L = math.hypot(dx, dy)
        angle = get_angle(-dy/L, dx/L)

        if angle in {4,12}: M.extend([{z1:1,z2:-1}, {y1:1,w1:1,y2:-1,w2:-1}])
        elif angle in {6,14}: M.extend([{w1:1,w2:-1}, {z1:1,x1:-1,z2:-1,x2:1}])
        elif angle in {0, 8}: M.extend([{x1:1,x2:-1}, {y1:1,w1:-1,y2:-1,w2:1}])
        elif angle in {2,10}: M.extend([{y1:1,y2:-1}, {x1:1,z1:1,x2:-1,z2:-1}])
        b.extend([0, 0])
    return M, b

def build_symmetry_constraints_4d(nodes, n2i, symmetry, N):
    M, b = [], []
    if symmetry == 'none': return M, b
    for u in nodes:
        i = 4*n2i[u]
        u_sym = (u[1], u[0]) if symmetry == 'diag' else (N - u[0], u[1])
        if u_sym in n2i:
            j = 4*n2i[u_sym]
            x1,y1,z1,w1 = i, i+1, i+2, i+3
            x2,y2,z2,w2 = j, j+1, j+2, j+3
            if i<j: 
                if symmetry == 'diag':
                    M.extend([{x1:1,z2:-1},{z1:1,x2:-1},{w1:1,w2:1},{y1:1,y2:-1}])
                    b.extend([0, 0, 0, 0])
                elif symmetry == 'book':
                    M.extend([{x1:1,x2:1},{z1:1,z2:-1},{y1:1,w2:-1},{w1:1,y2:-1}])
                    b.extend([1, 0, 0, 0])
            elif i==j: 
                if symmetry == 'diag':
                    M.extend([{x1:1,z1:-1},{w1:1}]); b.extend([0, 0])
                elif symmetry == 'book':
                    M.extend([{x1:2},{y1:1,w1:-1}]); b.extend([1, 0])
    return M, b

def build_boundary_constraints_4d(n2i, N):
    M, b = [], []
    if (0,0) in n2i:
        i = 4*n2i[(0,0)]
        M.extend([{i: 1}, {i+1: 1}, {i+2: 1}, {i+3: 1}])
        b.extend([0, 0, 0, 0])
    if (N,N) in n2i:
        j = 4*n2i[(N,N)]
        M.extend([{j: 1}, {j+1: 1}, {j+2: 1}, {j+3: 1}])
        b.extend([1, 0, 1, 0])
    return M, b

def build_quadruplet_constraint_4d(edges, n2i):
    import sympy as sp
    A_combined = []
    for edge in edges: A_combined.append(ANGLE_TO_4D[edge["angle"]] + [-1, 0])
    for edge in edges: A_combined.append(scale_sqrt2(ANGLE_TO_4D[edge["angle"]]) + [0, -1])
        
    null_basis = sp.Matrix(A_combined).T.nullspace()
    M_rows = []
    for sp_vec in null_basis:
        w_frac = [val for val in sp_vec]
        lcm = 1
        for val in w_frac:
            if val.q != 1: lcm = abs(lcm * val.q) // math.gcd(lcm, val.q)
        w_int = [int(val.p * lcm / val.q) for val in w_frac]
        g = 0
        for val in w_int: g = math.gcd(g, abs(val))
        if g > 0: w_int = [val // g for val in w_int]
        
        constraint = {}
        for local_idx, edge in enumerate(edges):
            idx = 4 * n2i[edge["u"]]
            coeffs = w_int[local_idx] * np.array(ANGLE_TO_4D[edge["angle"]]) + w_int[local_idx + 4] * np.array(scale_sqrt2(ANGLE_TO_4D[edge["angle"]]))
            for c in range(4):
                if coeffs[c] != 0:
                    constraint[idx + c] = constraint.get(idx + c, 0) + coeffs[c]
        if constraint: M_rows.append(constraint)
    return M_rows, [0] * len(M_rows)

# =============================================================================
# 3. QUADRUPLET HARVESTER
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
        edge_data.append({'e': (u, v), 'u': u, 'v': v, 'n': (nx, ny), 'eta': eta, 'pu': p_u, 'pv': p_v, 'angle': get_angle(nx, ny)})
    return edge_data

def ray_segment_intersect(O, D, A, B):
    x1, y1 = O
    x2, y2 = O[0] + D[0], O[1] + D[1]
    x3, y3 = A
    x4, y4 = B
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-7: return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / den
    if t > 1e-5 and -1e-5 <= u <= 1.0 + 1e-5: return t
    return None

def is_boundary_edge(ed, N):
    x1, y1, x2, y2 = ed['pu'][0], ed['pu'][1], ed['pv'][0], ed['pv'][1]
    return (math.isclose(x1, 0, abs_tol=1e-5) and math.isclose(x2, 0, abs_tol=1e-5)) or \
           (math.isclose(x1, N, abs_tol=1e-5) and math.isclose(x2, N, abs_tol=1e-5)) or \
           (math.isclose(y1, 0, abs_tol=1e-5) and math.isclose(y2, 0, abs_tol=1e-5)) or \
           (math.isclose(y1, N, abs_tol=1e-5) and math.isclose(y2, N, abs_tol=1e-5))

def harvest_candidates(faces, pos_init, symmetry, N=4):
    candidates = []
    for face_idx, face in enumerate(faces):
        k = len(face)
        if k < 4: continue

        area = sum(pos_init[face[i]][0] * pos_init[face[(i+1)%k]][1] - pos_init[face[(i+1)%k]][0] * pos_init[face[i]][1] for i in range(k))
        if area > -1e-5: continue
        
        if symmetry == 'diag' and not any(pos_init[v][1] < pos_init[v][0] - 1e-5 for v in face): continue
        elif symmetry == 'book' and not any(pos_init[v][0] > N/2 + 1e-5 for v in face): continue
                
        edge_data = get_edge_data(face, pos_init)
        quad_indices = set()
        reflex_pairs = set()
        
        for i in range(k):
            quad_indices.add(tuple(sorted([i, (i+1)%k, (i+2)%k, (i+3)%k])))
            if k >= 5:
                quad_indices.add(tuple(sorted([i, (i+1)%k, (i+2)%k, (i+4)%k])))
                quad_indices.add(tuple(sorted([i, (i+1)%k, (i+3)%k, (i+4)%k])))
                quad_indices.add(tuple(sorted([i, (i+2)%k, (i+3)%k, (i+4)%k])))
                
            u_id, v_id, w_id = face[i-1], face[i], face[(i+1)%k]
            dx1, dy1 = pos_init[v_id][0] - pos_init[u_id][0], pos_init[v_id][1] - pos_init[u_id][1]
            dx2, dy2 = pos_init[w_id][0] - pos_init[v_id][0], pos_init[w_id][1] - pos_init[v_id][1]
            if (dx1 * dy2 - dy1 * dx2) > -1e-5: 
                reflex_pairs.add(tuple(sorted([(i-1)%k, i])))
                
        for r_pair in reflex_pairs:
            idx1, idx2 = r_pair
            n1, n2 = edge_data[idx1]['n'], edge_data[idx2]['n']
            Dx, Dy = -n1[0] - n2[0], -n1[1] - n2[1]
            mag = math.hypot(Dx, Dy)
            if mag < 1e-5: continue
            Dx, Dy = Dx/mag, Dy/mag
            p_v = pos_init[face[max(idx1, idx2)]] if abs(idx1 - idx2) == 1 else pos_init[face[0]] 
            
            min_t, hit_j = float('inf'), -1
            for j in range(k):
                if j == idx1 or j == idx2: continue 
                t = ray_segment_intersect(p_v, (Dx, Dy), edge_data[j]['pu'], edge_data[j]['pv'])
                if t is not None and t < min_t: min_t, hit_j = t, j
            if hit_j != -1:
                quad_indices.add(tuple(sorted([idx1, idx2, (hit_j-1)%k, hit_j])))
                quad_indices.add(tuple(sorted([idx1, idx2, hit_j, (hit_j+1)%k])))
                    
        for indices in quad_indices:
            combo = [edge_data[idx] for idx in indices]
            if all(is_boundary_edge(ed, N) for ed in combo): continue
            A, eta_init = [], []
            for ed in combo:
                A.append([ed['n'][0], ed['n'][1], -1])
                eta_init.append(ed['eta'])
                
            try:
                xi, _, _, _ = np.linalg.lstsq(np.array(A), np.array(eta_init), rcond=None)
                Px, Py, r = xi[0], xi[1], xi[2]
            except np.linalg.LinAlgError:
                continue
                
            if abs(r) > 2*N or not (-N <= Px <= 2*N) or not (-N <= Py <= 2*N): continue
                
            has_reflex = any(tuple(sorted([indices[a], indices[b]])) in reflex_pairs for a in range(4) for b in range(a+1, 4))
            candidates.append({
                'face_idx': face_idx, 'edges': combo, 'edge_indices': indices, 
                'P': (Px, Py), 'has_reflex': has_reflex, 'poly_size': k,
                'norm_dist': abs(r) # Used for sorting unused candidates in fallback
            })
    return candidates

# =============================================================================
# 4. MILP CONSTRAINT SELECTION
# =============================================================================

def get_safe_invader_edges(quad_indices, reflex_vertices, k):
    safe_edges = set(quad_indices)
    for q in quad_indices:
        curr = q
        while (curr + 1) % k not in reflex_vertices:
            curr = (curr + 1) % k
            if curr in safe_edges: break 
            safe_edges.add(curr)
        curr = q
        while curr not in reflex_vertices:
            curr = (curr - 1) % k
            if curr in safe_edges: break
            safe_edges.add(curr)
    return safe_edges

def run_milp_selection(G, pos_init, nodes, faces, all_candidates, symmetry, N):
    epsilon = EPSILON / N
    C = C_scale * N
    prob = pulp.LpProblem("Skeleton_Quadruplets", pulp.LpMaximize)

    x_vars = {u: pulp.LpVariable(f"x_{u}", lowBound=-N, upBound=2*N) for u in nodes}
    y_vars = {u: pulp.LpVariable(f"y_{u}", lowBound=-N, upBound=2*N) for u in nodes}
    z_vars, P_vars, r_vars = [], [], []
    
    for k in range(len(all_candidates)):
        z_vars.append(pulp.LpVariable(f"z_{k}", cat=pulp.LpBinary))
        P_vars.append((pulp.LpVariable(f"Px_{k}", lowBound=-N, upBound=2*N), pulp.LpVariable(f"Py_{k}", lowBound=-N, upBound=2*N)))
        r_vars.append(pulp.LpVariable(f"r_{k}", lowBound=0, upBound=N))

    objective_terms = []
    for k, cand in enumerate(all_candidates):
        base_weight = MILPTuning.CONCAVE_WEIGHT if cand['has_reflex'] else MILPTuning.BASE_WEIGHT
        final_weight = base_weight * MILPTuning.penalty_func(cand['poly_size'])
        objective_terms.append(final_weight * z_vars[k])
    prob += pulp.lpSum(objective_terms)

    for u, v in G.edges():
        dx, dy = pos_init[v][0] - pos_init[u][0], pos_init[v][1] - pos_init[u][1]
        if math.isclose(dy, 0, abs_tol=1e-5): prob += y_vars[u] == y_vars[v]
        elif math.isclose(dx, 0, abs_tol=1e-5): prob += x_vars[u] == x_vars[v]
        elif math.isclose(dy, dx, abs_tol=1e-5): prob += y_vars[v] - y_vars[u] == x_vars[v] - x_vars[u]
        elif math.isclose(dy, -dx, abs_tol=1e-5): prob += y_vars[v] - y_vars[u] == -(x_vars[v] - x_vars[u])

        L = math.hypot(dx, dy)
        prob += (x_vars[v] - x_vars[u])*(dx/L) + (y_vars[v] - y_vars[u])*(dy/L) >= epsilon

    if symmetry != 'none':
        for u in nodes:
            u_sym = (u[1], u[0]) if symmetry == 'diag' else (N - u[0], u[1])
            if u_sym in nodes:
                if symmetry == 'diag':
                    prob += x_vars[u] == y_vars[u_sym]
                    prob += y_vars[u] == x_vars[u_sym]
                elif symmetry == 'book':
                    prob += x_vars[u] == N - x_vars[u_sym]
                    prob += y_vars[u] == y_vars[u_sym]

    if (0,0) in nodes: prob += x_vars[(0,0)] == 0; prob += y_vars[(0,0)] == 0
    if (N,N) in nodes: prob += x_vars[(N,N)] == N; prob += y_vars[(N,N)] == N

    face_reflex_verts = {}
    for face_idx, face in enumerate(faces):
        k = len(face)
        reflex_set = set()
        for i in range(k):
            u_id, v_id, w_id = face[i-1], face[i], face[(i+1)%k]
            dx1, dy1 = pos_init[v_id][0] - pos_init[u_id][0], pos_init[v_id][1] - pos_init[u_id][1]
            dx2, dy2 = pos_init[w_id][0] - pos_init[v_id][0], pos_init[w_id][1] - pos_init[v_id][1]
            if (dx1 * dy2 - dy1 * dx2) > -1e-5: reflex_set.add(i)
        face_reflex_verts[face_idx] = reflex_set

    for k, cand in enumerate(all_candidates):
        z, Px, Py, r = z_vars[k], P_vars[k][0], P_vars[k][1], r_vars[k]
        face_idx, face, poly_size = cand['face_idx'], faces[cand['face_idx']], cand['poly_size']
        
        for edge in cand['edges']:
            u = edge['u']
            nx, ny = edge['n'] 
            expr = nx*Px + ny*Py - nx*x_vars[u] - ny*y_vars[u] + r
            prob += expr <= C*(1 - z)
            prob += expr >= -C*(1 - z)

        if poly_size > 4:
            safe_edge_indices = get_safe_invader_edges(cand['edge_indices'], face_reflex_verts[face_idx], poly_size)
            face_edges = [(face[i], face[(i+1)%poly_size]) for i in range(poly_size)]
            for i, (u_f, v_f) in enumerate(face_edges):
                if i in safe_edge_indices and i not in cand['edge_indices']:
                    dx, dy = pos_init[v_f][0] - pos_init[u_f][0], pos_init[v_f][1] - pos_init[u_f][1]
                    L = math.hypot(dx, dy)
                    nx, ny = -dy/L, dx/L
                    expr = -nx*Px - ny*Py + nx*x_vars[u_f] + ny*y_vars[u_f] - r
                    prob += expr >= -C*(1 - z)

    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=60))
    active_candidates = []
    if prob.status == pulp.LpStatusOptimal:
        for k, cand in enumerate(all_candidates):
            if pulp.value(z_vars[k]) is not None and pulp.value(z_vars[k]) > 0.5:
                active_candidates.append(cand)
    return active_candidates


# =============================================================================
# 5. EXACT SOLVER
# =============================================================================

def build_dense(M_list, num_vars):
    arr = np.zeros((len(M_list), num_vars))
    for r, d in enumerate(M_list):
        for c, v in d.items(): arr[r, c] = v
    return arr

def exact_fraction_solve(M_list, b_list, num_vars):
    mat = [[Fraction(0) for _ in range(num_vars + 1)] for _ in range(len(M_list))]
    for r, row_dict in enumerate(M_list):
        for c, coef in row_dict.items(): mat[r][c] = Fraction(int(coef))
        mat[r][-1] = Fraction(int(b_list[r]) if isinstance(b_list[r], (int, float)) and float(b_list[r]).is_integer() else b_list[r])
        
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

# ==== Main function ====
def solve_tiling(G_in, symmetry='none', N=4):
    """
    Main entry point for pipeline. 
    1. Extracts topology and candidates.
    2. Runs MILP.
    3. Triggers greedy fallback if underconstrained.
    4. Exact solves and returns the serializable Graph and positions.
    """
    G, pos_init, nodes, n2i = clean_deg2_vertices(G_in, N)
    n = len(nodes)
    num_vars = 4 * n 
    
    M_ang, b_ang = build_angle_constraints_4d(G, pos_init, n2i)
    M_sym, b_sym = build_symmetry_constraints_4d(nodes, n2i, symmetry, N)
    M_bnd, b_bnd = build_boundary_constraints_4d(n2i, N)
    M_list = M_ang + M_sym + M_bnd
    b_list = b_ang + b_sym + b_bnd
    
    faces = extract_oriented_faces(G, pos_init)
    all_candidates = harvest_candidates(faces, pos_init, symmetry)
    applied = run_milp_selection(G, pos_init, nodes, faces, all_candidates, symmetry, N)
    
    # Assembly
    for cand in applied:
        M_eq, b_eq = build_quadruplet_constraint_4d(cand['edges'], n2i)
        if M_eq: 
            M_list += M_eq
            b_list += b_eq

    M_dense = build_dense(M_list, num_vars)
    current_rank = np.linalg.matrix_rank(M_dense, tol=1e-7)

    # -------------------------------------------------------------------------
    # GREEDY FALLBACK: Exhaust unused candidates if MILP underconstrained system
    # -------------------------------------------------------------------------
    if current_rank < num_vars:
        unused = [c for c in all_candidates if c not in applied]
        unused.sort(key=lambda c: c['norm_dist']) # Prioritize physically tighter incircles
        
        for cand in unused:
            M_eq, b_eq = build_quadruplet_constraint_4d(cand['edges'], n2i)
            if not M_eq: continue
            
            M_test = M_list + M_eq
            M_test_dense = build_dense(M_test, num_vars)
            new_rank = np.linalg.matrix_rank(M_test_dense, tol=1e-7)
            
            if new_rank > current_rank:
                M_list = M_test
                b_list += b_eq
                current_rank = new_rank
                applied.append(cand)
                if current_rank == num_vars:
                    break

    # Graceful Crash if all quadruplets mathematically exhausted
    if current_rank < num_vars:
        raise ValueError(f"Exhausted all quadruplets. System inherently underconstrained: Rank {current_rank} / {num_vars}")

    # Exact Solve
    ans = exact_fraction_solve(M_list, b_list, num_vars)
    
    pos_solved_exact = {}
    for i, u in enumerate(nodes):
        pos_solved_exact[u] = Vertex4D(ans[4*i], ans[4*i+1], ans[4*i+2], ans[4*i+3])
        
    return G, pos_init, pos_solved_exact, faces, n2i


# =============================================================================
# 6. FREEZE & EXPORT
# =============================================================================

def export_frozen_blob(G, pos_solved_exact, n2i, faces):
    """
    Serializes the exact algebraic state and graph topology into a flat dictionary 
    suitable for long-term database storage.
    """
    blob = {
        "vertices": {n2i[u]: (u[0], u[1]) for u in G.nodes()},
        "edges": [(n2i[u], n2i[v]) for u, v in G.edges()],
        "pos_4d": {},
        "faces": [[n2i[u] for u in face] for face in faces]
    }
    
    # Store Fractions explicitly
    for u, v4d in pos_solved_exact.items():
        blob["pos_4d"][n2i[u]] = (
            v4d.x.num, v4d.x.den,
            v4d.y.num, v4d.y.den,
            v4d.z.num, v4d.z.den,
            v4d.w.num, v4d.w.den
        )
        
    return blob


# =============================================================================
# 7. DEBUG & VISUALIZATION
# =============================================================================

if __name__ == "__main__":
    import random
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    def draw_debug_ax(ax, G, pos_init, pos_solved_exact, title):
        ax.set_title(title, fontsize=14)
        
        # Float conversion just for debug plotting
        S2 = math.sqrt(2) / 2.0
        pos_float = {}
        for u, v_ex in pos_solved_exact.items():
            pos_float[u] = (float(v_ex.x) + S2*(float(v_ex.y)-float(v_ex.w)), 
                            float(v_ex.z) + S2*(float(v_ex.y)+float(v_ex.w)))

        # Plot original lightly
        for u, v in G.edges():
            ax.plot([pos_init[u][0], pos_init[v][0]], [pos_init[u][1], pos_init[v][1]], 'k-', lw=1.5, alpha=0.2)
            
        # Plot Solved Constraints
        for u, v in G.edges():
            ax.plot([pos_float[u][0], pos_float[v][0]], [pos_float[u][1], pos_float[v][1]], 'b-', lw=2, zorder=2, alpha=0.)
            
        for u in G.nodes():
            ax.plot(pos_float[u][0], pos_float[u][1], 'ko', markersize=4, zorder=3)

        ax.set_aspect('equal')
        ax.axis('off')

    # Example batch run
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes_flat = axes.flatten()
    for i, db_id in enumerate([483,4613]):
    # for i, db_id in enumerate(random.sample(range(1, 9000), 10)):
        try:
            G_raw = extract_topology(db_id, db_name="topologies_4_none.db", N=4)
            G, pos_init, pos_solved_exact, faces, n2i = solve_tiling(G_raw, symmetry='none', N=4)
            blob = export_frozen_blob(G, pos_solved_exact, n2i, faces)
            
            print(f"ID {db_id}: Successfully frozen.")
            draw_debug_ax(axes_flat[i], G, pos_init, pos_solved_exact, f"ID {db_id} (Solved)")
        except Exception as e:
            print(f"Failed to solve topology {db_id}: {e}")
            axes_flat[i].set_title(f"ID {db_id} (Failed)", color='red')
            axes_flat[i].axis('off')
            
    plt.tight_layout()
    plt.show()


    "There is an issue that needs another constraint added to the MILP. sometimes the mouth of a concave polygon clips through the other end, without violating any edge collapse constraints. To solve this, a constraint needs to be added for every concave vertex in every polygon. For every other edge  in the polygon (not including the two edges connected to the concave vertex), the vertex needs to be behind it (defined by the other edge's outward normal). Also, when the vertex is projected onto the other edge, it actually needs to land in the shadow. how would this be formulated mathematically"