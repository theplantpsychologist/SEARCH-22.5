"""
Load planar graph from database.
Map initial positions, construct constraints in an exact 4D basis:
Basis: (1,0), (s,s), (0,1), (-s,s) where s = sqrt(2)/2.
Variables per node: x, y, z, w.
Solve exactly over Rational fractions when full rank.
Output Cp225 object.
"""

from database.tilings.build_topologies import extract_topology
from src.engine.math225_core import Vertex4D, Fraction, AplusBsqrt2
from src.engine.cp225 import Cp225, intersection
from py_straight_skeleton import compute_skeleton

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sympy as sp
import math
import random
from scipy.linalg import null_space
from itertools import combinations
import pulp
import cProfile
import pstats


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
                    # M.extend([{x1:1,z2:-1},{z1:1,x2:-1},{w1:1,w2:-1},{y1:1,y2:-1}])
                    M.extend([{x1:1,z2:-1},{z1:1,x2:-1},{w1:1,w2:1},{y1:1,y2:-1}])
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

def scale_sqrt2(vec):
    """
    Scale a 4d vector by sqrt 2 by swapping x for y-w, y for x + z, etc
    """
    x, y, z, w = vec
    return [y - w, x + z, w + y, z - x]

def build_quadruplet_constraint_4d(edges, n2i):
    """
    Constrain 4 edges to have an equidistant incircle center. 
    Returns constraint in sparse form (list of dicts).
    """
    A_combined = []
    
    # First half of rows rows: Rational equations
    for edge in edges:
        normal = ANGLE_TO_4D[edge["angle"]]
        A_combined.append(normal + [-1, 0])
        
    # Second half of rows: Irrational equations
    for edge in edges:
        normal = ANGLE_TO_4D[edge["angle"]]
        scaled_normal = scale_sqrt2(normal)
        A_combined.append(scaled_normal + [0, -1])
        
    # 1. Use SymPy for Exact Rational Null Space
    M_sympy = sp.Matrix(A_combined).T
    null_basis = M_sympy.nullspace()
    
    M_rows = []
    for sp_vec in null_basis:
        # 2. Extract the exact rational fractions
        w_frac = [val for val in sp_vec]
        
        # 3. Find the Least Common Multiple (LCM) of all denominators
        lcm = 1
        for val in w_frac:
            if val.q != 1: # .q is the denominator in SymPy rationals
                lcm = abs(lcm * val.q) // math.gcd(lcm, val.q)
                
        # 4. Scale the vector up to pure integers
        w_int = [int(val.p * lcm / val.q) for val in w_frac] # .p is numerator
        
        # 5. Divide out the Greatest Common Divisor (GCD) to keep numbers small
        g = 0
        for val in w_int:
            g = math.gcd(g, abs(val))
        if g > 0:
            w_int = [val // g for val in w_int]
        
        # 6. Map back to the sparse constraint dictionary
        constraint = {}
        for local_idx, edge in enumerate(edges):
            u = edge["u"]
            idx = 4 * n2i[u]
            
            w_rat = w_int[local_idx]
            w_irr = w_int[local_idx + 4]
            
            normal = np.array(ANGLE_TO_4D[edge["angle"]])
            scaled = np.array(scale_sqrt2(ANGLE_TO_4D[edge["angle"]]))
            
            coeffs = w_rat * normal + w_irr * scaled
            
            for c in range(4):
                if coeffs[c] != 0:
                    constraint[idx + c] = constraint.get(idx + c, 0) + coeffs[c]
                    
        if constraint:
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

def is_boundary_edge(ed, N):
    """Returns True if the edge lies entirely on the N x N bounding box."""
    x1, y1 = ed['pu']
    x2, y2 = ed['pv']
    return (math.isclose(x1, 0, abs_tol=1e-5) and math.isclose(x2, 0, abs_tol=1e-5)) or \
           (math.isclose(x1, N, abs_tol=1e-5) and math.isclose(x2, N, abs_tol=1e-5)) or \
           (math.isclose(y1, 0, abs_tol=1e-5) and math.isclose(y2, 0, abs_tol=1e-5)) or \
           (math.isclose(y1, N, abs_tol=1e-5) and math.isclose(y2, N, abs_tol=1e-5))

def compute_skeleton_wrapper(exterior, holes=[]):
    # =========================================================================
    # 1. AGGRESSIVE QUANTIZATION
    # Snap to 5 decimal places to force float-noise degeneracies into exact collisions.
    # =========================================================================
    quant_exterior = [(round(p[0], 3), round(p[1], 3)) for p in exterior]
    
    k = len(quant_exterior)
    area = sum(quant_exterior[i][0] * quant_exterior[(i+1)%k][1] - quant_exterior[(i+1)%k][0] * quant_exterior[i][1] for i in range(k))
    
    if abs(area) / 2.0 > 0.95: 
        return None
    if area < 0: 
        quant_exterior.reverse()
   
    try:
        return compute_skeleton(exterior=exterior, holes=[])
    except Exception as e:
        # If it still crashes here, the geometry has folded into an unresolvable bowtie.
        print(f"Skeleton failed after quantization: {e}")
        return None
def get_py_skeleton_lines(face, pos):
    if len(face) < 3: return []
    exterior = [[float(pos[n][0]), float(pos[n][1])] for n in face]
    skeleton = compute_skeleton_wrapper(exterior=exterior, holes=[])
    if skeleton is None:
        return []
        
    lines = []
    for skv1, skv2 in skeleton.arc_iterator():
        p1 = (float(getattr(skv1.position, 'x', skv1.position[0])), float(getattr(skv1.position, 'y', skv1.position[1])))
        p2 = (float(getattr(skv2.position, 'x', skv2.position[0])), float(getattr(skv2.position, 'y', skv2.position[1])))
        lines.append((p1, p2))
    return lines

# =============================================================================
# HARVESTER HELPERS & GAUNTLET
# =============================================================================

def ray_segment_intersect(O, D, A, B):
    """Finds intersection t for Ray(O, D) and Segment(A, B). Returns t or None."""
    x1, y1 = O
    x2, y2 = O[0] + D[0], O[1] + D[1]
    x3, y3 = A
    x4, y4 = B

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-7:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / den

    # Intersects if t > 0 (forward ray) and 0 <= u <= 1 (within segment bounds)
    if t > 1e-5 and -1e-5 <= u <= 1.0 + 1e-5:
        return t
    return None

# =============================================================================
# MAIN HARVESTER (Updated with Reflex Tagging)
# =============================================================================

def harvest_candidates(faces, pos_init, symmetry, N=4):
    """
    Harvests quadruplets using contiguous windows and reflex raycasts.
    Filters out mirrored polygons and mathematically impossible incircles.
    Tags candidates with reflex presence and polygon size for MILP weighting.
    """
    candidates = []
    
    for face_idx, face in enumerate(faces):
        k = len(face)
        if k < 4: continue

        #reject external face, outer boundary of the square
        area = sum(pos_init[face[i]][0] * pos_init[face[(i+1)%k]][1] - pos_init[face[(i+1)%k]][0] * pos_init[face[i]][1] for i in range(k))
        if area > -1e-5:
            continue
        # 1. Polygon-Level Symmetry Sieve
        if symmetry == 'diag':
            if not any(pos_init[v][1] < pos_init[v][0] - 1e-5 for v in face):
                continue
        elif symmetry == 'book':
            if not any(pos_init[v][0] > N/2 + 1e-5 for v in face):
                continue
                
        edge_data = get_edge_data(face, pos_init)
        quad_indices = set()
        reflex_pairs = set() # Track pairs of edges that form a reflex vertex
        
        # 2. Strategy A: Contiguous Windows & Reflex Detection
        for i in range(k):
            # Standard contiguous 4
            quad_indices.add(tuple(sorted([i, (i+1)%k, (i+2)%k, (i+3)%k])))
            
            # Plus 1 Tolerance
            if k >= 5:
                quad_indices.add(tuple(sorted([i, (i+1)%k, (i+2)%k, (i+4)%k])))
                quad_indices.add(tuple(sorted([i, (i+1)%k, (i+3)%k, (i+4)%k])))
                quad_indices.add(tuple(sorted([i, (i+2)%k, (i+3)%k, (i+4)%k])))
                
            # Detect Reflex Vertices for this face
            u_id, v_id, w_id = face[i-1], face[i], face[(i+1)%k]
            dx1, dy1 = pos_init[v_id][0] - pos_init[u_id][0], pos_init[v_id][1] - pos_init[u_id][1]
            dx2, dy2 = pos_init[w_id][0] - pos_init[v_id][0], pos_init[w_id][1] - pos_init[v_id][1]
            cross = dx1 * dy2 - dy1 * dx2
            
            if cross > -1e-5: 
                reflex_pairs.add(tuple(sorted([(i-1)%k, i])))
                
        # 3. Strategy B: Reflex Raycasts (Targeting Split Events)
        for r_pair in reflex_pairs:
            idx1, idx2 = r_pair
            n1 = edge_data[idx1]['n']
            n2 = edge_data[idx2]['n']
            
            Dx, Dy = -n1[0] - n2[0], -n1[1] - n2[1]
            mag = math.hypot(Dx, Dy)
            if mag < 1e-5: continue
            Dx, Dy = Dx/mag, Dy/mag
            
            p_v = pos_init[face[max(idx1, idx2)]] if abs(idx1 - idx2) == 1 else pos_init[face[0]] # Get exact reflex vertex
            
            min_t, hit_j = float('inf'), -1
            for j in range(k):
                if j == idx1 or j == idx2: continue 
                t = ray_segment_intersect(p_v, (Dx, Dy), edge_data[j]['pu'], edge_data[j]['pv'])
                if t is not None and t < min_t:
                    min_t, hit_j = t, j
                    
            if hit_j != -1:
                quad_indices.add(tuple(sorted([idx1, idx2, (hit_j-1)%k, hit_j])))
                quad_indices.add(tuple(sorted([idx1, idx2, hit_j, (hit_j+1)%k])))
                    
        # 4. The Mathematical Gauntlet & Tagging
        for indices in quad_indices:
            combo = [edge_data[idx] for idx in indices]
            # if not passes_angle_span(combo): continue
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
                
            if abs(r) > 2*N or not (-N <= Px <= 2*N) or not (-N <= Py <= 2*N):
                continue
                
            # Check if this specific quadruplet contains a reflex pair
            has_reflex = any(tuple(sorted([indices[a], indices[b]])) in reflex_pairs 
                             for a in range(4) for b in range(a+1, 4))
                
            candidates.append({
                'face_idx': face_idx,
                'edges': combo,
                'edge_indices': indices, # <--- NEW
                'skel_edge': ((Px, Py), (Px, Py)), 
                'P': (Px, Py),
                'has_reflex': has_reflex,
                'poly_size': k
            })
            
    return candidates
# =============================================================================
# 3.5 MILP solver for optimal quadruplet selection
# =============================================================================

def get_safe_invader_edges(quad_indices, reflex_vertices, k):
    """
    Expands outward from the quadruplet edges until reflex vertices are hit.
    Returns the set of edge indices that belong to the same convex chains.
    """
    safe_edges = set(quad_indices)
    
    for q in quad_indices:
        # Expand forward (increasing index)
        # Vertex (curr + 1) connects edge curr and edge curr+1
        curr = q
        while (curr + 1) % k not in reflex_vertices:
            curr = (curr + 1) % k
            if curr in safe_edges: break # Prevent infinite loops in pure convex shapes
            safe_edges.add(curr)
            
        # Expand backward (decreasing index)
        # Vertex curr connects edge curr-1 and edge curr
        curr = q
        while curr not in reflex_vertices:
            curr = (curr - 1) % k
            if curr in safe_edges: break
            safe_edges.add(curr)
            
    return safe_edges
def run_milp_selection(G, pos_init, nodes, faces, all_candidates, symmetry, N, epsilon=None, C=10):
    """
    Constructs and solves the Mixed-Integer Linear Program to find the optimal 
    valid subset of straight-skeleton quadruplet collapses.
    """
    if epsilon is None:
        epsilon = 1/(4*N) #minimum feature size relative to tiling density
    prob = pulp.LpProblem("Skeleton_Quadruplets", pulp.LpMaximize)

    # ==========================================
    # 1. State Variables
    # ==========================================
    x_vars = {u: pulp.LpVariable(f"x_{u}", lowBound=-N, upBound=2*N) for u in nodes}
    y_vars = {u: pulp.LpVariable(f"y_{u}", lowBound=-N, upBound=2*N) for u in nodes}
    
    z_vars = []
    P_vars = []
    r_vars = []
    
    for k in range(len(all_candidates)):
        z_vars.append(pulp.LpVariable(f"z_{k}", cat=pulp.LpBinary))
        P_vars.append((
            pulp.LpVariable(f"Px_{k}", lowBound=-N, upBound=2*N),
            pulp.LpVariable(f"Py_{k}", lowBound=-N, upBound=2*N)
        ))
        r_vars.append(pulp.LpVariable(f"r_{k}", lowBound=0, upBound=N))

    # ==========================================
    # 2. Objective Function
    # ==========================================
    # prob += pulp.lpSum(z_vars)
    CONCAVE_WEIGHT = 5.0
    
    # Tunable divisor function (safeguarded against division by zero for triangles, 
    # though harvester guarantees k >= 4)
    def penalty_func(n):
        """A weight to deprioritize quadruplets from large faces, which have multiple redundant quadruplets """
        return 0.5/max(1.0, n - 3.0) 

    objective_terms = []
    for k, cand in enumerate(all_candidates):
        base_weight = CONCAVE_WEIGHT if cand['has_reflex'] else 1.0
        
        final_weight = base_weight * penalty_func(cand['poly_size'])
        objective_terms.append(final_weight * z_vars[k])

    prob += pulp.lpSum(objective_terms)
    # ==========================================
    # 3. Base Topological Constraints
    # ==========================================
    # Angle Constraints
    for u, v in G.edges():
        dx = pos_init[v][0] - pos_init[u][0]
        dy = pos_init[v][1] - pos_init[u][1]
        
        if math.isclose(dy, 0, abs_tol=1e-5):     
            prob += y_vars[u] == y_vars[v]
        elif math.isclose(dx, 0, abs_tol=1e-5):   
            prob += x_vars[u] == x_vars[v]
        elif math.isclose(dy, dx, abs_tol=1e-5):  
            prob += y_vars[v] - y_vars[u] == x_vars[v] - x_vars[u]
        elif math.isclose(dy, -dx, abs_tol=1e-5): 
            prob += y_vars[v] - y_vars[u] == -(x_vars[v] - x_vars[u])

    # Symmetry Constraints
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

    # Boundary Constraints
    if (0,0) in nodes:
        prob += x_vars[(0,0)] == 0
        prob += y_vars[(0,0)] == 0
    if (N,N) in nodes:
        prob += x_vars[(N,N)] == N
        prob += y_vars[(N,N)] == N

    # No-Cross / Positive Edge Length
    for u, v in G.edges():
        dx = pos_init[v][0] - pos_init[u][0]
        dy = pos_init[v][1] - pos_init[u][1]
        L = math.hypot(dx, dy)
        nx, ny = dx/L, dy/L
        prob += (x_vars[v] - x_vars[u])*nx + (y_vars[v] - y_vars[u])*ny >= epsilon
    # Pre-calculate reflex vertices for every face to define convex chains
    face_reflex_verts = {}
    for face_idx, face in enumerate(faces):
        k = len(face)
        reflex_set = set()
        for i in range(k):
            u_id, v_id, w_id = face[i-1], face[i], face[(i+1)%k]
            dx1, dy1 = pos_init[v_id][0] - pos_init[u_id][0], pos_init[v_id][1] - pos_init[u_id][1]
            dx2, dy2 = pos_init[w_id][0] - pos_init[v_id][0], pos_init[w_id][1] - pos_init[v_id][1]
            # Cross > 0 means a right turn (concave vertex)
            if (dx1 * dy2 - dy1 * dx2) > -1e-5:
                reflex_set.add(i)
        face_reflex_verts[face_idx] = reflex_set

    # ==========================================
    # 4. Quadruplet Toggles (Big-M)
    # ==========================================
    for k, cand in enumerate(all_candidates):
        z = z_vars[k]
        Px, Py = P_vars[k]
        r = r_vars[k]
        face_idx = cand['face_idx']
        face = faces[face_idx]
        poly_size = cand['poly_size']
        
        # A. Equidistant Constraint Toggles
        for edge in cand['edges']:
            u = edge['u']
            nx, ny = edge['n'] 
            expr = nx*Px + ny*Py - nx*x_vars[u] - ny*y_vars[u] + r
            prob += expr <= C*(1 - z)
            prob += expr >= -C*(1 - z)

        # B. Selective Incircle Invasion Veto (Convex Chains)
        if poly_size > 4:
            # 1. Determine which edges are in the same convex chain(s) as the quadruplet
            reflex_verts = face_reflex_verts[face_idx]
            safe_edge_indices = get_safe_invader_edges(cand['edge_indices'], reflex_verts, poly_size)
            
            face_edges = [(face[i], face[(i+1)%poly_size]) for i in range(poly_size)]
            
            # 2. Only apply the infinite line veto to safe edges not already in the quadruplet
            for i, (u_f, v_f) in enumerate(face_edges):
                if i in safe_edge_indices and i not in cand['edge_indices']:
                    dx = pos_init[v_f][0] - pos_init[u_f][0]
                    dy = pos_init[v_f][1] - pos_init[u_f][1]
                    L = math.hypot(dx, dy)
                    nx, ny = -dy/L, dx/L
                    
                    expr = -nx*Px - ny*Py + nx*x_vars[u_f] + ny*y_vars[u_f] - r
                    prob += expr >= -C*(1 - z)

    # ==========================================
    # 5. Solve & Extract
    # ==========================================
    # Run solver silently, cap at 60 seconds to prevent combinatorial lockup on massive shapes
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=60))
    
    active_candidates = []
    if prob.status == pulp.LpStatusOptimal:
        for k, cand in enumerate(all_candidates):
            if pulp.value(z_vars[k]) is not None and pulp.value(z_vars[k]) > 0.5:
                active_candidates.append(cand)
                
    return active_candidates


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
    num_vars = 4 * n # target rank
    X_init_list = []
    for u in nodes:
        X_init_list.extend([pos_init[u][0], 0, pos_init[u][1], 0])
        
    M_ang, b_ang = build_angle_constraints_4d(G, pos_init, n2i)
    M_sym, b_sym = build_symmetry_constraints_4d(nodes, n2i, symmetry, N)
    M_bnd, b_bnd = build_boundary_constraints_4d(n2i, N)
    M_list = M_ang + M_sym + M_bnd
    b_list = b_ang + b_sym + b_bnd
    
    faces = extract_oriented_faces(G, pos_init)
    
    # =========== MILP Quadruplet Selection ==========

    # 1. Gather all n-choose-4 candidates from eligible polygons
    all_candidates = harvest_candidates(faces, pos_init, symmetry)
        
    print(f"Harvested {len(all_candidates)} candidates for MILP oracle.")
    
    # 2. Run the MILP
    applied = run_milp_selection(G, pos_init, nodes, faces, all_candidates, symmetry, N)
    print(f"MILP activated {len(applied)} valid constraint blocks.")
    
    # 3. Blind Trust Matrix Assembly (Exact 4D)
    for cand in applied:
        M_eq, b_eq = build_quadruplet_constraint_4d(cand['edges'], n2i)
        if M_eq: 
            cand['M_eq'] = M_eq
            cand['b_eq'] = b_eq
            M_list += M_eq
            b_list += b_eq

    # Evaluate the Final Matrix
    M_dense = build_dense(M_list, num_vars)
    current_rank = np.linalg.matrix_rank(M_dense, tol=1e-7)
    print(f"Final Matrix Rank: {current_rank} / {num_vars}")

    # ==========================================================
    # 5. Output Parsing & Exact Solve
    # ==========================================================

    def on_edge(u,v):
        ux,uy = u
        vx,vy = v
        return (ux == 0 and vx == 0) or (ux == N and vx == N) or (uy == 0 and vy == 0) or (uy == N and vy == N)
        
    S2 = math.sqrt(2) / 2.0
    
    if current_rank == num_vars:
        print(f"Matrix Full Rank. Solving Exactly.")
        ans = exact_fraction_solve(M_list, b_list, num_vars)
        
        # Preserve the EXACT algebraic states
        pos_solved_exact = {u: Vertex4D(ans[4*i], ans[4*i+1], ans[4*i+2], ans[4*i+3]) for i, u in enumerate(nodes)}
        
        vertices = [pos_solved_exact[u] for u in nodes]
        cp_edges = [(n2i[u], n2i[v], 'b' if (on_edge(u, v)) else 'v') for u, v in G.edges()]
        cp = Cp225(vertices, cp_edges)
        
        # Build the purely algebraic skeleton
        construct_exact_straight_skeleton(cp, pos_init, pos_solved_exact, faces, n2i)
        
        # Demote to floats for the rest of the plotting pipeline
        pos_solved = {}
        for u in nodes:
            v_ex = pos_solved_exact[u]
            pos_solved[u] = (float(v_ex.x) + S2*(float(v_ex.y)-float(v_ex.w)), float(v_ex.z) + S2*(float(v_ex.y)+float(v_ex.w)))

    else:
        print("Warning: MILP returned underconstrained set. Float displacement solve.")
        M_dense = build_dense(M_list, num_vars)
        delta = np.linalg.lstsq(M_dense, np.array(b_list) - M_dense @ np.array(X_init_list), rcond=None)[0]
        ans = np.array(X_init_list) + delta
        cp = None
        pos_solved = {}
        for i, u in enumerate(nodes):
            x, y, z, w = ans[4*i:4*i+4]
            pos_solved[u] = (x + 0.70710678*(y-w), z + 0.70710678*(y+w))
            
    return G, pos_init, faces, applied, cp, pos_solved

def exact_solve_6x6(M, RHS):
    """Solves a 6x6 exact fractional system using basic Gauss-Jordan."""
    mat = [[M[r][c] for c in range(6)] + [RHS[r]] for r in range(6)]
    for col in range(6):
        pivot = -1
        for i in range(col, 6):
            if mat[i][col].num != 0:
                pivot = i
                break
        if pivot == -1: return None # Singular/Parallel
        mat[col], mat[pivot] = mat[pivot], mat[col]
        inv = Fraction(mat[col][col].den, mat[col][col].num)
        for j in range(col, 7):
            mat[col][j] = mat[col][j] * inv
        for i in range(6):
            if i != col and mat[i][col].num != 0:
                factor = mat[i][col]
                for j in range(col, 7):
                    mat[i][j] = mat[i][j] - factor * mat[col][j]
    return [mat[i][6] for i in range(6)]

def construct_exact_straight_skeleton(cp, pos_init, pos_solved_exact, faces, n2i):
    """
    Computes the straight skeleton using a Pure Algebraic Lower-Envelope algorithm.
    Guarantees mathematically perfect topology without float libraries or wavefront degeneracies.
    """
    basis = [Vertex4D(Fraction(1),Fraction(0),Fraction(0),Fraction(0)), 
             Vertex4D(Fraction(0),Fraction(1),Fraction(0),Fraction(0)), 
             Vertex4D(Fraction(0),Fraction(0),Fraction(1),Fraction(0)), 
             Vertex4D(Fraction(0),Fraction(0),Fraction(0),Fraction(1))]
             
    HALF = Fraction(1, 2)
             
    for face in faces:
        k = len(face)
        edges = []
        for i in range(k):
            u = face[i]
            v = face[(i+1)%k]
            # Use pos_init to get the exact outward normal, immune to final 0-length collapses
            dx = pos_init[v][0] - pos_init[u][0]
            dy = pos_init[v][1] - pos_init[u][1]
            L = math.hypot(dx, dy)
            nx, ny = -dy/L, dx/L
            
            N_arr = ANGLE_TO_4D[get_angle(nx, ny)]
            N_vec = Vertex4D(Fraction(N_arr[0]), Fraction(N_arr[1]), Fraction(N_arr[2]), Fraction(N_arr[3]))
            
            edges.append({
                'u': u, 'v': v, 
                'V': pos_solved_exact[u], # Exact 4D starting point
                'N': N_vec
            })
            
        nodes = []
        
        # 1. Add Exact Boundary Nodes
        for i in range(k):
            P = edges[i]['V']
            t = AplusBsqrt2(Fraction(0))
            S = set()
            for j in range(k):
                dj = (edges[j]['V'] - P).dot_product(edges[j]['N'])
                diff = t - dj
                if diff.sign() == 0:
                    S.add(j)
            nodes.append({'pos': P, 't': t, 'S': S})
            
        # 2. Algebraic Triplet Intersections (The Medial Axis)
        for a, b, c in combinations(range(k), 3):
            M = []
            RHS = []
            for idx in (a, b, c):
                N_vec = edges[idx]['N']
                V_vec = edges[idx]['V']
                rhs_dot = V_vec.dot_product(N_vec)
                
                row_rat = [Fraction(0)]*6
                row_irr = [Fraction(0)]*6
                
                for dim in range(4):
                    dot = basis[dim].dot_product(N_vec)
                    row_rat[dim] = dot.A
                    row_irr[dim] = dot.B
                    
                row_rat[4] = Fraction(1)
                row_irr[5] = Fraction(1)
                
                M.append(row_rat)
                M.append(row_irr)
                RHS.append(rhs_dot.A)
                RHS.append(rhs_dot.B)
                
            ans = exact_solve_6x6(M, RHS)
            if ans is None: continue
            
            P = Vertex4D(ans[0], ans[1], ans[2], ans[3])
            t = AplusBsqrt2(ans[4], ans[5])
            
            if t.sign() < 0: continue # Intersection is in the "past"
            
            # The Lower Envelope Validity Check
            valid = True
            S = set()
            for j in range(k):
                dj = (edges[j]['V'] - P).dot_product(edges[j]['N'])
                diff = t - dj
                if diff.sign() > 0:
                    valid = False
                    break
                elif diff.sign() == 0:
                    S.add(j)
                    
            if valid and len(S) >= 3:
                nodes.append({'pos': P, 't': t, 'S': S})
                
        # 3. Exact 4D Node Deduplication
        unique_nodes = []
        for n in nodes:
            found = False
            for un in unique_nodes:
                if n['pos'] == un['pos']:
                    un['S'].update(n['S']) # Union generating edges for degree-4+ vertices
                    found = True
                    break
            if not found:
                unique_nodes.append(n)
                
        # 4. Add to CP
        node_to_idx = {}
        for n in unique_nodes:
            idx = -1
            for i, v in enumerate(cp.vertices):
                if v == n['pos']:
                    idx = i; break
            if idx == -1:
                cp.vertices.append(n['pos'])
                idx = len(cp.vertices) - 1
            node_to_idx[id(n)] = idx
            
        # Tag Reflex Bounds
        reflex_cp_indices = set()
        for i in range(k):
            u_id, v_id, w_id = face[i-1], face[i], face[(i+1)%k]
            # Use init for reflex to bypass 0-length degeneracies
            dx1, dy1 = pos_init[v_id][0] - pos_init[u_id][0], pos_init[v_id][1] - pos_init[u_id][1]
            dx2, dy2 = pos_init[w_id][0] - pos_init[v_id][0], pos_init[w_id][1] - pos_init[v_id][1]
            if (dx1 * dy2 - dy1 * dx2) > -1e-5:
                reflex_cp_indices.add(n2i[v_id])
                
        # 5. Dual-Graph Topology Assembly
        for a in range(k):
            for b in range(a+1, k):
                # Nodes sharing two generating edges lie on the same bisector
                shared_nodes = [n for n in unique_nodes if a in n['S'] and b in n['S']]
                if len(shared_nodes) >= 2:
                    shared_nodes.sort(key=lambda n: n['t'].__float__())
                    for i in range(len(shared_nodes) - 1):
                        n1, n2 = shared_nodes[i], shared_nodes[i+1]
                        idx1, idx2 = node_to_idx[id(n1)], node_to_idx[id(n2)]
                        if idx1 == idx2: continue
                        
                        # Verify the bisector isn't broken by a third edge (Concave split event check)
                        M_pos = (n1['pos'] + n2['pos']) * HALF
                        M_t = AplusBsqrt2((n1['t'].A + n2['t'].A) * HALF, (n1['t'].B + n2['t'].B) * HALF)
                        
                        valid_midpoint = True
                        for j in range(k):
                            dj = (edges[j]['V'] - M_pos).dot_product(edges[j]['N'])
                            if (M_t - dj).sign() > 0:
                                valid_midpoint = False
                                break
                                
                        if valid_midpoint:
                            edge_exists = False
                            for u_e, v_e, _ in cp.edges:
                                if (u_e == idx1 and v_e == idx2) or (u_e == idx2 and v_e == idx1):
                                    edge_exists = True; break
                            if not edge_exists:
                                line_type = "v" if ((idx1 in reflex_cp_indices) or (idx2 in reflex_cp_indices)) else "m"
                                cp.edges.append((idx1, idx2, line_type))
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

plot_colors = {
    'm': 'red',
    'v': 'blue',
    'b': 'black',

    'ridge': 'red', 
    'axial': 'blue', 
    'hinge': 'grey'
}
def draw_after_ax(ax, G, faces, cp, pos_solved, title):
    ax.set_title(title, fontsize=14)    
    if cp:
        for t, x1, y1, x2, y2 in cp.render():
            ax.plot([x1, x2], [y1, y2], color = plot_colors[t], lw=2, zorder=2, alpha=0.7)
    # for u, v in G.edges():
    #     ax.plot([pos_solved[u][0], pos_solved[v][0]], [pos_solved[u][1], pos_solved[v][1]], 'k-', lw=1.5, zorder=2, alpha=0.3)
    # for face in faces:
    #     for p1, p2 in get_py_skeleton_lines(face, pos_solved):
    #         ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'red', lw=1.5, alpha=0.5, zorder=1)
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
        G, pos_init, faces, applied_constraints, cp, pos_solved = results[i]
        draw_before_ax(axes_flat[i * 2], G, pos_init, faces, applied_constraints, f"Graph {i+1}: Before")
        draw_after_ax(axes_flat[i * 2 + 1], G, faces, cp, pos_solved, f"Graph {i+1}: After")
        
    for j in range(n_res * 2, len(axes_flat)): axes_flat[j].axis('off')
    plt.tight_layout()
    if filename: plt.savefig(filename, dpi=200)
    plt.show()

# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    solved_results = []
    for i in random.sample(range(1, 9000), 4):
        G_raw = extract_topology(i, db_name="topologies_4_none.db", N=4)
        G_solved, pos_init, faces, applied, cp, pos_solved = solve_absolute_positions(G_raw, symmetry='none', N=4)
        print(f"ID {i}: Applied {len(applied)} constraints.\n")
        solved_results.append((G_solved, pos_init,faces, applied, cp, pos_solved))
        
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")  # Sort by cumulative time
    stats.print_stats(10)  # Print the top  functions

    plot_multiple_before_after(solved_results, filename="renders/debug_batch_none_exact.png")

    
"""
TODO: run time speedup
avoid concave bow tie intersections
go back for more quadruplets if still rank deficient
"""