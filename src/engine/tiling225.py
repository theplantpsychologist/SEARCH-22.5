"""
Load planar graph from database

Clean up deg 2 vertices. From this point on, vertices will be assumed to lie in the unit square with floats, rather than grid points

(helper for reindexing vertices into position vector x1,y1,x2,y2,... after cleaning up deg 2 vertices)

generate matrix and homogeneous b (column of zeros) for angle constraints. for horizontal edges, y1 = y2. for vertical edges, x1 = x2. similar for diagonal edges. If using the dot product, make sure the constraint matrix only uses 0 and 1 (do not normalize vectors and end up with sqrts)

generate matrix and homogeneous b (column of zeros) for diag or book symmetry. for diag symmetry, x1 = y2, y1 = x2. for book symmetry, x1 = 1-x2, y1 = 1-y2.

generate matrix and homogeneous b (zeros for the bottom left, ones for the top right vertex) to pin the bottom left vertex to 0,0 and the top right vertex to 1,1. 

Check how many dofs are needed to be removed to fully constrain the system. For now, arbitrarily choose sets of 4 or more sides in polygons with 4 or more sides. Later I will revisit this to choose which specific edges to constrain together. occasionally, due to internal symmetries, some constraints may be redundant, so check the rank again. If there are still dofs left, add more constraints until the system is fully constrained. If there are too many constraints, remove some until the system is fully constrained.

Finally, solve the system by taking the pseudoinverse of the constraint matrix and multiplying by the homogeneous b vector. Output the vertex positions as a new graph with the same topology, and plot, with straight skeletons overlaid for each polygon.

"""


from database.tilings.build_topologies import extract_topology

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
from scipy.linalg import null_space
import random
from itertools import combinations

from ladybug_geometry.geometry2d.pointvector import Point2D
from ladybug_geometry.geometry2d.polygon import Polygon2D
from ladybug_geometry_polyskel.polyskel import skeleton_as_edge_list

# =============================================================================
# 1. GRAPH CLEANUP & METADATA
# =============================================================================

def clean_deg2_vertices(G_in, N):
    """Removes degree-2 vertices that lie on a straight line and scales to unit square."""
    G = G_in.copy()
    # Initial float mapping to unit square
    pos = {n: (n[0]/N, n[1]/N) for n in G.nodes()} 
    
    while True:
        removed_any = False
        deg2_nodes = [n for n in G.nodes() if G.degree(n) == 2]
        
        for node in deg2_nodes:
            nbrs = list(G.neighbors(node))
            if len(nbrs) != 2: continue
            u, v = nbrs
            
            # Cross product to check collinearity
            dx1, dy1 = pos[u][0] - pos[node][0], pos[u][1] - pos[node][1]
            dx2, dy2 = pos[v][0] - pos[node][0], pos[v][1] - pos[node][1]
            cross = dx1*dy2 - dy1*dx2
            
            if math.isclose(cross, 0, abs_tol=1e-7):
                G.remove_node(node)
                G.add_edge(u, v)
                removed_any = True
                break # Graph mutated, restart loop
                
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
    """Extracts faces using CCW angular sorting."""
    adj = {}
    for u, v in G.edges():
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    # Sort neighbors counter-clockwise based on geometric position
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
            w = neighbors[(idx + 1) % len(neighbors)] # CCW turn
            next_he = (v, w)

            if next_he == start_he: break
            if next_he in unvisited_he: unvisited_he.remove(next_he)
            
            face.append(next_he[0])
            curr_he = next_he

        # Filter out the outer bounding box face (it's CW oriented and has largest area)
        faces.append(face)
        
    return faces

# =============================================================================
# 3. CONSTRAINT GENERATORS
# =============================================================================

def build_angle_constraints(G, pos, n2i):
    """Enforces lines to remain Horiz, Vert, or +/- 45 deg diagonals using 0, 1, -1."""
    n = len(n2i)
    M, b = [], []
    for u, v in G.edges():
        i, j = n2i[u], n2i[v]
        dx, dy = pos[v][0] - pos[u][0], pos[v][1] - pos[u][1]
        
        row = [0] * (2*n)
        if math.isclose(dy, 0, abs_tol=1e-5):     # Horizontal
            row[2*i+1] = 1; row[2*j+1] = -1
        elif math.isclose(dx, 0, abs_tol=1e-5):   # Vertical
            row[2*i] = 1; row[2*j] = -1
        elif math.isclose(dy, dx, abs_tol=1e-5):  # Diag 1 (/)
            row[2*i+1] = 1; row[2*i] = -1; row[2*j+1] = -1; row[2*j] = 1
        elif math.isclose(dy, -dx, abs_tol=1e-5): # Diag 2 (\)
            row[2*i+1] = 1; row[2*i] = 1; row[2*j+1] = -1; row[2*j] = -1
            
        if any(row):
            M.append(row); b.append(0)
    return np.array(M), np.array(b)

def build_symmetry_constraints(nodes, pos, n2i, symmetry, N):
    """Links mirrored nodes. Uses N to detect geometric mirror pairs from original indices."""
    n = len(n2i)
    M, b = [], []
    if symmetry == 'none':
        return np.empty((0, 2*n)), np.empty(0)
        
    for u in nodes:
        i = n2i[u]
        # Reflect original integer coordinates
        if symmetry == 'diag':
            u_sym = (u[1], u[0])
        elif symmetry == 'book':
            u_sym = (N - u[0], u[1])
            
        if u_sym in n2i:
            j = n2i[u_sym]
            if i < j: # Prevent redundant mirror constraints
                if symmetry == 'diag':
                    r1 = [0]*(2*n); r1[2*i] = 1; r1[2*j+1] = -1
                    r2 = [0]*(2*n); r2[2*i+1] = 1; r2[2*j] = -1
                    M.extend([r1, r2]); b.extend([0, 0])
                elif symmetry == 'book':
                    r1 = [0]*(2*n); r1[2*i] = 1; r1[2*j] = 1
                    r2 = [0]*(2*n); r2[2*i+1] = 1; r2[2*j+1] = -1
                    M.extend([r1, r2]); b.extend([1, 0]) # x1 + x2 = 1
            elif i == j: # On the axis
                if symmetry == 'diag':
                    r1 = [0]*(2*n); r1[2*i] = 1; r1[2*i+1] = -1
                    M.append(r1); b.append(0)
                elif symmetry == 'book':
                    r1 = [0]*(2*n); r1[2*i] = 1
                    M.append(r1); b.append(0.5)
                    
    return np.array(M), np.array(b)

def build_boundary_constraints(nodes, n2i, N):
    """Pins bottom-left to (0,0) and top-right to (1,1)."""
    n = len(n2i)
    M, b = [], []
    
    bl = (0, 0)
    tr = (N, N)
    
    if bl in n2i:
        idx = n2i[bl]
        r1 = [0]*(2*n); r1[2*idx] = 1
        r2 = [0]*(2*n); r2[2*idx+1] = 1
        M.extend([r1, r2]); b.extend([0, 0])
        
    if tr in n2i:
        idx = n2i[tr]
        r1 = [0]*(2*n); r1[2*idx] = 1
        r2 = [0]*(2*n); r2[2*idx+1] = 1
        M.extend([r1, r2]); b.extend([1, 1])
        
    return np.array(M), np.array(b)


def build_equidistant_for_edges(edges, pos, n2i):
    """Calculates left-nullspace rows and initial error for 4+ arbitrary edges."""
    k = len(edges)
    A = []
    d_orig = []
    
    # 1. Build Normal Matrix A and Initial Distances d_orig
    perimeter = 0
    for u, v in edges:
        dx, dy = pos[v][0] - pos[u][0], pos[v][1] - pos[u][1]
        L = math.hypot(dx, dy)
        perimeter += L
        nx, ny = -dy/L, dx/L # CCW inward normal
        A.append([nx, ny, -1])
        
        # Calculate original absolute distance to origin for the error metric
        d_orig.append(nx * pos[u][0] + ny * pos[u][1])
        
    A = np.array(A)
    d_orig = np.array(d_orig)
    
    # 2. Get Left Null Space
    W = null_space(A.T) # shape: (k, k-3)
    
    # Calculate geometric error: w^T * d_orig (how close it already is to 0)
    error = np.linalg.norm(W.T @ d_orig) * perimeter # Scale by perimeter to make it size-invariant
    
    # 3. Project back to absolute coordinate indices
    n = len(n2i)
    M_edges, b_edges = [], []
    
    for col in range(W.shape[1]):
        w = W[:, col]
        row = [0] * (2*n)
        for i in range(k):
            u = edges[i][0] # We only need the origin node u of the edge
            idx = n2i[u]
            row[2*idx] += w[i] * A[i, 0]     # w_i * n_x
            row[2*idx+1] += w[i] * A[i, 1]   # w_i * n_y
        M_edges.append(row)
        b_edges.append(0) # Absolute existence sum must be 0
        
    return np.array(M_edges), np.array(b_edges), error

def get_reflex_edge_pairs(face, pos):
    """Returns edge pairs forming a reflex (concave) angle. Handles both CCW and CW face loops."""
    k = len(face)
    if k < 3: return []
    
    # 1. Determine face orientation via signed area (Shoelace formula)
    signed_area = 0
    for i in range(k):
        u, v = face[i], face[(i+1)%k]
        signed_area += pos[u][0]*pos[v][1] - pos[v][0]*pos[u][1]
        
    is_ccw = signed_area > 0
    
    reflex_pairs = []
    for i in range(k):
        u = face[i-1]
        v = face[i]
        w = face[(i+1)%k]
        
        dx1, dy1 = pos[v][0] - pos[u][0], pos[v][1] - pos[u][1]
        dx2, dy2 = pos[w][0] - pos[v][0], pos[w][1] - pos[v][1]
        
        cross = dx1 * dy2 - dy1 * dx2
        
        # For CCW faces, a reflex angle is a right turn (cross < 0)
        # For CW faces, a reflex angle is a left turn (cross > 0)
        if (is_ccw and cross < -1e-5) or (not is_ccw and cross > 1e-5):
            reflex_pairs.append( ((u, v), (v, w)) )
            
    return reflex_pairs
# =============================================================================
# 4. SOLVER PIPELINE
# =============================================================================

def solve_absolute_positions(G_in, symmetry='none', N=4):
    # 1. Setup
    G, pos_init, nodes, n2i = clean_deg2_vertices(G_in, N)
    n = len(nodes)
    target_rank = 2 * n
    
    # 2. Base Constraints
    M_ang, b_ang = build_angle_constraints(G, pos_init, n2i)
    M_sym, b_sym = build_symmetry_constraints(nodes, pos_init, n2i, symmetry, N)
    M_bnd, b_bnd = build_boundary_constraints(nodes, n2i, N)
    
    M = np.vstack([M_ang, M_sym, M_bnd]) if M_sym.size else np.vstack([M_ang, M_bnd])
    b = np.concatenate([b_ang, b_sym, b_bnd]) if b_sym.size else np.concatenate([b_ang, b_bnd])
    # =========================================================================
    # 3. SMART EQUIDISTANT INJECTION (Concave Priority + Lowest Error)
    # =========================================================================
    faces = extract_oriented_faces(G, pos_init)
    
    current_rank = np.linalg.matrix_rank(M, tol=1e-7)
    print(f"Base Matrix Rank: {current_rank} / {target_rank} DOFs")
    
    candidate_constraints_concave = []
    candidate_constraints_convex = []
    
    # A. Harvest 4-edge combinations and classify them by concavity
    for face in faces:
        k = len(face)
        if k < 4: continue
        
        face_edges = [(face[i], face[(i+1)%k]) for i in range(k)]
        reflex_pairs = get_reflex_edge_pairs(face, pos_init)
        
        # Get all sets of exactly 4 edges from this face
        for edge_combo in combinations(face_edges, 4):
            M_eq, b_eq, error = build_equidistant_for_edges(edge_combo, pos_init, n2i)
            
            # Check if this 4-edge combination contains any of the reflex pairs
            has_concave_pair = False
            for e1, e2 in reflex_pairs:
                if e1 in edge_combo and e2 in edge_combo:
                    has_concave_pair = True
                    break
                    
            candidate = {
                'error': error,
                'M_eq': M_eq,
                'b_eq': b_eq,
                'edges': edge_combo
            }
            
            if has_concave_pair:
                candidate_constraints_concave.append(candidate)
            else:
                candidate_constraints_convex.append(candidate)
                
    # B. Sort both groups internally by their geometric error
    candidate_constraints_concave.sort(key=lambda x: x['error'])
    candidate_constraints_convex.sort(key=lambda x: x['error'])
    
    # Chain them: process all concave-containing subsets before purely convex subsets
    prioritized_candidates = candidate_constraints_concave + candidate_constraints_convex
    
    # C. Apply constraints greedily checking rank
    if current_rank < target_rank:
        for candidate in prioritized_candidates:
            M_eq = candidate['M_eq']
            b_eq = candidate['b_eq']
            
            for r, val in zip(M_eq, b_eq):
                M_test = np.vstack([M, r])
                new_rank = np.linalg.matrix_rank(M_test, tol=1e-7)
                
                # Only keep constraint if it removes a mathematically independent DOF
                if new_rank > current_rank:
                    M = M_test
                    b = np.append(b, val)
                    current_rank = new_rank
                    
                if current_rank == target_rank:
                    break
            if current_rank == target_rank:
                break


    print(f"Final Matrix Rank: {current_rank} / {target_rank}")
    # 4. Pseudoinverse Solve
    X = np.linalg.pinv(M) @ b
    
    # 5. Reconstruct 
    pos_solved = {}
    for u in nodes:
        idx = n2i[u]
        pos_solved[u] = (X[2*idx], X[2*idx+1])
        
    return G, pos_solved, faces

# =============================================================================
# 5. PLOTTING
# =============================================================================

def plot_with_skeletons(G, pos, faces):
    fig, ax = plt.subplots(figsize=(8,8))
    
    # 1. Plot Graph Edges
    for u, v in G.edges():
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 'k-', lw=2, zorder=2,alpha = 0.5)
        
    # 2. Plot Nodes
    for u in G.nodes():
        ax.plot(pos[u][0], pos[u][1], 'ko', markersize=4, zorder=3)

    # 3. Plot Straight Skeletons
    for face in faces:
        pts = [Point2D(pos[n][0], pos[n][1]) for n in face]
        poly = Polygon2D(pts)
        
        # Outer boundary check (skip plotting skeleton for the entire canvas)
        if poly.area > 0.95: 
            continue 
            
        try:
            skel_edges = skeleton_as_edge_list(poly)
            for seg in skel_edges:
                ax.plot([seg.p1.x, seg.p2.x], [seg.p1.y, seg.p2.y], 'r-', lw=1.5, alpha=0.6, zorder=1)
        except Exception as e:
            # PolySkel can occasionally fail on degenerate geometry
            pass
            
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_multiple(results, filename="renders/solved_batch.png"):
    """
    Plots a grid of graphs with their solved positions and straight skeletons.
    results: List of tuples (G, pos, faces)
    """
    if not results:
        print("No results to plot.")
        return
        
    n_graphs = len(results)
    cols = int(np.ceil(np.sqrt(n_graphs)))
    rows = int(np.ceil(n_graphs / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    # Flatten axes for easy iteration if more than one subplot
    if n_graphs > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]
    
    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        if i < n_graphs:
            G, pos, faces = results[i]
            
            # 1. Plot Graph Edges
            for u, v in G.edges():
                ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                        'k-', lw=1.5, zorder=2, alpha=0.5)
            
            # 2. Plot Nodes
            for u in G.nodes():
                ax.plot(pos[u][0], pos[u][1], 'ko', markersize=3, zorder=3)

            # 3. Plot Straight Skeletons
            for face in faces:
                try:
                    pts = [Point2D(pos[n][0], pos[n][1]) for n in face]
                    poly = Polygon2D(pts)
                    
                    # Skip outer boundary/canvas face
                    if poly.area > 0.95: 
                        continue 
                        
                    skel_edges = skeleton_as_edge_list(poly)
                    for seg in skel_edges:
                        ax.plot([seg.p1.x, seg.p2.x], [seg.p1.y, seg.p2.y], 
                                'r-', lw=1.2, alpha=0.6, zorder=1)
                except Exception:
                    # Catch PolySkel failures on degenerate geometries
                    pass
            
            ax.set_aspect('equal')
            ax.axis('off')
        else:
            # Hide empty subplots
            ax.axis('off')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=200)
        print(f"Saved batch plot to {filename}")
    plt.show()

# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
 

    # G_raw = extract_topology(3000, db_name = "topologies_4_none.db")
    # G_solved, pos_solved, faces = solve_absolute_positions(G_raw, symmetry='none', N=4)
    # plot_with_skeletons(G_solved, pos_solved, faces)
   
    solved_results = []
    # Assuming you've extracted a list of raw graphs from the DB
    for G_raw in [extract_topology(i, db_name="topologies_4_none.db") for i in random.sample(range(1,9000), 36)]: 
        try:
            # Solve using your refined displacement logic
            G_solved, pos_solved, faces = solve_absolute_positions(G_raw, symmetry='none', N=4)
            solved_results.append((G_solved, pos_solved, faces))
        except Exception as e:
            print(f"Failed to solve one topology: {e}")

    # Render the 4x4 grid
    plot_multiple(solved_results, filename="renders/n4_none_greedy_results.png")
