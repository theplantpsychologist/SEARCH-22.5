"""
Datatype for tiling objects as a planar graph of directed half-edges (doubly connected edge list)
"""
from time import time

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
from z3 import BoolVal, Solver, Bool, And, Or, Not, If, sat, Implies
import cProfile
import pstats

# =============================================================================
# 1. GEOMETRY & SYMMETRY HELPERS (D4 Group)
# =============================================================================

def apply_transform(u, N, t_type):
    """Applies one of the 8 Dihedral (D4) transformations to a coordinate. N is grid size."""
    x, y = u
    if t_type == 0:   return (x, y)               # Identity
    elif t_type == 1: return (y, N - x)           # Rot 90
    elif t_type == 2: return (N - x, N - y)       # Rot 180
    elif t_type == 3: return (N - y, x)           # Rot 270
    elif t_type == 4: return (N - x, y)           # Ref X (Book)
    elif t_type == 5: return (x, N - y)           # Ref Y
    elif t_type == 6: return (y, x)               # Ref D1 (Diag)
    elif t_type == 7: return (N - y, N - x)       # Ref D2

def transform_edges(edges, N, t_type):
    """Transforms a set of edges and returns them in a sorted, standard format."""
    transformed = []
    for u, v in edges:
        tu, tv = apply_transform(u, N, t_type), apply_transform(v, N, t_type)
        transformed.append(tuple(sorted((tu, tv))))
    return tuple(sorted(transformed))

def get_canonical_hash(edges, N):
    """Returns the lexicographically smallest edge tuple among all 8 symmetries."""
    return min(transform_edges(edges, N, i) for i in range(8))

def check_symmetry(edges, N, condition):
    """Checks if the graph satisfies the requested internal symmetry."""
    if condition == 'none': return True
    t_type = 6 if condition == 'diag' else 4 # 6 = y=x reflection, 4 = x-axis reflection
    return tuple(sorted(edges)) == transform_edges(edges, N, t_type)

# =============================================================================
# 2. FACE EXTRACTION & VALIDATION
# =============================================================================

def get_dir(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    g = math.gcd(abs(dx), abs(dy)) if dx or dy else 1
    return (dx // g, dy // g)

def check_unique_directions(cycle_verts):
    if len(cycle_verts) < 3: return False
    dirs = []
    for i in range(len(cycle_verts)):
        u, v = cycle_verts[i], cycle_verts[(i + 1) % len(cycle_verts)]
        d = get_dir(u, v)
        if not dirs or dirs[-1] != d: dirs.append(d)
    if len(dirs) > 1 and dirs[0] == dirs[-1]: dirs.pop()
    return len(dirs) == len(set(dirs))

def extract_faces(edges):
    """Extracts all faces using CCW angular sorting (DCEL logic)."""
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    # Sort neighbors counter-clockwise
    for u in adj:
        adj[u].sort(key=lambda v: math.atan2(v[1] - u[1], v[0] - u[0]))

    unvisited_he = set()
    for u, v in edges:
        unvisited_he.add((u, v))
        unvisited_he.add((v, u))

    faces = []
    while unvisited_he:
        start_he = unvisited_he.pop()
        face = [start_he]
        curr_he = start_he

        while True:
            u, v = curr_he
            neighbors = adj[v]
            idx = neighbors.index(u)
            # The edge immediately CCW from our arrival vector is the leftmost turn
            w = neighbors[(idx + 1) % len(neighbors)]
            next_he = (v, w)

            if next_he == start_he: break
            
            if next_he in unvisited_he:
                unvisited_he.remove(next_he)
            face.append(next_he)
            curr_he = next_he

        faces.append([he[0] for he in face])
    return faces
def is_valid_tiling_global(edges):
    """Post-filter for global properties: Connectivity and Unique Normals."""
    G = nx.Graph()
    G.add_edges_from(edges)
    
    if not nx.is_connected(G): return False
    
    faces = extract_faces(edges)
    for f in faces:
        if not check_unique_directions(f): return False
        
    return True
def lex_le(vars_A, vars_B):
    """Returns a Z3 constraint that boolean array A is lexicographically <= array B."""
    # Base case: if arrays are equal, they are <=
    res = BoolVal(True)
    # Walk backwards to build the AST efficiently
    for a, b in reversed(list(zip(vars_A, vars_B))):
        # A < B means (Not A and B). A == B means (a == b).
        res = Or(And(Not(a), b), And(a == b, res))
    return res


def enumerate_graphs(N=3, symmetry='none'):
    t0 = time()
    print(f"Initializing Accelerated Z3 Solver for N={N}, symmetry='{symmetry}'...")
    s = Solver()
    
    # 1. Define edges and Z3 Variables (Strictly Sorted)
    boundary_edges = [
        tuple(sorted(((i, 0), (i+1, 0)))) for i in range(N)
    ] + [
        tuple(sorted(((i, N), (i+1, N)))) for i in range(N)
    ] + [
        tuple(sorted(((0, i), (0, i+1)))) for i in range(N)
    ] + [
        tuple(sorted(((N, i), (N, i+1)))) for i in range(N)
    ]
    
    edge_vars = {}
    
    # Orthogonal
    for i in range(N):
        for j in range(1, N):
            edge_vars[tuple(sorted(((i, j), (i+1, j))))] = Bool(f"e_h_{i}_{j}")
            edge_vars[tuple(sorted(((j, i), (j, i+1))))] = Bool(f"e_v_{j}_{i}")
            
    # Diagonals
    cells = {}
    for x in range(N):
        for y in range(N):
            d1 = tuple(sorted(((x, y), (x+1, y+1))))
            d2 = tuple(sorted(((x+1, y), (x, y+1))))
            v1, v2 = Bool(f"d1_{x}_{y}"), Bool(f"d2_{x}_{y}")
            edge_vars[d1] = v1
            edge_vars[d2] = v2
            cells[(x, y)] = (v1, v2)

    # =========================================================================
    # Z3 CONSTRAINTS
    # =========================================================================

    # A. Planarity (No crossing diagonals)
    for cx, cy in cells:
        s.add(Not(And(cells[(cx, cy)][0], cells[(cx, cy)][1])))

    # B. No Dangling Edges (Degree != 1)
    incident_vars = { (x,y): [] for x in range(N+1) for y in range(N+1) }
    for u, v in boundary_edges:
        incident_vars[u].append(1)
        incident_vars[v].append(1)
    for e, var in edge_vars.items():
        val = If(var, 1, 0)
        incident_vars[e[0]].append(val)
        incident_vars[e[1]].append(val)
        
    for node, vars_list in incident_vars.items():
        s.add(sum(vars_list) != 1)

    # C. Symmetry
    if symmetry != 'none':
        t_type = 6 if symmetry == 'diag' else 4
        for e, var in edge_vars.items():
            re = tuple(sorted((apply_transform(e[0], N, t_type), apply_transform(e[1], N, t_type))))
            if re in edge_vars:
                s.add(var == edge_vars[re])

    # D. LOCAL TOPOLOGY RULE (T-Junctions Only)
    for x in range(1, N):
        for y in range(1, N):
            # 8 neighbors in CCW order starting from Right
            neighbors = [
                (x+1, y), (x+1, y+1), (x, y+1), (x-1, y+1),
                (x-1, y), (x-1, y-1), (x, y-1), (x+1, y-1)
            ]
            e_vars = [edge_vars[tuple(sorted(((x,y), nbr)))] for nbr in neighbors]
                
            # Strict T-Junctions
            pairs = [
                (e_vars[0], e_vars[4]), # R, L
                (e_vars[2], e_vars[6]), # U, D
                (e_vars[1], e_vars[5]), # UR, DL
                (e_vars[3], e_vars[7])  # UL, DR
            ]
            any_straight = Or(*[And(p[0], p[1]) for p in pairs])
            all_paired = And([p[0] == p[1] for p in pairs])
            s.add(Implies(any_straight, all_paired))

    # =========================================================================
    # NATIVE ISOMORPHISM REJECTION (The Game Changer)
    # =========================================================================
    # Establish a strict canonical ordering of all internal edges
    ordered_edges = sorted(list(edge_vars.keys()))
    base_vars = [edge_vars[e] for e in ordered_edges]

    # Apply the 7 non-identity transformations of the D4 group
    for t_type in range(1, 8):
        transformed_vars = []
        for e in ordered_edges:
            tu = apply_transform(e[0], N, t_type)
            tv = apply_transform(e[1], N, t_type)
            re = tuple(sorted((tu, tv)))
            
            if re in edge_vars:
                transformed_vars.append(edge_vars[re])
            else:
                # Boundary edges are implicitly True
                transformed_vars.append(BoolVal(True)) 
                
        # Force Z3 to only solve for the lexicographically smallest orientation
        s.add(lex_le(base_vars, transformed_vars))

    valid_graphs = []
    seen_hashes = set()
    model_count = 0
    
    print("Z3 Constraints built. Searching for valid models...")
    
    # We use a list of values to avoid dictionary `.items()` iteration during the loop
    all_vars = list(edge_vars.values())
    all_edges = list(edge_vars.keys())
    while s.check() == sat:
        model = s.model()
        current_edges = list(boundary_edges)
        block_clause = []
        
        for e, var in zip(all_edges, all_vars):
            is_active = bool(model.evaluate(var, model_completion=True))
            if is_active:
                current_edges.append(e)
            # Optimized Blocking: native Z3 != instead of And/Not chains
            block_clause.append(var != is_active)
                
        s.add(Or(*block_clause))
        
        # Build and yield the graph immediately
        G = nx.Graph()
        G.add_edges_from(current_edges)
        nx.set_node_attributes(G, {n: n for n in G.nodes()}, 'pos')
        if is_valid_tiling_global(current_edges):        
            yield G
# =============================================================================
# 4. BATCH PLOTTING
# =============================================================================

def plot_multiple_graphs(graphs, filename="renders/tilings.png", labels = []):
    if not graphs:
        print("No graphs to plot.")
        return
        
    n = len(graphs)
    cols = int((np.floor(np.sqrt(n))))
    rows = int(np.ceil(n / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if n > 1: axes = axes.flatten()
    else: axes = [axes]
    
    for i in range(len(axes)):
        ax = axes[i]
        if i < n:
            G = graphs[i]
            pos = nx.get_node_attributes(G, 'pos')
            for u, v in G.edges():
                ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 'k-', lw=1.5)
            for node, (x, y) in pos.items():
                ax.plot(x, y, 'ko', markersize=3)
            ax.set_aspect('equal')
        ax.axis('off')
        if labels and i < len(labels):
            ax.set_title(f"Index: {labels[i]}", fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"Saved plot to {filename}")

# Run it for N=2 with Diagonal Symmetry
if __name__ == "__main__":
    print("======= Start =========")
    n = 2

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        graphs = [g for g in enumerate_graphs(N=n, symmetry='book')]
        plot_multiple_graphs(graphs, f"renders/tilings_n{n}_book.png")
        print("=======")
        graphs = [g for g in enumerate_graphs(N=n, symmetry='diag')]
        plot_multiple_graphs(graphs[:100], f"renders/tilings_n{n}_diag.png")
        print("=======")
        graphs = [g for g in enumerate_graphs(N=n, symmetry='none')]
        plot_multiple_graphs(graphs, f"renders/tilings_n{n}_none.png")
        print("======= End =========")
    except KeyboardInterrupt as e:
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")  # Sort by cumulative time
        stats.print_stats(20)  # Print the top  functions