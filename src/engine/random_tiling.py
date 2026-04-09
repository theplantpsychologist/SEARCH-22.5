import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from matplotlib.patches import Polygon
import cProfile
import pstats
import math
from src.engine.math225_core import (
    Fraction,
    Vertex4D,
    AplusBsqrt2,
)

DIR_4D = {
    0: Vertex4D(1, 0, 0, 0),    # +X (0 deg)
    2: Vertex4D(0, 1, 0, 0),    # +Y (45 deg)
    4: Vertex4D(0, 0, 1, 0),    # +Z (90 deg)
    6: Vertex4D(0, 0, 0, 1),    # +W (135 deg)
    8: Vertex4D(-1, 0, 0, 0),   # -X (180 deg)
    10: Vertex4D(0, -1, 0, 0),  # -Y (225 deg)
    12: Vertex4D(0, 0, -1, 0),  # -Z (270 deg)
    14: Vertex4D(0, 0, 0, -1),  # -W (315 deg)
}
# precomputed tan values
TAN_225 = {
    0: AplusBsqrt2(0, 0),  # tan( 0 *22.5)=0+0sqrt(2)
    1: AplusBsqrt2(-1, 1),  # tan( 1 *22.5)=-1+1sqrt(2)
    2: AplusBsqrt2(1, 0),  # tan( 2 *22.5)=1+0sqrt(2)
    3: AplusBsqrt2(1, 1),  # tan( 3 *22.5)=1
    # 4:                     vertical, tan is infinite
    5: AplusBsqrt2(-1, -1),  # tan( 5 *22.5)=-(sqrt(2)+1)
    6: AplusBsqrt2(-1, 0),  # -1
    7: AplusBsqrt2(1, -1),  # 1-sqrt(2)
}
COT_225 = {
    #0: vertical, infinite
    1: AplusBsqrt2(1, 1),  
    2: AplusBsqrt2(1, 0), 
    3: AplusBsqrt2(-1, 1),  
    4: AplusBsqrt2(0, 1), 
    5: AplusBsqrt2(1, -1),  
    6: AplusBsqrt2(-1, 0),  
    7: AplusBsqrt2(-1, -1), 
}

#only defined for multiples of 45 because multiples of 22.5 break out of A+B*sqrt(2) form
COS_225 = {
    0: AplusBsqrt2(1, 0),
    2: AplusBsqrt2(0, Fraction(1,2)),
    4: AplusBsqrt2(0, 0),
    6: AplusBsqrt2(0, -Fraction(1,2)),
    8: AplusBsqrt2(-1, 0),
    10: AplusBsqrt2(0, -Fraction(1,2)),
    12: AplusBsqrt2(0, 0),
    14: AplusBsqrt2(0, Fraction(1,2)),
}
SIN_225 = {
    0: AplusBsqrt2(0, 0),
    2: AplusBsqrt2(0, Fraction(1,2)),
    4: AplusBsqrt2(1, 0),
    6: AplusBsqrt2(0, Fraction(1,2)),
    8: AplusBsqrt2(0, 0),
    10: AplusBsqrt2(0, -Fraction(1,2)),
    12: AplusBsqrt2(-1, 0),
    14: AplusBsqrt2(0, -Fraction(1,2)),
}
# basis vectors that become x and y in cartesian
X = Vertex4D(1,0,0,0)
Z = Vertex4D(0,0,1,0)
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import random

def get_dir(p1, p2):
    """Returns the normalized integer direction vector."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    g = math.gcd(abs(dx), abs(dy))
    return (dx // g, dy // g)

def check_unique_directions(cycle_verts, pos):
    """
    Checks if a polygon boundary uses any directed angle more than once.
    cycle_verts is a sequence of vertices forming a simple loop.
    """
    if len(cycle_verts) < 3: return False
    
    dirs = []
    for i in range(len(cycle_verts)):
        u = cycle_verts[i]
        v = cycle_verts[(i + 1) % len(cycle_verts)]
        d = get_dir(pos[u], pos[v])
        
        if not dirs:
            dirs.append(d)
        elif dirs[-1] != d:
            dirs.append(d)
            
    # Wrap-around check: merge last and first if they are collinear
    if len(dirs) > 1 and dirs[0] == dirs[-1]:
        dirs.pop()
        
    # Constraint 3: Are all collapsed directions strictly unique?
    return len(dirs) == len(set(dirs))

def merge_faces(f1_edges, f2_edges):
    """
    Attempts to merge two faces by removing all shared edges.
    Returns a sequence of vertices forming the new boundary, or None if invalid.
    """
    set1 = set(f1_edges)
    set2_rev = set((v, u) for (u, v) in f2_edges)
    
    # Keep edges that are not shared
    e1_keep = [e for e in f1_edges if e not in set2_rev]
    e2_keep = [e for e in f2_edges if (e[1], e[0]) not in set1]
    all_new_edges = e1_keep + e2_keep
    
    # Build adjacency mapping to trace the new loop
    adj = {}
    for u, v in all_new_edges:
        if u in adj: return None # Branching means it pinched itself (invalid topology)
        adj[u] = v
        
    if not adj: return None
    
    start = list(adj.keys())[0]
    cycle = []
    curr = start
    while True:
        if curr not in adj: return None # Dead end
        nxt = adj[curr]
        cycle.append(curr)
        curr = nxt
        if curr == start: break
        
    # If the cycle didn't consume all edges, it means the faces shared multiple 
    # disconnected segments and broke into pieces. Reject.
    if len(cycle) != len(all_new_edges): 
        return None 
        
    return cycle

def generate_agglomerative_graph(width=8, height=8, target_faces=8):
    """
    Starts with a fully triangulated planar mesh and randomly merges faces
    until the target complexity is reached, strictly enforcing unique normal vectors.
    """
    pos = {}
    faces = {} # face_id -> list of directed edges
    face_id = 0
    
    # 1. Initialize Base Mesh
    for x in range(width):
        for y in range(height):
            pos[(x, y)] = (x, y)
            
    for x in range(width - 1):
        for y in range(height - 1):
            p1, p2, p3, p4 = (x,y), (x+1,y), (x+1,y+1), (x,y+1)
            # Tri 1
            faces[face_id] = [(p1, p2), (p2, p3), (p3, p1)]
            face_id += 1
            # Tri 2
            faces[face_id] = [(p1, p3), (p3, p4), (p4, p1)]
            face_id += 1
            
    # 2. Build Edge Lookup
    edge_to_face = {}
    for fid, edges in faces.items():
        for e in edges:
            edge_to_face[e] = fid
            
    # 3. Extract and Shuffle Internal Edges
    internal_edges = []
    for e, fid1 in edge_to_face.items():
        rev_e = (e[1], e[0])
        if rev_e in edge_to_face and e[0] < e[1]: # Only add undirected pair once
            internal_edges.append(e)
            
    random.shuffle(internal_edges)
    
    # 4. Agglomerate (Merge)
    for u, v in internal_edges:
        if len(faces) <= target_faces:
            break
            
        e = (u, v)
        rev_e = (v, u)
        
        # Look up current faces for this edge
        if e not in edge_to_face or rev_e not in edge_to_face: continue
        f1_id = edge_to_face[e]
        f2_id = edge_to_face[rev_e]
        if f1_id == f2_id: continue
        
        f1_edges = faces[f1_id]
        f2_edges = faces[f2_id]
        
        # Try to merge
        new_cycle = merge_faces(f1_edges, f2_edges)
        if not new_cycle: continue
        
        # Enforce Constraint 3: Unique Directed Angles
        if not check_unique_directions(new_cycle, pos):
            continue
            
        # Merge is Successful! Update structures.
        new_edges = [(new_cycle[i], new_cycle[(i+1)%len(new_cycle)]) for i in range(len(new_cycle))]
        
        del faces[f1_id]
        del faces[f2_id]
        
        new_fid = face_id
        face_id += 1
        faces[new_fid] = new_edges
        
        # Update lookup (remove old, point new edges to new_fid)
        for old_e in f1_edges + f2_edges:
            if old_e in edge_to_face:
                del edge_to_face[old_e]
        for new_e in new_edges:
            edge_to_face[new_e] = new_fid

    # 5. Build Final NetworkX Graph
    G = nx.Graph()
    for n, p in pos.items():
        G.add_node(n, pos=p)
        
    for edges in faces.values():
        for u, v in edges:
            G.add_edge(u, v)
            
    # Clean up isolated nodes and degree-2 collinear artifacts from the merge
    G.remove_nodes_from(list(nx.isolates(G)))
    
    for u in list(G.nodes()):
        if u in G and G.degree(u) == 2:
            neighbors = list(G.neighbors(u))
            v1, v2 = neighbors
            d1 = get_dir(pos[u], pos[v1])
            d2 = get_dir(pos[u], pos[v2])
            if d1[0] == -d2[0] and d1[1] == -d2[1]:
                G.add_edge(v1, v2)
                G.remove_node(u)
                
    return nx.convert_node_labels_to_integers(G)

def sweep_plot(n=6):
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten() if n > 1 else [axes]
    
    for i in range(n):
        # We target ~10 faces. It will merge until it hits this, 
        # naturally forming long structural creases.
        G = generate_agglomerative_graph(width=9, height=9, target_faces=12)
        ax = axes[i]
        
        pos = nx.get_node_attributes(G, 'pos')
        for u, v in G.edges():
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 'k-', lw=1.5)
        for node, (x, y) in pos.items():
            ax.plot(x, y, 'ko', markersize=3)
            
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"Complexity: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    print("==== Start ====")
    
    sweep_plot(n=6)

    print("=== End ====")
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")  # Sort by cumulative time
    # stats.print_stats(20)  # Print the top functions