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
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    g = math.gcd(abs(dx), abs(dy))
    return (dx // g, dy // g)

def check_unique_directions(cycle_verts, pos):
    if len(cycle_verts) < 3: return False
    dirs = []
    for i in range(len(cycle_verts)):
        u = cycle_verts[i]
        v = cycle_verts[(i + 1) % len(cycle_verts)]
        d = get_dir(pos[u], pos[v])
        if not dirs: dirs.append(d)
        elif dirs[-1] != d: dirs.append(d)
        
    if len(dirs) > 1 and dirs[0] == dirs[-1]:
        dirs.pop()
    return len(dirs) == len(set(dirs))

def merge_faces(f1_edges, f2_edges):
    set1 = set(f1_edges)
    set2_rev = set((v, u) for (u, v) in f2_edges)
    e1_keep = [e for e in f1_edges if e not in set2_rev]
    e2_keep = [e for e in f2_edges if (e[1], e[0]) not in set1]
    all_new_edges = e1_keep + e2_keep
    
    adj = {}
    for u, v in all_new_edges:
        if u in adj: return None 
        adj[u] = v
        
    if not adj: return None
    start = list(adj.keys())[0]
    cycle = []
    curr = start
    while True:
        if curr not in adj: return None 
        nxt = adj[curr]
        cycle.append(curr)
        curr = nxt
        if curr == start: break
        
    if len(cycle) != len(all_new_edges): return None 
    return cycle

def try_merge(u, v, faces, edge_to_face, pos):
    """Attempts to merge two faces across edge (u, v). Updates dicts if successful."""
    e_dir, rev_e = (u, v), (v, u)
    if e_dir not in edge_to_face or rev_e not in edge_to_face: return False
    
    f1_id, f2_id = edge_to_face[e_dir], edge_to_face[rev_e]
    if f1_id == f2_id: return False
    
    f1_edges, f2_edges = faces[f1_id], faces[f2_id]
    new_cycle = merge_faces(f1_edges, f2_edges)
    
    if not new_cycle or not check_unique_directions(new_cycle, pos): 
        return False
        
    new_edges = [(new_cycle[i], new_cycle[(i+1)%len(new_cycle)]) for i in range(len(new_cycle))]
    del faces[f1_id]
    del faces[f2_id]
    
    new_fid = max(faces.keys()) + 1 if faces else 0
    faces[new_fid] = new_edges
    
    for old_e in f1_edges + f2_edges:
        if old_e in edge_to_face: del edge_to_face[old_e]
    for new_e in new_edges:
        edge_to_face[new_e] = new_fid
        
    return True

def get_t_junction_stems(faces, pos, width, height):
    """Finds all internal edges that act as stems for a T-junction."""
    stems = set()
    node_to_edges = {}
    for edges in faces.values():
        for u, v in edges:
            if u not in node_to_edges: node_to_edges[u] = []
            node_to_edges[u].append(v)
            
    for edges in faces.values():
        cycle_verts = [e[0] for e in edges]
        for i in range(len(cycle_verts)):
            prev_v = cycle_verts[i-1]
            curr_v = cycle_verts[i]
            next_v = cycle_verts[(i+1) % len(cycle_verts)]
            
            # Boundary T-Junctions are allowed (they form the square hull)
            if curr_v[0] in [0, width-1] or curr_v[1] in [0, height-1]: continue
                
            dir_in = get_dir(pos[prev_v], pos[curr_v])
            dir_out = get_dir(pos[curr_v], pos[next_v])
            
            # If the face has a 180-degree flat edge at this node
            if dir_in == dir_out: 
                # Any other edge connected here is a T-junction stem
                for v in node_to_edges[curr_v]:
                    edge_dir = get_dir(pos[curr_v], pos[v])
                    if edge_dir != dir_out and edge_dir != (-dir_out[0], -dir_out[1]):
                        stems.add(tuple(sorted([curr_v, v])))
    return list(stems)

def generate_strict_tiling(width=8, height=8, target_faces=12):
    """Generates a perfect edge-to-edge tiling with no internal T-junctions."""
    attempts = 0
    while True:
        attempts += 1
        pos, faces, face_id = {}, {}, 0
        for x in range(width):
            for y in range(height): pos[(x, y)] = (x, y)
                
        for x in range(width - 1):
            for y in range(height - 1):
                p1, p2, p3, p4 = (x,y), (x+1,y), (x+1,y+1), (x,y+1)
                faces[face_id] = [(p1, p2), (p2, p3), (p3, p1)]; face_id += 1
                faces[face_id] = [(p1, p3), (p3, p4), (p4, p1)]; face_id += 1
                
        edge_to_face = {e: fid for fid, edges in faces.items() for e in edges}
        internal_edges = [e for e, fid in edge_to_face.items() if (e[1], e[0]) in edge_to_face and e[0] < e[1]]
        random.shuffle(internal_edges)
        
        # Phase 1: Agglomerate to structural faces
        for u, v in internal_edges:
            if len(faces) <= target_faces: break
            try_merge(u, v, faces, edge_to_face, pos)
            
        # Phase 2: Targeted T-Junction Eradication
        stems = get_t_junction_stems(faces, pos, width, height)
        stuck = False
        
        while stems:
            merged_any = False
            for u, v in stems:
                if try_merge(u, v, faces, edge_to_face, pos):
                    merged_any = True
                    break # Topology changed, recalculate stems
            
            if not merged_any:
                stuck = True # We hit a geometric dead end
                break
            stems = get_t_junction_stems(faces, pos, width, height)
            
        # If we successfully eradicated all T-junctions, finalize the graph
        if not stuck and not stems:
            G = nx.Graph()
            for n, p in pos.items(): G.add_node(n, pos=p)
            for edges in faces.values():
                for u, v in edges: G.add_edge(u, v)
                
            G.remove_nodes_from(list(nx.isolates(G)))
            
            # Final Cleanup: Dissolve remaining degree-2 collinear artifacts
            for u in list(G.nodes()):
                if u in G and G.degree(u) == 2:
                    v1, v2 = list(G.neighbors(u))
                    d1 = get_dir(pos[u], pos[v1])
                    d2 = get_dir(pos[u], pos[v2])
                    if d1[0] == -d2[0] and d1[1] == -d2[1]:
                        G.add_edge(v1, v2)
                        G.remove_node(u)
                        
            print(f"Success after {attempts} attempt(s).")
            return G

def plot_final_tilings(graphs):
    n = len(graphs)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten() if n > 1 else [axes]
    
    for i in range(n):
        G = graphs[i]
        ax = axes[i]
        pos = nx.get_node_attributes(G, 'pos')
        for u, v in G.edges():
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 'k-', lw=1.5)
        for node, (x, y) in pos.items():
            ax.plot(x, y, 'ko', markersize=4)
        ax.set_aspect('equal'); ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    print("==== Start ====")
    N = 7
    target_faces = np.linspace(10, N*N, 20)
    graphs = [generate_strict_tiling(width=N, height=N, target_faces=int(faces)) for faces in target_faces]
    plot_final_tilings(graphs)

    print("=== End ====")
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")  # Sort by cumulative time
    stats.print_stats(20)  # Print the top functions