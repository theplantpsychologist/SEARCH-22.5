"""
Core Crease Pattern Module: Frozen Tiling Blob -> Exact Cp225
Trimmed for speed and database integration. 
Reads the serialized exact 4D states, calculates the straight skeleton using a quantized float bandaid, and routes the creases on an exact 22.5 grid.
"""

from src.engine.math225_core import Vertex4D, Fraction
from src.engine.cp225 import Cp225, intersection
from py_straight_skeleton import compute_skeleton

import math
import networkx as nx

# =============================================================================
# CONFIGURATION & TUNING
# =============================================================================

class SkeletonTuning:
    """
    Tunable parameters for the floating-point straight skeleton bandaid.
    """
    # Rounding threshold to force float-noise near-misses into exact collisions 
    # for the C++ sweep-line algorithm.
    QUANTIZATION_DECIMALS = 3 
    
    # Capture radius for mapping the float boundary nodes back to exact 4D nodes
    SNAP_TOLERANCE = 1e-3


# =============================================================================
# 1. DESERIALIZATION (Blob -> Exact State)
# =============================================================================

def load_frozen_blob(blob):
    """
    Deserializes the database blob back into exact mathematical objects and topologies.
    """
    # Reconstruct integer-indexed topology
    nodes = list(blob["vertices"].keys())
    
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(blob["edges"])
    
    faces = blob["faces"]
    
    # Reconstruct Exact 4D Vertices
    pos_solved_exact = {}
    for n_idx, frac_tuple in blob["pos_4d"].items():
        n1, d1, n2, d2, n3, d3, n4, d4 = frac_tuple
        pos_solved_exact[n_idx] = Vertex4D(
            Fraction(n1, d1), Fraction(n2, d2), 
            Fraction(n3, d3), Fraction(n4, d4)
        )
        
    return G, pos_solved_exact, faces


# =============================================================================
# 2. STRAIGHT SKELETON BANDAID
# =============================================================================

def dedupe_exterior(exterior):
    """Removes overlapping vertices and antiparallel spikes caused by exact solver merges."""
    if not exterior: return []
    
    cleaned = [exterior[0]]
    for p in exterior[1:]:
        if math.hypot(p[0]-cleaned[-1][0], p[1]-cleaned[-1][1]) > 1e-5:
            cleaned.append(p)
    if len(cleaned) > 1 and math.hypot(cleaned[0][0]-cleaned[-1][0], cleaned[0][1]-cleaned[-1][1]) < 1e-5:
        cleaned.pop()
        
    while len(cleaned) >= 3:
        spike_found = False
        for i in range(len(cleaned)):
            prev, curr, nxt = cleaned[i-1], cleaned[i], cleaned[(i+1)%len(cleaned)]
            dx1, dy1 = curr[0] - prev[0], curr[1] - prev[1]
            dx2, dy2 = nxt[0] - curr[0], nxt[1] - curr[1]
            
            L1, L2 = math.hypot(dx1, dy1), math.hypot(dx2, dy2)
            if L1 > 1e-5 and L2 > 1e-5:
                cross = (dx1*dy2 - dy1*dx2) / (L1*L2)
                if abs(cross) < 1e-5: 
                    cleaned.pop(i) 
                    spike_found = True
                    break
        if not spike_found: break
            
    return cleaned

def compute_skeleton_wrapper(exterior):
    """
    Aggressively quantizes the input to bypass IEEE-754 float singularities 
    in the py_straight_skeleton library.
    """
    d = SkeletonTuning.QUANTIZATION_DECIMALS
    quant_exterior = [(round(p[0], d), round(p[1], d)) for p in exterior]
    
    k = len(quant_exterior)
    area = sum(quant_exterior[i][0] * quant_exterior[(i+1)%k][1] - quant_exterior[(i+1)%k][0] * quant_exterior[i][1] for i in range(k))
    
    if abs(area) / 2.0 > 0.95: return None
    if area < 0: quant_exterior.reverse()
        
    cleaned = dedupe_exterior(quant_exterior)
    if len(cleaned) < 3: return None
    
    try:
        return compute_skeleton(exterior=cleaned, holes=[])
    except Exception:
        return None

def canonical_float(p):
    """Standardizes float coordinates for dictionary hashing."""
    return (round(p[0], 5), round(p[1], 5))


# =============================================================================
# 3. EXACT TOPOLOGY ROUTER
# =============================================================================

def build_crease_pattern(G, pos_solved_exact, faces, N=4):
    """
    Main entry point. Initializes the base Cp225 with boundary geometry, then 
    routes straight skeleton creases using exact algebraic raycasting.
    """
    # 1. Initialize Base Cp225
    n2i = {n: i for i, n in enumerate(G.nodes())}
    vertices = [pos_solved_exact[n] for n in G.nodes()]
    
    def is_border(v1_ex, v2_ex):
        x1, y1 = v1_ex.to_cartesian()
        x2, y2 = v2_ex.to_cartesian()
        return (math.isclose(x1, 0, abs_tol=1e-5) and math.isclose(x2, 0, abs_tol=1e-5)) or \
               (math.isclose(x1, N, abs_tol=1e-5) and math.isclose(x2, N, abs_tol=1e-5)) or \
               (math.isclose(y1, 0, abs_tol=1e-5) and math.isclose(y2, 0, abs_tol=1e-5)) or \
               (math.isclose(y1, N, abs_tol=1e-5) and math.isclose(y2, N, abs_tol=1e-5))
               
    cp_edges = []
    for u, v in G.edges():
        l_type = 'b' if is_border(pos_solved_exact[u], pos_solved_exact[v]) else 'v'
        cp_edges.append((n2i[u], n2i[v], l_type))
        
    cp = Cp225(vertices, cp_edges)
    
    # Float conversion needed ONLY to feed the C++ skeleton library
    S2 = math.sqrt(2) / 2.0
    pos_float = {u: (float(v.x) + S2*(float(v.y)-float(v.w)), 
                     float(v.z) + S2*(float(v.y)+float(v.w))) 
                 for u, v in pos_solved_exact.items()}

    # 2. Process Faces
    for face in faces:
        exterior = [pos_float[n] for n in face]
        skeleton = compute_skeleton_wrapper(exterior)
        if skeleton is None: continue
            
        # Build un-directed topological graph of the float skeleton
        skel_graph = {}
        for skv1, skv2 in skeleton.arc_iterator():
            p1 = canonical_float((float(getattr(skv1.position, 'x', skv1.position[0])), float(getattr(skv1.position, 'y', skv1.position[1]))))
            p2 = canonical_float((float(getattr(skv2.position, 'x', skv2.position[0])), float(getattr(skv2.position, 'y', skv2.position[1]))))
            skel_graph.setdefault(p1, set()).add(p2)
            skel_graph.setdefault(p2, set()).add(p1)
            
        exact_positions = {}
        node_to_idx = {}
        
        # Anchor boundaries
        for u in face:
            p_f = pos_float[u]
            min_d, closest_node = float('inf'), None
            for node in skel_graph:
                d = math.hypot(node[0] - p_f[0], node[1] - p_f[1])
                if d < min_d: min_d, closest_node = d, node
                    
            if min_d < SkeletonTuning.SNAP_TOLERANCE and closest_node is not None:
                exact_positions[closest_node] = cp.vertices[n2i[u]]
                node_to_idx[closest_node] = n2i[u]
                
        # Iteratively resolve internal nodes
        unresolved = set(skel_graph.keys()) - set(exact_positions.keys())
        while unresolved:
            progress = False
            for node in list(unresolved):
                resolved_nbrs = [nbr for nbr in skel_graph[node] if nbr in exact_positions]
                if len(resolved_nbrs) >= 2:
                    N_e = None
                    for i in range(len(resolved_nbrs)):
                        for j in range(i+1, len(resolved_nbrs)):
                            A_f, B_f = resolved_nbrs[i], resolved_nbrs[j]
                            angle_A = int(round(math.atan2(node[1] - A_f[1], node[0] - A_f[0]) / (math.pi/8))) % 16
                            angle_B = int(round(math.atan2(node[1] - B_f[1], node[0] - B_f[0]) / (math.pi/8))) % 16
                            
                            N_e = intersection(exact_positions[A_f], exact_positions[B_f], angle_A, angle_B)
                            if N_e is not None: break
                        if N_e is not None: break
                            
                    if N_e is not None:
                        exact_positions[node] = N_e
                        cp.vertices.append(N_e)
                        node_to_idx[node] = len(cp.vertices) - 1
                        unresolved.remove(node)
                        progress = True
                        break 
                        
            if not progress:
                print("Warning: Skeleton topology contains unresolvable internal vertices. (Likely Float Degeneracy)")
                break

        # Tag Reflex Bounds for MV assignments
        reflex_cp_indices = set()
        k = len(face)
        for i in range(k):
            u_id, v_id, w_id = face[i-1], face[i], face[(i+1)%k]
            p_u, p_v, p_w = pos_float[u_id], pos_float[v_id], pos_float[w_id]
            dx1, dy1 = p_v[0] - p_u[0], p_v[1] - p_u[1]
            dx2, dy2 = p_w[0] - p_v[0], p_w[1] - p_v[1]
            if (dx1 * dy2 - dy1 * dx2) > -1e-5:
                reflex_cp_indices.add(n2i[v_id])

        # Write creases to CP
        for p1 in skel_graph:
            for p2 in skel_graph[p1]:
                if p1 < p2: 
                    if p1 in node_to_idx and p2 in node_to_idx:
                        idx1, idx2 = node_to_idx[p1], node_to_idx[p2]
                        v1, v2 = cp.vertices[idx1], cp.vertices[idx2]
                        
                        # Implicit Degeneracy Filter: Skip 0-length edges created by exact collisions
                        if v1.x == v2.x and v1.y == v2.y and v1.z == v2.z and v1.w == v2.w:
                            continue
                        
                        exists = any(((u == idx1 and v == idx2) or (u == idx2 and v == idx1)) for u, v, _ in cp.edges)
                        if not exists:
                            l_type = "v" if (idx1 in reflex_cp_indices or idx2 in reflex_cp_indices) else "m"
                            cp.edges.append((idx1, idx2, l_type))
                            
    return cp


# =============================================================================
# 4. DEBUG & VISUALIZATION
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import random
    from src.engine.topology2tiling import solve_tiling, export_frozen_blob, extract_topology
    
    # 1. Plotter
    def draw_cp_ax(ax, cp, title="Crease Pattern"):
        plot_colors = {'m': 'red', 'v': 'blue', 'b': 'black'}
        ax.set_title(title, fontsize=14)    
        for t, x1, y1, x2, y2 in cp.render():
            ax.plot([x1, x2], [y1, y2], color=plot_colors.get(t, 'grey'), lw=2, zorder=2, alpha=0.7)
        ax.set_aspect('equal')
        ax.axis('off')

    # 2. Pipeline Integration Test
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes_flat = axes.flatten()
    
    for i, db_id in enumerate(random.sample(range(1, 9000), 4)):
        try:
            # Simulate Module 1 (Tiling -> Blob)
            G_raw = extract_topology(db_id, db_name="topologies_4_none.db", N=4)
            G_solved, pos_init, pos_solved_exact, faces, n2i = solve_tiling(G_raw, symmetry='none', N=4)
            blob = export_frozen_blob(G_solved, pos_solved_exact, n2i, faces)
            
            # Simulate Module 2 (Blob -> CP)
            loaded_G, loaded_pos, loaded_faces = load_frozen_blob(blob)
            cp = build_crease_pattern(loaded_G, loaded_pos, loaded_faces, N=4)
            
            draw_cp_ax(axes_flat[i], cp, title=f"CP from Blob (ID {db_id})")
            print(f"ID {db_id}: Successfully processed full pipeline.")
            
        except Exception as e:
            print(f"Failed to process topology {db_id}: {e}")
            axes_flat[i].set_title(f"ID {db_id} (Failed)", color='red')
            axes_flat[i].axis('off')
            
    plt.tight_layout()
    plt.show()