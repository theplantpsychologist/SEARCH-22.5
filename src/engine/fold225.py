"""
=================
LEVEL 2
=================

Core data structure to model a physical flat-folded state of a 22.5 crease pattern.

A Fold225 object contains:

- vertices: a list of Vertex4D objects, each representing a unique vertex position in the folded state.

- edges: a list of unique edges in the folded state, where each edge is stored as a tuple of vertex indices (v1,v2).

- faces: a list of unique faces in the folded state, where each face is stored as a list of edge indices [e1,e2,e3,...]. The winding order is arbitrary and may vary between faces, but the relative order of edges should be sorted.

- instances: A list of stacks, where stack f corresponds to face f in the faces list.
    - Each stack is a list of instances, where each instance is a physical copy of face f in the folded state.
        - Each instance has has its own implied address (f, i) where f is the face index and i is the instance index within that face's stack.
        - Each instance is represented in the stack as a list of connections, where connection e corresponds to the instance's connection across edge e from face f.
            - Each connection is a tuple (f2, i2) representing the address of the instance on the other side of that edge.
            - If the edge is a boundary edge, the connection is stored as None.

For example, the folded state for a square folded diagonally in half would look like:

Fold225(
    vertices=[
        Vertex4D(-1,0,-1,0),    # bottom left
        Vertex4D(1,0,1,0),      # top right
        Vertex4D(1,0,-1,0),     # bottom right
    ],
    edges=[
        (0,1),                  # diagonal fold edge
        (0,2),                  # bottom edge
        (1,2),                  # right edge
    ],
    faces=[
        [0,1,2],                # One unique face: triangle
    ],
    instances=[
        [                       # Only one stack, for the one face
            [(0,1), None, None], # Inst 0 is connected to inst1 across edge 0, and has boundary edges 1 and 2
            [(0,0), None, None], # vice versa
        ],
    ]


Note: abbreviations like v or v_idx refer to the index, not the object itself. Full variable names like vertex or vert refer to the object itself. Same goes for edges, faces, and instances.

"""

import cProfile
import pstats

import math
import os
from collections import deque, defaultdict
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import networkx as nx

from src.engine.math225_core import (
    Fraction,
    Vertex4D,
    reflect,
    reflect_group,
    canonicalize_fast,
    flatten_cpp,
)
from src.engine.cp225 import Cp225


ALPHA = 0.1  # transparency factor. 0.2 for 40gsm tracing paper
X = Vertex4D(1, 0, 0, 0)
Y = Vertex4D(0, 1, 0, 0)
Z = Vertex4D(0, 0, 1, 0)
W = Vertex4D(0, 0, 0, 1)

DIRECTIONS = [X, X + Y, Y, Y + Z, Z, Z + W, W, W - X]
# PRECOMPUTED_2D_DIRECTIONS = [d.get_2d_components() for d in DIRECTIONS]

PLOT_COLORS = {
    "m": "purple",
    "M": "purple",
    "v": "blue",
    "V": "blue",
    "b": "black",
    "A": "#42e7dc",
    "a": "#42e7dc",
    "i": "grey",
}


class Fold225:
    """
    A representation of a 22.5 folded state. This is the mutable "unfrozen" object.

    vertices: list of Vertex4D objects
    edges: each edge is a pair (v1,v2) vertex index pairs
    faces: each face is a counterclockwise ordered list of (e1,e2,e3,...) edge indices forming faces
    instances: list of lists, a list of instances for each face. Each instance is a list of tuples (face index, instance index) that the instance is connected to.
    """

    def __init__(self, vertices, edges, faces, instances):
        self.vertices = vertices
        self.edges = edges
        self.faces = faces
        self.instances = instances

    def __repr__(self):
        vertices = [vert.__repr__() for vert in self.vertices]
        concat = ""
        for vert in vertices:
            concat += vert + ","
        return (
            "Fold225(vertices=["
            + concat
            + f"],edges={self.edges},faces={self.faces},instances={self.instances})"
        )

    def __str__(self):
        instances_str = ""
        for i, instance in enumerate(self.instances):
            instances_str += f"\n \t Face{i}: {instance}"
        return f"Fold225 with Vertices={self.vertices},\nEdges={self.edges},\nFaces={self.faces},{instances_str}"

    def cpp_instances(self):
        """Converts the instances list into a flat format suitable for C++ processing. Each connection is converted from (f2, i2) to a single integer index based on the face and instance indices. Boundary edges are represented as (-1, -1) instead of None."""
        cpp_instances = []
        for face_stack in self.instances:
            cpp_stack = []
            for inst in face_stack:
                cpp_inst = []
                for conn in inst:
                    if conn is None:
                        cpp_inst.append((-1, -1))
                    else:
                        cpp_inst.append(conn)
                cpp_stack.append(cpp_inst)
            cpp_instances.append(cpp_stack)
        return cpp_instances

    # ================= Children generation ==================

    def get_slices(
        self, raycast: bool, bp: bool, midpoints: bool
    ) -> list[tuple[Vertex4D, int]]:
        """
        Computes unique slice lines using a canonical line hash (Angle, Offset).
        Prunes raycasts that do not intersect the model's projected bounding box.
        """
        unique_lines = {}

        # Precompute the DIRECTIONS for quick access

        # --- Precomputation Phase ---
        # We store the min/max offset for each of the 8 directions
        angle_bounds = {}
        if raycast:
            for i in range(8):
                normal_vec = DIRECTIONS[(i + 4) % 8]
                offsets = [v.dot_product(normal_vec) for v in self.vertices]
                # Exact comparison works because these are AplusBsqrt2 objects
                angle_bounds[i] = (min(offsets), max(offsets))

        def try_add_slice(v: Vertex4D, angle: int, is_raycast: bool):
            canon_angle = angle % 8
            normal_vec = DIRECTIONS[(canon_angle + 4) % 8]
            offset = v.dot_product(normal_vec)

            # --- Projected Bounding Box Filter ---
            if is_raycast:
                b_min, b_max = angle_bounds[canon_angle]
                # A slice is only kept if it lies STRICTLY within the min/max range,
                # or if it hits at least two distinct points on the boundary.
                # If offset == b_min or offset == b_max, it's a tangent line
                # and only hits the model if other vertices share that same offset.
                if offset < b_min or offset > b_max:
                    return

                # Optional: Strict pruning of tangent lines that only hit one corner.
                # To be safe and keep slices that might be internal creases,
                # we check if any other vertex exists on this same line.
                # (If you prefer to keep all tangent lines, remove this block)
                if offset in {b_min, b_max}:
                    # Count how many vertices share this offset
                    # (Can be optimized further by pre-hashing vertex offsets)
                    on_line_count = sum(
                        1
                        for vert in self.vertices
                        if vert.dot_product(normal_vec) == offset
                    )
                    if on_line_count < 2:
                        return

            line_id = (canon_angle, offset)
            if line_id not in unique_lines:
                unique_lines[line_id] = (v, angle)

        # --- Execution Phase ---
        if midpoints:
            for v1_idx, v2_idx in self.edges:
                v1, v2 = self.vertices[v1_idx], self.vertices[v2_idx]
                midpoint = (v1 + v2) * Fraction(1, 2)
                angle = v1.angle_to(v2)
                if angle is not None:
                    perp_angle = (angle + 4) % 8
                    try_add_slice(midpoint, perp_angle, is_raycast=False)
        if raycast:
            for vertex in self.vertices:
                if bp:
                    for i in range(0, 8, 2):
                        try_add_slice(vertex, i, is_raycast=True)
                else:
                    for i in range(8):
                        try_add_slice(vertex, i, is_raycast=True)

        return list(unique_lines.values())

    def get_split_info(self, point: Vertex4D, angle: int):
        """
        Classifies vertices and identifies edge intersections.
        Returns:
            vertex_sides: list of -1 (static), 0 (on line), 1 (moving)
            intersections: dict mapping old_edge_idx -> intersection_vertex_idx
            new_vertices: updated vertex list containing intersection points
        """
        canon_angle = angle % 8
        normal_vec = DIRECTIONS[(canon_angle + 4) % 8]
        slice_offset = point.dot_product(normal_vec)

        dots = [v.dot_product(normal_vec) for v in self.vertices]
        # Side: -1 = Static, 1 = Moving (Reflect), 0 = On Line
        vertex_sides = []
        for d in dots:
            if d > slice_offset:
                vertex_sides.append(1)
            elif d < slice_offset:
                vertex_sides.append(-1)
            else:
                vertex_sides.append(0)
        if not (-1 in vertex_sides and 1 in vertex_sides):
            return vertex_sides, [], self.vertices

        new_vertices = list(self.vertices)

        intersections = {}

        # edge is intersected if one vertex is on each side
        for e_idx, (v1, v2) in enumerate(self.edges):
            s1, s2 = vertex_sides[v1], vertex_sides[v2]
            if (s1 == 1 and s2 == -1) or (s1 == -1 and s2 == 1):
                p1, p2 = self.vertices[v1], self.vertices[v2]
                d1, d2 = dots[v1], dots[v2]  # Use the cached values!
                denom = d2 - d1
                numer = slice_offset - d1
                t = numer / denom

                iv_idx = len(new_vertices)
                new_vertices.append(p1 + (p2 - p1) * t)
                vertex_sides.append(0)
                intersections[e_idx] = iv_idx

        return vertex_sides, intersections, new_vertices

    def split_topology(self, vertex_sides, intersections):
        """
        Rebuilds edges, faces, and instances for a Fold225 object after slicing.
        """
        new_edges = []
        edge_lookup = {}

        def get_e(u, v):
            key = tuple(sorted((u, v)))
            if key in edge_lookup:
                return edge_lookup[key]
            idx = len(new_edges)
            new_edges.append((u, v))
            edge_lookup[key] = idx
            return idx

        # Map old edge indices to a list of new segment indices
        old_edge_map = {}
        for e_idx, (v1, v2) in enumerate(self.edges):
            if e_idx in intersections:
                iv = intersections[e_idx]
                # Split: v1 -> Intersection, Intersection -> v2
                old_edge_map[e_idx] = [get_e(v1, iv), get_e(iv, v2)]
            else:
                old_edge_map[e_idx] = [get_e(v1, v2)]

        new_faces = []
        old_face_map = defaultdict(list)

        for f_idx, face_edges in enumerate(self.faces):
            # --- A. Reconstruct the full Vertex Loop (including intersection points) ---

            # Find start vertex by checking connectivity of first/last edges
            loop_verts = []
            e_last, e_first = face_edges[-1], face_edges[0]
            v_start = None
            set_first = set(self.edges[e_first])
            if self.edges[e_last][0] in set_first:
                v_start = self.edges[e_last][0]
            else:
                v_start = self.edges[e_last][1]

            curr_v = v_start
            for e_idx in face_edges:
                segments = old_edge_map[e_idx]
                v_def_1, _ = self.edges[e_idx]

                if curr_v == v_def_1:
                    for seg_idx in segments:
                        u, v = new_edges[seg_idx]
                        next_v = v if u == curr_v else u
                        loop_verts.append(curr_v)
                        curr_v = next_v
                else:
                    for seg_idx in reversed(segments):
                        u, v = new_edges[seg_idx]
                        next_v = v if u == curr_v else u
                        loop_verts.append(curr_v)
                        curr_v = next_v

            # --- B. Bifurcate the Loop ---
            loop_sides = [vertex_sides[v] for v in loop_verts]
            if not (1 in loop_sides and -1 in loop_sides):
                # Face is entirely on one side; map vertices back to edges
                f_new = [
                    get_e(loop_verts[i], loop_verts[(i + 1) % len(loop_verts)])
                    for i in range(len(loop_verts))
                ]
                new_f_idx = len(new_faces)
                new_faces.append(f_new)
                old_face_map[f_idx].append(new_f_idx)
                continue

            # Find Entry (L->R) and Exit (R->L) pivots (Vertices with side 0)
            entry_idx, exit_idx = -1, -1
            n = len(loop_verts)
            for i in range(n):
                curr_s, nxt_s = loop_sides[i], loop_sides[(i + 1) % n]
                if curr_s <= 0 and nxt_s == 1:
                    entry_idx = i
                elif curr_s >= 0 and nxt_s == -1:
                    exit_idx = i

            # Create Left and Right polygon chains
            left_chain, right_chain = [], []
            curr = exit_idx
            while True:
                left_chain.append(loop_verts[curr])
                if curr == entry_idx:
                    break
                curr = (curr + 1) % n

            curr = entry_idx
            while True:
                right_chain.append(loop_verts[curr])
                if curr == exit_idx:
                    break
                curr = (curr + 1) % n

            # Convert chains to edge indices and add the internal crease
            f_left = [
                get_e(left_chain[i], left_chain[i + 1])
                for i in range(len(left_chain) - 1)
            ]
            crease = get_e(left_chain[-1], left_chain[0])
            f_left.append(crease)

            f_right = [
                get_e(right_chain[i], right_chain[i + 1])
                for i in range(len(right_chain) - 1)
            ]
            f_right.append(crease)

            # Store new face indices
            idx_l, idx_r = len(new_faces), len(new_faces) + 1
            new_faces.extend([f_left, f_right])
            old_face_map[f_idx] = [idx_l, idx_r]

        return new_edges, new_faces, old_edge_map, old_face_map

    def rebuild_instances(
        self, old_edge_map: dict, old_face_map: dict, new_faces: list[list[int]]
    ):
        """
        Reconstructs the positional instance lists for a split Fold225.
        Uses 'Slot-Based Matching' to ensure instances[f][i][j] always maps to new_faces[f][j].
        """
        old_instances = self.instances
        old_faces = self.faces
        new_instances = [[] for _ in new_faces]

        # 1. First Pass: Create the new instances and map old (f, i) -> new [(f1, i1), (f2, i2)]
        # i_map stores: old_f_i_tuple -> list of new_f_i_tuples
        i_map = {}

        for old_f_idx, old_face_insts in enumerate(old_instances):
            new_f_indices = old_face_map[old_f_idx]

            for old_i_idx, old_inst in enumerate(old_face_insts):
                # Track where this specific instance is moving to
                i_map[(old_f_idx, old_i_idx)] = []

                for nf_idx in new_f_indices:
                    current_new_face = new_faces[nf_idx]
                    # Initialize instance with Nones matching the NEW face length
                    # This fixes the T-junction length mismatch!
                    new_inst = [None] * len(current_new_face)

                    # A. Map inherited connections from the parent instance
                    for old_slot, connection in enumerate(old_inst):
                        old_e = old_faces[old_f_idx][old_slot]
                        # Get the segments this old edge turned into
                        segments = old_edge_map[old_e]
                        for seg in segments:
                            if seg in current_new_face:
                                # Place the connection in the correct new positional slot
                                slot = current_new_face.index(seg)
                                new_inst[slot] = connection

                    # B. Handle the Internal Crease if the face was split
                    if len(new_f_indices) > 1:
                        other_nf = (
                            new_f_indices[1]
                            if nf_idx == new_f_indices[0]
                            else new_f_indices[0]
                        )
                        # Find the edge index shared between the two new siblings
                        shared = set(current_new_face) & set(new_faces[other_nf])
                        if shared:
                            crease_e = list(shared)[0]
                            crease_slot = current_new_face.index(crease_e)
                            # We use a placeholder; we'll resolve the exact instance index in Pass 2
                            new_inst[crease_slot] = ("INTERNAL", other_nf)

                    new_ni_idx = len(new_instances[nf_idx])
                    new_instances[nf_idx].append(new_inst)
                    i_map[(old_f_idx, old_i_idx)].append((nf_idx, new_ni_idx))

        # 2. Second Pass: Resolve indices (Update connection pointers)
        for nf_idx, face_insts in enumerate(new_instances):
            for ni_idx, inst in enumerate(face_insts):
                for slot, conn in enumerate(inst):
                    if conn is None:
                        continue

                    # Case A: Internal Crease (Sibling connection)
                    if isinstance(conn, tuple) and conn[0] == "INTERNAL":
                        other_nf = conn[1]
                        # In a split, instance 0 of sibling 1 always connects to instance 0 of sibling 2
                        # because we are mapping 1:1 from the parent stack index.
                        inst[slot] = (other_nf, ni_idx)

                    # Case B: Standard Inherited Connection
                    else:
                        targets = i_map.get(conn)
                        if not targets:
                            continue  # Should be border edge, already None

                        if len(targets) == 1:
                            inst[slot] = targets[0]
                        else:
                            # Split Target: Connect to the sibling that shares THIS specific edge
                            current_edge = new_faces[nf_idx][slot]
                            match_found = False
                            for t_f, t_i in targets:
                                if current_edge in new_faces[t_f]:
                                    inst[slot] = (t_f, t_i)
                                    match_found = True
                                    break
                            if not match_found:
                                # Fallback/Error case: Edge exists in parent but not in split child
                                inst[slot] = None
        return new_instances

    def merge_heal(self, sides, v_flip_set):
        """
        Uses sets for geometric stitching, active edge tracking, and dictionary-based remapping.
        """
        # Pre-calculations
        num_vertices = len(self.vertices)
        v_flip_set = set(v_flip_set)  # Ensure this is a set for O(1)

        # 1. Faster Hinge Identification
        hinge_edges = {
            e_idx
            for e_idx, (v1, v2) in enumerate(self.edges)
            if sides[v1] == 0 and sides[v2] == 0
        }

        # 2. Faster Face-Vertex Side Mapping
        # Instead of nested loops, use bitwise-style flags for side properties
        start_map = [False] * len(self.faces)
        flip_map = [False] * len(self.faces)
        for f_idx, face in enumerate(self.faces):
            for e_idx in face:
                v1, v2 = self.edges[e_idx]
                if not start_map[f_idx] and (sides[v1] == 1 or sides[v2] == 1):
                    start_map[f_idx] = True
                if not flip_map[f_idx]:
                    if (v1 in v_flip_set and sides[v1] != 0) or (
                        v2 in v_flip_set and sides[v2] != 0
                    ):
                        flip_map[f_idx] = True

        # Preparation for Phase 1
        i_map = {}
        modified_faces = [list(f) for f in self.faces]
        # Deepcopy replacement
        modified_instances = [
            [list(inst) if inst is not None else None for inst in f_insts]
            for f_insts in self.instances
        ]

        # --- PHASE 1: Detection & Geometric Stitching ---
        for f1_idx, face_insts in enumerate(self.instances):
            for i1_idx, inst1 in enumerate(face_insts):
                if modified_instances[f1_idx][i1_idx] is None:
                    continue

                for slot1, connection in enumerate(inst1):
                    if connection is None:
                        continue

                    hinge_idx = self.faces[f1_idx][slot1]
                    if hinge_idx not in hinge_edges:
                        continue

                    f2_idx, i2_idx = connection
                    if modified_instances[f2_idx][i2_idx] is None:
                        continue

                    # XOR check to see if reflection flattened this crease
                    if (flip_map[f1_idx] == flip_map[f2_idx]) ^ (
                        start_map[f1_idx] == start_map[f2_idx]
                    ):
                        f1_loop = modified_faces[f1_idx]
                        f2_loop = modified_faces[f2_idx]

                        # O(1) would need a dict, but index() on small face lists is okay.
                        try:
                            s1, s2 = f1_loop.index(hinge_idx), f2_loop.index(hinge_idx)
                        except ValueError:
                            continue

                        # Rotate and Stitch
                        f1_rot = f1_loop[s1 + 1 :] + f1_loop[: s1 + 1]
                        i1_rot = inst1[s1 + 1 :] + inst1[: s1 + 1]
                        f2_rot = f2_loop[s2:] + f2_loop[:s2]
                        i2_rot = (
                            self.instances[f2_idx][i2_idx][s2:]
                            + self.instances[f2_idx][i2_idx][:s2]
                        )

                        # Winding detection logic (unchanged but stabilized)
                        reverse_f2 = False
                        if len(f1_rot) >= 2 and len(f2_rot) >= 2:
                            v_h1, v_h2 = self.edges[hinge_idx]
                            v1_prev = set(self.edges[f1_rot[-2]])
                            v_merge = v_h1 if v_h1 in v1_prev else v_h2
                            if v_merge in set(self.edges[f2_rot[-1]]):
                                reverse_f2 = True

                        if reverse_f2:
                            stitched_edges = f1_rot[:-1] + f2_rot[1:][::-1]
                            stitched_conns = i1_rot[:-1] + i2_rot[1:][::-1]
                        else:
                            stitched_edges = f1_rot[:-1] + f2_rot[1:]
                            stitched_conns = i1_rot[:-1] + i2_rot[1:]

                        # --- THE PINCH REMOVER ---
                        # Using a list as a stack is more efficient for popping duplicates
                        changed = True
                        while changed and len(stitched_edges) >= 2:
                            changed = False
                            for idx in range(len(stitched_edges)):
                                next_idx = (idx + 1) % len(stitched_edges)
                                if stitched_edges[idx] == stitched_edges[next_idx]:
                                    high, low = (
                                        (next_idx, idx)
                                        if next_idx > idx
                                        else (idx, next_idx)
                                    )
                                    for p_idx in [high, low]:
                                        stitched_edges.pop(p_idx)
                                        stitched_conns.pop(p_idx)
                                    changed = True
                                    break

                        # Commit Merge
                        new_f_idx = len(modified_faces)
                        modified_faces.append(stitched_edges)
                        modified_instances.append([stitched_conns])
                        i_map[(f1_idx, i1_idx)] = (new_f_idx, 0)
                        i_map[(f2_idx, i2_idx)] = (new_f_idx, 0)
                        modified_instances[f1_idx][i1_idx] = None
                        modified_instances[f2_idx][i2_idx] = None

        # --- PHASE 2 & 3: Compact and Remap ---
        f_final, inst_final, fi_shift = [], [], {}
        for f_idx, face_insts in enumerate(modified_instances):
            temp_row = []
            for i_idx, inst in enumerate(face_insts):
                if inst is not None:
                    fi_shift[(f_idx, i_idx)] = (len(f_final), len(temp_row))
                    temp_row.append(inst)
            if temp_row:
                f_final.append(modified_faces[f_idx])
                inst_final.append(temp_row)

        for face_stack in inst_final:
            for inst in face_stack:
                for slot, conn in enumerate(inst):
                    if conn is None:
                        continue
                    curr = conn
                    while curr in i_map:
                        curr = i_map[curr]
                    inst[slot] = fi_shift.get(curr)

        self.faces, self.instances = f_final, inst_final

        # --- PHASE 4: Global Collinear Simplification (Set-Optimized) ---
        while True:
            # Re-identify currently ACTIVE edges using a set
            active_edges = {e for face in self.faces for e in face}

            # Build vertex-to-edge incidence map only for active edges
            v_edges = [[] for _ in range(num_vertices)]
            for e_idx in active_edges:
                v1, v2 = self.edges[e_idx]
                v_edges[v1].append(e_idx)
                v_edges[v2].append(e_idx)

            merged_any = False
            for v_idx, incident in enumerate(v_edges):
                # A vertex is mergeable if it sits on the slice line (0) and
                # exactly two active edges meet there.
                if sides[v_idx] == 0 and len(incident) == 2:
                    e1, e2 = incident
                    v_center = self.vertices[v_idx]
                    v_a = (
                        self.edges[e1][0]
                        if self.edges[e1][1] == v_idx
                        else self.edges[e1][1]
                    )
                    v_b = (
                        self.edges[e2][0]
                        if self.edges[e2][1] == v_idx
                        else self.edges[e2][1]
                    )

                    if (
                        v_center.angle_to(self.vertices[v_a])
                        == (v_center.angle_to(self.vertices[v_b]) + 8) % 16
                    ):
                        new_e_idx = len(self.edges)
                        self.edges.append(tuple(sorted([v_a, v_b])))

                        for f_idx, face in enumerate(self.faces):
                            if (
                                e1 in face and e2 in face
                            ):  # Now O(N), but faces are small
                                i1, i2 = face.index(e1), face.index(e2)
                                first, second = (i1, i2) if i1 < i2 else (i2, i1)

                                # Handle wrap-around index cases
                                keep, pop = (
                                    (0, second)
                                    if (first == 0 and second == len(face) - 1)
                                    else (first, second)
                                )

                                face[keep] = new_e_idx
                                face.pop(pop)
                                for inst in self.instances[f_idx]:
                                    if inst:
                                        c_keep, c_pop = inst[keep], inst.pop(pop)
                                        inst[keep] = (
                                            c_keep if c_keep is not None else c_pop
                                        )
                        merged_any = True
                        break
            if not merged_any:
                break

        # --- PHASE 5: Set-Based Garbage Collection ---
        active_e_indices = {e for face in self.faces for e in face}
        compact_edges = []
        e_shift = {}
        for old_e, geom in enumerate(self.edges):
            if old_e in active_e_indices:
                e_shift[old_e] = len(compact_edges)
                compact_edges.append(geom)

        for f_idx in range(len(self.faces)):
            self.faces[f_idx] = [e_shift[e] for e in self.faces[f_idx]]

        active_v_indices = {v for e in compact_edges for v in e}
        compact_vertices = []
        v_shift = {}
        for old_v, vert in enumerate(self.vertices):
            if old_v in active_v_indices:
                v_shift[old_v] = len(compact_vertices)
                compact_vertices.append(vert)

        self.edges = [
            tuple(sorted((v_shift[v1], v_shift[v2]))) for v1, v2 in compact_edges
        ]
        self.vertices = compact_vertices
        return self

    def weld(self):
        """
        Fuses geometrically identical vertices and edges.
        Crucial for repairing topology after splitting stacked layers
        or reflecting vertices onto existing coordinates.
        """
        # --- 1. Weld Vertices ---
        unique_vertices = []
        v_map = {}
        v_dict = {}  # Maps Vertex4D -> new index

        for old_v_idx, v in enumerate(self.vertices):
            if v not in v_dict:
                v_dict[v] = len(unique_vertices)
                unique_vertices.append(v)
            v_map[old_v_idx] = v_dict[v]

        self.vertices = unique_vertices

        # --- 2. Weld Edges ---
        unique_edges = []
        e_map = {}
        e_dict = {}  # Maps (v1, v2) -> new edge index

        for old_e_idx, e in enumerate(self.edges):
            if e[0] is None:
                continue

            # Map to new vertex indices and sort to ensure undirected equivalence
            v1, v2 = v_map[e[0]], v_map[e[1]]

            # Skip degenerate point-edges if they somehow exist
            if v1 == v2:
                continue

            geom_key = tuple(sorted((v1, v2)))

            if geom_key not in e_dict:
                e_dict[geom_key] = len(unique_edges)
                unique_edges.append(list(geom_key))

            e_map[old_e_idx] = e_dict[geom_key]

        self.edges = unique_edges

        # --- 3. Update Face Edge Pointers ---
        for f_idx in range(len(self.faces)):
            for slot in range(len(self.faces[f_idx])):
                old_e = self.faces[f_idx][slot]
                if old_e in e_map:
                    # Update the face loop without changing its length,
                    # keeping the instances array perfectly synced.
                    self.faces[f_idx][slot] = e_map[old_e]

        return self

    def isolate_component(self, component, base_sides):
        """
        'Tears' the component away from the rest of the mesh by duplicating
        any off-hinge vertices and edges.
        Returns the vertices to flip and an updated sides array.
        """
        # We must track the sides of the newly duplicated vertices
        child_sides = list(base_sides)

        i_map = {}
        v_map = {}  # Maps old_v_idx -> new_v_idx
        e_map = {}  # Maps old_e_idx -> new_e_idx

        for f_idx, i_idx in component:
            old_face = self.faces[f_idx]
            new_face = []

            for e_idx in old_face:
                if e_idx not in e_map:
                    v1, v2 = self.edges[e_idx]

                    # If the edge is entirely on the hinge, it stays shared!
                    if child_sides[v1] == 0 and child_sides[v2] == 0:
                        e_map[e_idx] = e_idx
                    else:
                        # Clone vertices if they are strictly off-hinge
                        if child_sides[v1] != 0 and v1 not in v_map:
                            v_map[v1] = len(self.vertices)
                            self.vertices.append(self.vertices[v1])
                            child_sides.append(child_sides[v1])

                        if child_sides[v2] != 0 and v2 not in v_map:
                            v_map[v2] = len(self.vertices)
                            self.vertices.append(self.vertices[v2])
                            child_sides.append(child_sides[v2])

                        # Create cloned edge
                        new_v1 = v_map.get(v1, v1)
                        new_v2 = v_map.get(v2, v2)
                        new_e_idx = len(self.edges)
                        self.edges.append((new_v1, new_v2))
                        e_map[e_idx] = new_e_idx

                new_face.append(e_map[e_idx])

            # Create a brand new face loop for this instance
            new_f_idx = len(self.faces)
            self.faces.append(new_face)

            # Move the instance to the new face
            inst_conns = self.instances[f_idx][i_idx]
            self.instances[f_idx][i_idx] = None
            self.instances.append([inst_conns])
            i_map[(f_idx, i_idx)] = (new_f_idx, 0)

        # Global Remap of Instance Connections
        for f_stack in self.instances:
            for inst in f_stack:
                if inst is None:
                    continue
                for slot, conn in enumerate(inst):
                    if conn in i_map:
                        inst[slot] = i_map[conn]

        # Return ONLY the newly cloned off-hinge vertices for flipping.
        return set(v_map.values()), child_sides

    def fold_along_slice(
        self, point: Vertex4D, angle: int, components_to_flip: str
    ) -> list["Fold225"]:
        """
        Generate children folds by slicing along a line and flipping connected components across
        """

        p2 = point + DIRECTIONS[angle % 8]
        # 1. Get Geometric context

        sides, inters, new_v = self.get_split_info(point, angle)

        # 2. Slice the paper (Topological change only)
        new_edges, new_faces, old_edge_map, old_face_map = self.split_topology(
            sides, inters
        )

        # 3. Rebuild instances
        # old_edge_map: old edge index as key -> new edge indices, if split. If not split, just returns itself. old_face_map is similar
        new_instances = self.rebuild_instances(old_edge_map, old_face_map, new_faces)
        sliced_fold = Fold225(
            vertices=new_v, edges=new_edges, faces=new_faces, instances=new_instances
        )

        if components_to_flip == "ALL":
            final_verts = []
            for i, v in enumerate(sliced_fold.vertices):
                # Logic: If vertex is on the 'Right' side, reflect it.
                # Note: 'sides' corresponds to sliced_fold.vertices
                if sides[i] == 1:
                    final_verts.append(reflect(point, p2, v))
                else:
                    final_verts.append(v)
            return [flatten(sliced_fold)]

        # --- 4. Identify Connected Components (Flaps) using NetworkX ---
        G = nx.Graph()

        # A. Populate the graph with all instances as nodes
        for f_idx, face_insts in enumerate(sliced_fold.instances):
            for i_idx in range(len(face_insts)):
                G.add_node((f_idx, i_idx))

        # # B. Add edges where connectivity is NOT across a hinge
        edges_to_add = []
        hinge_edges = {
            i
            for i, (v1, v2) in enumerate(sliced_fold.edges)
            if sides[v1] == 0 and sides[v2] == 0
        }

        for f_idx, face_insts in enumerate(sliced_fold.instances):
            face_edges = sliced_fold.faces[f_idx]
            for i_idx, inst_conns in enumerate(face_insts):
                u = (f_idx, i_idx)
                for slot, neighbor in enumerate(inst_conns):
                    if neighbor and face_edges[slot] not in hinge_edges:
                        edges_to_add.append((u, neighbor))
        G.add_edges_from(edges_to_add)

        # C. Extract connected components as a list of lists of (f_idx, i_idx)
        connected_components = [list(c) for c in nx.connected_components(G)]

        # --- 5. Generate Children Folds ---
        children = []
        if components_to_flip == "ONE":
            for component in connected_components:
                # 1. Create the child state

                child_vertices = list(
                    sliced_fold.vertices
                )  # Shallow copy of the vertex list

                child_edges = list(
                    sliced_fold.edges
                )  # Edges are likely tuples, so shallow is fine
                child_faces = [
                    list(f) for f in sliced_fold.faces
                ]  # Faces need one level of nesting copy
                child_instances = [
                    [list(inst) if inst else None for inst in stack]
                    for stack in sliced_fold.instances
                ]

                child = Fold225(
                    child_vertices, child_edges, child_faces, child_instances
                )
                # 2. Tear the component away from the underlying layers
                verts_to_flip, child_sides = child.isolate_component(component, sides)
                # 3. Apply the reflection matrix ONLY to those specific vertices

                for v_idx in verts_to_flip:
                    child.vertices[v_idx] = reflect(point, p2, child.vertices[v_idx])

                # 4. Heal the cut and aggressively re-weld everything

                child.merge_heal(child_sides, verts_to_flip)

                child.weld()

                children.append(flatten(child))
        return children

    def get_children(
        self, raycast=True, bp=False, midpoints=True, components_to_flip="ONE"
    ) -> dict:
        """
        returns a set of canonicalized children

        Slice parameters:
        - raycast: slice from 22.5 angles out of all vertices
        - bp: restrict slices to 45 degree angles
        - midpoints: allow perpendicular edge bisectors as slices

        Flipping parameter:
        - components_to_flip: "ALL" to fold through all layers, "ONE" to fold one component at a time, or "ANY" to fold any combination of layers
        """
        slices = self.get_slices(raycast=raycast, bp=bp, midpoints=midpoints)

        canonical_self = canonicalize(self)
        # children = {}
        # for _, slice_ in enumerate(slices):
        #     new_children = self.fold_along_slice(*slice_, components_to_flip)
        #     for child in new_children:
        #         canonical_child = canonicalize(child)
        #         if (
        #             canonical_child != canonical_self
        #             and canonical_child not in children
        #         ):
        #             children[canonical_child] = canonical_self
        # return children
        children = set()
        for _, slice_ in enumerate(slices):
            new_children = self.fold_along_slice(*slice_, components_to_flip)
            for child in new_children:
                canonical_child = canonicalize(child)
                if (
                    canonical_child != canonical_self
                    and canonical_child not in children
                ):
                    children.add(canonical_child)
        return children

    # ================= Visualization and evaluation ==================
    def render(self) -> list[list[list[float, float]]]:
        """
        Renders the fold into a list of faces. Each face consists of a list of 2D points.
        """
        multiplicities = [len(instances) for instances in self.instances]

        rendered_faces = []
        for face in self.faces:
            vertices = []
            for i, e in enumerate(face):
                v1, v2 = self.edges[e]
                if v1 not in self.edges[face[(i + 1) % len(face)]]:
                    vertices.append(v1)
                else:
                    vertices.append(v2)
            vertices = [self.vertices[v].to_cartesian() for v in vertices]
            rendered_faces.append(vertices)
        return rendered_faces, multiplicities

    def get_tree_and_packing(self, include_packing=False) -> tuple[nx.Graph, tuple]:
        """
        Computes the uniaxial tree by sweeping the fold along a projection axis.
        """
        # --- 1. Axis Selection ---
        # Check even angles 0, 2, 4, 6 for the largest projected bounding box width
        best_angles = []
        max_width = -1.0

        for angle in [0, 2, 4, 6]:
            axis_vec = DIRECTIONS[angle]
            projections = [float(vert.dot_product(axis_vec)) for vert in self.vertices]
            width = max(projections) - min(projections)

            if width > max_width + 1e-9:
                max_width = width
                best_angles = [angle]
            elif abs(width - max_width) < 1e-9:
                best_angles.append(angle)

        # If ties, we proceed with the first best angle found. A folded state where the axis is ambiguous is probably not a tree we really want (very simple shape, or highly nonuniaxial. If a uniaxial variant exists, it probably would have been upstream)
        chosen_angle = best_angles[0]
        axis_vec = DIRECTIONS[chosen_angle]
        perp_angle = (chosen_angle + 4) % 8

        # --- 2. Hinge Point Generation & Slicing ---
        # Project and find unique hinge points along the axis
        projections = [
            (vert.dot_product(axis_vec), v_idx)
            for v_idx, vert in enumerate(self.vertices)
        ]
        # Remove duplicates
        unique_projections_map = {}
        for proj, v_idx in projections:
            if proj not in unique_projections_map:
                unique_projections_map[proj] = v_idx
        # 3. Sort the unique projection values using the AplusBsqrt2 dunder methods
        sorted_unique_values = sorted(unique_projections_map.keys())

        # 4. Final list of unique "hinge points" for the sweep-line
        # Format: [(AplusBsqrt2, v_idx), ...]
        hinge_points = [
            (proj, unique_projections_map[proj]) for proj in sorted_unique_values
        ]

        # Sweep-slice across every hinge point using split_topology
        current_fold = self
        for _, v_idx in hinge_points:
            # Re-calculating split info for the modified fold
            sides, inters, new_v = current_fold.get_split_info(
                self.vertices[v_idx], perp_angle
            )

            # Slicing creates new topology where faces are split along the 'hinge'
            new_edges, new_faces, old_e_map, old_f_map = current_fold.split_topology(
                sides, inters
            )
            new_insts = current_fold.rebuild_instances(old_e_map, old_f_map, new_faces)
            current_fold = Fold225(new_v, new_edges, new_faces, new_insts)

        # --- 3. Component Analysis ---
        # Build adjacency ignoring edges perpendicular to the axis (hinge-parallel)
        instance_to_idx = {}
        idx_to_instance = {}
        curr_idx = 0
        for f_idx, stack in enumerate(current_fold.instances):
            for i_idx in range(len(stack)):
                instance_to_idx[(f_idx, i_idx)] = curr_idx
                idx_to_instance[curr_idx] = (f_idx, i_idx)
                curr_idx += 1

        hinge_edges = set()
        for e_idx, (v1, v2) in enumerate(current_fold.edges):
            if (
                current_fold.vertices[v1].angle_to(current_fold.vertices[v2]) % 8
                == perp_angle
            ):
                hinge_edges.add(e_idx)
        instances_adj = nx.Graph()
        for f_idx, stack in enumerate(current_fold.instances):
            for i_idx, inst in enumerate(stack):
                u = instance_to_idx[(f_idx, i_idx)]
                for slot, conn in enumerate(inst):
                    if conn is None:
                        continue

                    # Filter: Ignore connections across hinge-parallel edges
                    e_idx = current_fold.faces[f_idx][slot]
                    if e_idx not in hinge_edges:
                        instances_adj.add_edge(u, instance_to_idx[conn])

        # Find Connected Components. List of sets of instance indices
        instance_components = list(nx.connected_components(instances_adj))
        # --- 4. Tree Construction ---
        tree = nx.Graph()

        # 4.1 Identify Hinge Events (Sorted unique projection values)
        # Using AplusBsqrt2 objects as keys for exact comparison
        hinge_positions = [val for val, _ in hinge_points]

        # 4.2 Map Components to Hinge Events
        # Every component must connect exactly 2 hinge events (or 1 if it's a leaf tip)
        component_hinges = (
            []
        )  # for each component, a tuple of (min_hinge_val, max_hinge_val)
        for component in instance_components:
            v_indices = set()
            for idx in component:
                f_idx, i_idx = idx_to_instance[idx]
                for e_idx in current_fold.faces[f_idx]:
                    v_indices.update(current_fold.edges[e_idx])

            # Get unique projection values for this component
            comp_h_vals = sorted(
                list(
                    set(
                        current_fold.vertices[v].dot_product(axis_vec)
                        for v in v_indices
                    )
                )
            )

            if len(comp_h_vals) > 2:
                # This check ensures the slicing was thorough enough
                raise RuntimeError(
                    f"Component spans {len(comp_h_vals)} hinge points. Slicing failed."
                )

            # (min_h, max_h, component_id)
            component_hinges.append((comp_h_vals[0], comp_h_vals[-1]))

        # 4.3 Cluster Components at each Hinge to Create Tree Nodes
        # tree_nodes[(hinge_idx, cluster_id)] = node_id
        tree_nodes = {}
        node_counter = 0

        for hinge in hinge_positions:
            # Find all components incident to this hinge. component indices
            incident_components = []
            for c_idx, (h_start, h_end) in enumerate(component_hinges):
                if hinge in {h_start, h_end}:
                    incident_components.append(c_idx)
            # Build local adjacency of components connected to this hinge, which components are connected to each other. Connect components if they share a hinge-parallel edge
            local_adj = nx.Graph()
            local_adj.add_nodes_from(incident_components)

            for i in range(len(incident_components)):
                for j in range(i + 1, len(incident_components)):
                    c1_idx, c2_idx = incident_components[i], incident_components[j]
                    component1, component2 = (
                        instance_components[c1_idx],
                        instance_components[c2_idx],
                    )

                    # Check if any instance in comp1 connects to any in comp2
                    is_connected = False
                    for idx1 in component1:
                        f1, i1 = idx_to_instance[idx1]
                        for slot, conn in enumerate(current_fold.instances[f1][i1]):
                            if conn is None:
                                continue
                            if instance_to_idx[conn] in component2:
                                # Not only is it connected, but it needs to be connected by an edge that's colinear with this specific hinge
                                e_idx = current_fold.faces[f1][slot]
                                v1, v2 = current_fold.edges[e_idx]
                                if (
                                    current_fold.vertices[v1].dot_product(axis_vec)
                                    == hinge
                                    and e_idx in hinge_edges
                                ):
                                    is_connected = True
                                    break
                        if is_connected:
                            break
                    if is_connected:
                        local_adj.add_edge(c1_idx, c2_idx)
            # Each local cluster of connected components means there's a connected set of hinge creases unique to this cluster. each of these cluster hinges becomes a node in the tree
            for comp_cluster in nx.connected_components(local_adj):
                new_node = node_counter
                tree.add_node(new_node, h_val=hinge)
                for c_idx in comp_cluster:
                    # Store which tree node this component belongs to at this hinge
                    # We use a tuple (c_idx, h_val) as the key
                    tree_nodes[(c_idx, hinge)] = new_node
                node_counter += 1

        # 4.4 Bridge Nodes with Component Edges (1-to-1 Mapping)
        for c_idx, (h_start, h_end) in enumerate(component_hinges):
            u = tree_nodes[(c_idx, h_start)]
            v = tree_nodes[(c_idx, h_end)]

            # Length is the axial distance
            dist = float(h_end - h_start) / SQUARE_SIZE

            tree.add_edge(u, v, length=dist, weight=1.0 / dist, comp_id=c_idx)

        # --- 5. Simplification ---
        removed_nodes = set()
        while True:
            to_merge = [n for n in tree.nodes() if tree.degree(n) == 2]
            if not to_merge:
                break

            n = to_merge[0]
            neighbors = list(tree.neighbors(n))
            u, v = neighbors[0], neighbors[1]
            new_len = tree[u][n]["length"] + tree[n][v]["length"]
            tree.remove_node(n)
            tree.add_edge(u, v, length=new_len, weight=1.0 / new_len)
            removed_nodes.add(n)

        if not include_packing:
            return tree, (current_fold, None)

        # --- 6. Instance Connection Classification  ---
        instances_adj.add_nodes_from(instance_to_idx.values())
        instance_components = list(nx.connected_components(instances_adj))
        inst_to_comp = {}
        for c_idx, comp in enumerate(instance_components):
            for idx in comp:
                inst_to_comp[idx] = c_idx

        annotated_G = nx.Graph()

        # Helper to exactly determine which side of a hinge plane a face sits on.
        # Because slicing guarantees a face doesn't cross the plane, we just need
        # to find any single vertex on the face that isn't exactly ON the hinge.
        def get_face_side(f_idx, h_val):
            for e_idx in current_fold.faces[f_idx]:
                for v_idx in current_fold.edges[e_idx]:
                    d = current_fold.vertices[v_idx].dot_product(axis_vec)
                    if d != h_val:
                        return d > h_val
            raise RuntimeError(
                "Face is exactly on the hinge plane, which should be impossible after slicing."
            )

        for f_idx, stack in enumerate(current_fold.instances):
            for i_idx, inst in enumerate(stack):
                u = (f_idx, i_idx)
                annotated_G.add_node(u)

                for slot, conn in enumerate(inst):
                    if conn is None:
                        continue
                    v = conn
                    e_idx = current_fold.faces[f_idx][slot]
                    if e_idx not in hinge_edges:
                        # Not a hinge plane at all -> Standard physical fold
                        annotated_G.add_edge(
                            u, v, edge_idx=e_idx, is_aux=False, is_invisible=False
                        )
                    else:
                        v_hinge = current_fold.edges[e_idx][0]
                        h_val = current_fold.vertices[v_hinge].dot_product(axis_vec)
                        # Exact topological side detection
                        side1 = get_face_side(f_idx, h_val)
                        side2 = get_face_side(v[0], h_val)

                        # is_aux represents the physical folded state: False = Folded, True = Flat
                        is_aux = side1 != side2

                        is_invisible = False
                        # ONLY artificial flat slices that get merged away become invisible.
                        # Real physical folds (is_aux=False) are preserved forever.
                        if is_aux:
                            c_idx = inst_to_comp[instance_to_idx[u]]
                            t_node = tree_nodes.get((c_idx, h_val))
                            if t_node is None or t_node in removed_nodes:
                                is_invisible = True

                        annotated_G.add_edge(
                            u,
                            v,
                            edge_idx=e_idx,
                            is_aux=is_aux,
                            is_invisible=is_invisible,
                        )

        return tree, (current_fold, annotated_G)

    def goodness(self) -> float:
        """
        Heurisic for base quality
        TODO: better design this. should the denominator be added or multiplied? look at correlation with tree efficiency
        """
        num_instances = sum(len(stack) for stack in self.instances)
        return num_instances / (len(self.faces) + len(self.vertices) + len(self.edges))


# =============================================================================
# Canonicalization and Freezing
# =============================================================================


def canonicalize(fold: "Fold225"):
    """
    Wrapper for the C++ canonicalization function. Converts the Fold225 object into a flat tuple of integers that represents its canonical form.
    """
    return tuple(
        canonicalize_fast(fold.vertices, fold.edges, fold.faces, fold.cpp_instances())
    )


def unfreeze(flat_tuple: tuple) -> Fold225:
    """
    Reconstructs a Fold225 object from the C++ canonical integer tuple.
    """
    data = flat_tuple
    it = iter(data)

    def next_int():
        return next(it)

    # 1. Read Header
    num_v = next_int()
    num_e = next_int()
    num_f = next_int()

    # 2. Reconstruct Vertices (x, y, z, w)
    vertices = []
    for _ in range(num_v):
        v_coords = []
        for _ in range(4):
            num = next_int()
            den = next_int()
            v_coords.append(Fraction(num, den))
        vertices.append(Vertex4D(*v_coords))

    # 3. Reconstruct Edges
    edges = []
    for _ in range(num_e):
        v1 = next_int()
        v2 = next_int()
        edges.append((v1, v2))

    # 4. Reconstruct Faces
    faces = []
    for _ in range(num_f):
        face_len = next_int()
        face_edges = [next_int() for _ in range(face_len)]
        faces.append(face_edges)

    # 5. Reconstruct Instances (Face -> Stack -> Slot)
    instances = []
    for _ in range(num_f):
        num_instances = next_int()
        face_stack = []
        for _ in range(num_instances):
            slot_count = next_int()
            inst_conns = []
            for _ in range(slot_count):
                tf = next_int()
                ti = next_int()
                # Restore (-1, -1) as None for Python
                inst_conns.append((tf, ti) if tf != -1 else None)
            face_stack.append(inst_conns)
        instances.append(face_stack)

    return Fold225(vertices, edges, faces, instances)


# =============================================================================
# Visualization
# =============================================================================


def plot_multiple(folds: list[Fold225], debug=False):
    """Plots multiple folds in a grid layout. Each fold is rendered as a subplot with its faces drawn as polygons. If debug is True, vertex and edge indices are also annotated for easier debugging."""
    n = len(folds)
    rows = math.ceil(math.sqrt(n / 2))
    cols = math.ceil(n / rows)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten() if n > 1 else [axes]

    bounds = []
    for i, fold in enumerate(folds):
        ax = axes[i]
        rendered_faces, multiplicities = fold.render()
        for j, face in enumerate(rendered_faces):
            polygon = Polygon(
                face, closed=True, alpha=1 - ((1 - ALPHA) ** multiplicities[j])
            )
            ax.add_patch(polygon)

        xs = [p[0] for face in rendered_faces for p in face]
        ys = [p[1] for face in rendered_faces for p in face]
        if xs and ys:
            bounds.append((min(xs), max(xs), min(ys), max(ys)))

        ax.set_aspect("equal")
        # Hide axes and bounding box
        ax.set_title(f"Fold {i}", fontsize=20)

        if debug:
            # Plot vertex indices
            for v_idx, vertex in enumerate(fold.vertices):
                v_float = vertex.to_cartesian()
                ax.plot(v_float[0], v_float[1], "ro", markersize=4)
                ax.text(
                    v_float[0],
                    v_float[1],
                    str(v_idx),
                    fontsize=8,
                    color="red",
                    ha="center",
                    va="bottom",
                )

            # Plot edges with indices
            for e_idx, (v1_idx, v2_idx) in enumerate(fold.edges):
                v1 = fold.vertices[v1_idx].to_cartesian()
                v2 = fold.vertices[v2_idx].to_cartesian()

                # Draw the edge
                ax.plot([v1[0], v2[0]], [v1[1], v2[1]], "b-", linewidth=0.5, alpha=0.3)

                # Plot edge index at midpoint
                mid_x = (v1[0] + v2[0]) / 2
                mid_y = (v1[1] + v2[1]) / 2
                ax.text(
                    mid_x,
                    mid_y,
                    str(e_idx),
                    fontsize=6,
                    color="blue",
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.7,
                    ),
                )

    width = max(b[1] - b[0] for b in bounds)
    height = max(b[3] - b[2] for b in bounds)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        center = ((bounds[i][0] + bounds[i][1]) / 2, (bounds[i][2] + bounds[i][3]) / 2)
        ax.set_xlim(center[0] - width / 2, center[0] + width / 2)
        ax.set_ylim(center[1] - height / 2, center[1] + height / 2)

    # Save figure
    renders_dir = "renders"
    os.makedirs(renders_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(renders_dir) if f.endswith(".png")]
    file_count = len(existing_files)
    filename = f"rendered_folds_{file_count}.png"
    filepath = os.path.join(renders_dir, filename)
    plt.tight_layout(pad=0)
    plt.savefig(filepath)
    plt.close(fig)
    print(f"Saved render to {filepath}")


def get_proportional_tree_pos(G):
    """
    Tree plot helper
    """
    if not G.nodes():
        return {}

    # 1. Calculate the full distance matrix for the tree
    full_dist_matrix = dict(nx.all_pairs_dijkstra_path_length(G, weight="length"))

    try:
        pos = nx.kamada_kawai_layout(G, dist=full_dist_matrix, scale=1.0)
    except:
        # Fallback to a basic tree layout if the matrix is problematic
        pos = nx.spring_layout(G, weight="weight", iterations=200)

    return pos


def plot_multi_state_grid(
    folds, cps=None, trees=None, packings=None, packing_instead_of_cp=True
):
    """
    Plots n entries in a grid.
    Layout: c macro-columns. Each macro-column = [Fold, CP/Packing, Tree, Spacer]

    If cps/packing/trees are provided, they will be plotted. Otherwise, they will be computed based on the fold objects.

    Note: a packing consists of a folded state sliced along all the hinges, an adjacency map so you know to ignore hinge connections, and a list of hinge edges so you know which edges to highlight as hinges in the cp plot.
    """
    if cps is None and not packing_instead_of_cp:
        cps = [fold_to_cp(fold) for fold in folds]

    if trees is None or (packings is None and packing_instead_of_cp):
        zipped = [
            fold.get_tree_and_packing(include_packing=packing_instead_of_cp)
            for fold in folds
        ]
        if trees is None:
            trees = [z[0] for z in zipped]
        if packings is None and packing_instead_of_cp:
            packings = []
            sliced_folds = [z[1] for z in zipped]
            for sliced_fold, annotated_G in sliced_folds:
                packings.append(fold_to_cp(sliced_fold, inst_graph=annotated_G))

    n = len(folds)
    if n == 0:
        return

    # Layout calculations
    # c macro-columns
    c = max(1, math.floor(math.sqrt(n / 4)))
    r = math.ceil(n / c)

    # Total internal columns = c * 4 (Fold, CP, Tree, Space)
    total_cols = c * 4
    fig, axes = plt.subplots(r, total_cols, figsize=(total_cols * 4, r * 5))

    # Ensure axes is 2D
    if r == 1:
        axes = axes.reshape(1, -1)

    bounds = []
    for i in range(n):
        row = i // c
        macro_col = i % c
        start_col = macro_col * 4

        ax_fold = axes[row, start_col]
        ax_cp = axes[row, start_col + 1]
        ax_tree = axes[row, start_col + 2]
        ax_space = axes[row, start_col + 3]

        # 1. --- FOLDED STATE ---
        fold = folds[i]
        rendered_faces, multiplicities = fold.render()
        for j, face in enumerate(rendered_faces):
            # Using 0.5 as a placeholder for your ALPHA constant
            alpha_val = 1 - ((1 - ALPHA) ** multiplicities[j])
            poly = Polygon(
                face, closed=True, alpha=alpha_val, color="teal", ec="black", lw=0.5
            )
            ax_fold.add_patch(poly)

        ax_fold.set_aspect("equal")
        ax_fold.axis("off")
        xs = [p[0] for face in rendered_faces for p in face]
        ys = [p[1] for face in rendered_faces for p in face]
        if xs and ys:
            bounds.append((min(xs), max(xs), min(ys), max(ys)))

        # 2. --- CREASE PATTERN ---
        if packing_instead_of_cp:
            cp = packings[i]
        else:
            cp = cps[i]
        for edge in cp.render():
            l_type, x1, y1, x2, y2 = edge
            # Assumes PLOT_COLORS exists in your namespace
            ax_cp.plot([x1, x2], [y1, y2], color=PLOT_COLORS[l_type], linewidth=3)

        ax_cp.set_aspect("equal")
        ax_cp.axis("off")

        # 3. --- TREE (Proportional Layout) ---
        G = trees[i]

        pos = get_proportional_tree_pos(G)

        nx.draw_networkx_nodes(G, pos, ax=ax_tree, node_size=50, node_color="gray")
        nx.draw_networkx_edges(
            G, pos, ax=ax_tree, width=3, edge_color="gray", alpha=0.6
        )

        # Optional display edge lengths
        # edge_labels = {k: f"{v/SQUARE_SIZE:.2f}" for k, v in nx.get_edge_attributes(G, 'length').items()}
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax_tree, font_size=8)

        ax_tree.set_aspect("equal")
        ax_tree.axis("off")

        # 4. --- SPACER ---
        ax_space.axis("off")

        ax_cp.set_title(
            f"Fold ({i}),   Goodness: {fold.goodness():.4f},   Efficiency: {sum(nx.get_edge_attributes(G, 'length').values()) /4 :.2f}",
            fontsize=24,
        )

    width = max(b[1] - b[0] for b in bounds)
    height = max(b[3] - b[2] for b in bounds)
    for i in range(n):
        row = i // c
        macro_col = i % c
        start_col = macro_col * 4

        ax_fold = axes[row, start_col]
        ax_fold.axis("off")
        if i >= n:
            continue
        center = ((bounds[i][0] + bounds[i][1]) / 2, (bounds[i][2] + bounds[i][3]) / 2)
        ax_fold.set_xlim(center[0] - width / 2, center[0] + width / 2)
        ax_fold.set_ylim(center[1] - height / 2, center[1] + height / 2)

    # Hide any remaining empty subplots in the grid
    for total_idx in range(n, r * c):
        row = total_idx // c
        m_col = total_idx % c
        for offset in range(4):
            axes[row, (m_col * 4) + offset].axis("off")

    plt.tight_layout()

    # Save Logic
    renders_dir = "renders"
    os.makedirs(renders_dir, exist_ok=True)
    f_path = os.path.join(
        renders_dir, f"rendered_states_{len(os.listdir(renders_dir))}.png"
    )
    plt.savefig(f_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Rendered {n} states to {f_path}")


# =============================================================================
# Conversion between Cp225 and Fold225
# =============================================================================


def cp_to_fold(cp: Cp225) -> Fold225:
    """
    Fold up a crease pattern. Before flattening, edges/vertices can overlap, and separate instances are counted as separate faces.
    """

    faces = cp.compute_faces()
    num_faces = len(faces)
    num_vertices = len(cp.vertices)

    # First, map every edge to the face(s) that own it.
    edge_to_faces = defaultdict(list)
    for f_idx, face_edges in enumerate(faces):
        for e_idx in face_edges:
            edge_to_faces[e_idx].append(f_idx)

    # initialize instances
    instances = []
    for face in faces:
        instances.append(
            [  # list of instances: just one instance
                [None] * len(face)  # starter instance: all Nones
            ]
        )

    # Also track neighbor relationships for the Kinematics BFS
    face_neighbors = defaultdict(list)

    for e_idx in range(len(cp.edges)):
        owners = edge_to_faces[e_idx]

        if len(owners) == 2:
            # Internal Crease
            f1, f2 = owners
            instances[f1][0][faces[f1].index(e_idx)] = (f2, 0)
            instances[f2][0][faces[f2].index(e_idx)] = (f1, 0)

            face_neighbors[f1].append((f2, e_idx))
            face_neighbors[f2].append((f1, e_idx))

        # Note: If len(owners) == 0, the edge is floating (unused).
        #       If len(owners) == 1, edge is border. Already default None in that spot
        #       If len(owners) > 2, the CP is non-manifold (invalid).

    # --- 2. Kinematics (BFS for Folded Positions) ---
    face_parent = [None] * num_faces
    face_parent[0] = (0, None)  # Root face
    q = deque([0])

    while q:
        f1 = q.popleft()
        for f2, e_idx in face_neighbors[f1]:
            if face_parent[f2] is None:
                face_parent[f2] = (f1, e_idx)
                q.append(f2)

    # Map vertices to faces for coordinate calculation
    vertex_to_face = {}
    for f_idx, face_edges in enumerate(faces):
        for e_idx in face_edges:
            v1, v2, _ = cp.edges[e_idx]
            vertex_to_face[v1] = f_idx
            vertex_to_face[v2] = f_idx

    # Calculate positions
    new_vertices = [None] * num_vertices
    for v_idx in range(num_vertices):
        pos = cp.vertices[v_idx]
        curr_f = vertex_to_face.get(v_idx, 0)

        # Walk up the tree to the root
        while curr_f != 0:
            parent_info = face_parent[curr_f]
            if parent_info is None:
                # Disconnected face (shouldn't happen in valid CP)
                break
            parent_f, e_idx = parent_info

            # Get original crease vertices for reflection
            v1_idx, v2_idx, _ = cp.edges[e_idx]
            pos = reflect(cp.vertices[v1_idx], cp.vertices[v2_idx], pos)
            curr_f = parent_f

        new_vertices[v_idx] = pos
    fold = Fold225(
        vertices=new_vertices,
        edges=[(v1, v2) for v1, v2, m in cp.edges],
        faces=faces,
        instances=instances,
    )

    return flatten(fold)


def flatten(fold: Fold225) -> Fold225:
    """
    Remove redundant vertices, edges, faces from a Fold225 representation.
    Delegates heavy topology tracking to C++.
    """

    # 2. Call the C++ Core
    unique_verts, unique_edges, unique_faces, cpp_new_insts = flatten_cpp(
        fold.vertices, fold.edges, fold.faces, fold.cpp_instances()
    )

    # 3. Rebuild the Python object (Restore -1 to None)
    new_instances = []
    for face_stack in cpp_new_insts:
        new_stack = []
        for inst in face_stack:
            new_inst = []
            for conn in inst:
                if conn[0] == -1:
                    new_inst.append(None)
                else:
                    new_inst.append(tuple(conn))
            new_stack.append(new_inst)
        new_instances.append(new_stack)

    return Fold225(
        vertices=unique_verts,
        edges=[tuple(e) for e in unique_edges],
        faces=unique_faces,
        instances=new_instances,
    )


def fold_to_cp(fold: Fold225, inst_graph: nx.Graph = None) -> Cp225:
    """
    Unfolds the state back to a Crease Pattern.
    Uses inst_graph to label 'a' (auxiliary) and erase invisible slices.
    If no inst_graph is passed, fall back to a standard physical fold (assumes all edges are folded, no auxiliary joins)
    """

    # 1. Fallback Graph Generation
    if inst_graph is None:
        inst_graph = nx.Graph()
        for f_idx, stack in enumerate(fold.instances):
            for i_idx, inst in enumerate(stack):
                u = (f_idx, i_idx)
                inst_graph.add_node(u)
                for slot, conn in enumerate(inst):
                    if conn is not None:
                        e_idx = fold.faces[f_idx][slot]
                        inst_graph.add_edge(
                            u, conn, edge_idx=e_idx, is_aux=False, is_invisible=False
                        )

    # 2. Pathing
    root = (0, 0)
    paths = nx.single_source_shortest_path(inst_graph, root)

    cp_verts = []
    edge_tracker = {}  # Maps (v_low, v_high) -> 'b', 'm', 'a'

    # 3. Unfolding and Coloring
    for node, path_nodes in paths.items():
        f_idx, i_idx = node

        # Extract path instructions
        path_info = []
        for i in range(len(path_nodes) - 1):
            data = inst_graph[path_nodes[i]][path_nodes[i + 1]]
            path_info.append((data["edge_idx"], data["is_aux"]))

        # --- TOPOLOGY FIX: Aligning vertices to slots ---
        face_edges = fold.faces[f_idx]
        face_verts_idx = []
        for i, e in enumerate(face_edges):
            # Look BACKWARDS to find the starting vertex of edge 'e'
            prev_e = face_edges[(i - 1) % len(face_edges)]
            v1, v2 = fold.edges[e]
            face_verts_idx.append(v1 if v1 in fold.edges[prev_e] else v2)

        f_verts = [fold.vertices[v] for v in face_verts_idx]

        # Geometry of the unfolding path
        path_geoms = [
            [fold.vertices[v1], fold.vertices[v2]]
            for v1, v2 in [fold.edges[eid] for eid, _ in path_info]
        ]

        # Unfolding must happen from leaf to root. Reflecting across the exact folded hinge geometries in reverse order mathematically guarantees the flat state without needing to compound reflections.
        for i in reversed(range(len(path_geoms))):
            if path_info[i][1]:  # Skip aux bridges
                continue
            f_verts = reflect_group(*path_geoms[i], f_verts)

        # Map to CP indices
        f_cp_idxs = []
        for v in f_verts:
            if v not in cp_verts:
                cp_verts.append(v)
            f_cp_idxs.append(cp_verts.index(v))

        # 4. Process Edges for this Face
        for slot, v1_cp in enumerate(f_cp_idxs):
            v2_cp = f_cp_idxs[(slot + 1) % len(f_cp_idxs)]
            edge_key = tuple(sorted((v1_cp, v2_cp)))

            target_inst = fold.instances[f_idx][i_idx][slot]

            if edge_key not in edge_tracker:
                edge_tracker[edge_key] = "b"  # First encounter is always boundary
            else:
                # Second encounter completes the connection based on local graph data
                if target_inst is not None and inst_graph.has_edge(node, target_inst):
                    edge_data = inst_graph[node][target_inst]
                    if edge_data["is_invisible"]:
                        # Erase the boundary entirely
                        del edge_tracker[edge_key]
                    else:
                        edge_tracker[edge_key] = "a" if edge_data["is_aux"] else "m"
                else:
                    # Safety fallback for invalid manifolds
                    edge_tracker[edge_key] = "v"

    # Compile the final Crease Pattern
    final_edges = [(v1, v2, t) for (v1, v2), t in edge_tracker.items()]
    return Cp225(cp_verts, final_edges)


# =============================================================================
# Main Testing and Profiling
# =============================================================================


class FoldEvolver:
    """
    Wrapper object for driving the fold generation engine
    """
    def __init__(
        self,
        root_fold,
        raycast=True,
        bp=False,
        midpoints=False,
        components_to_flip="ONE",
    ):
        self.root = root_fold
        self.raycast = raycast
        self.bp = bp
        self.midpoints = midpoints
        self.components_to_flip = components_to_flip

        self.family_tree = root_fold.get_children(
            raycast=self.raycast,
            bp=self.bp,
            midpoints=self.midpoints,
            components_to_flip=self.components_to_flip,
        )
        # We store the "current" generation as a list of frozen states
        # Start with root, but we won't add root to the family_tree per your request
        self.current_generation = list(self.family_tree.keys())
        self.generation_count = 1

    def evolve(self, num_generations=1, select_n=None, select_percent=None):
        """
        Processes generations.
        select_n: Top N folds to keep as parents for the next gen.
        select_percent: Top X% of folds to keep as parents.
        """
        for _ in range(1, num_generations + 1):
            next_gen_candidates = {}  # {frozen_child: frozen_parent}

            print(f"--- Processing Generation {self.generation_count+1} ---")
            print(f"Parents in this gen: {len(self.current_generation)}")

            for i, frozen_parent in enumerate(self.current_generation):
                parent_fold = unfreeze(frozen_parent)
                # Generate children
                children = parent_fold.get_children(
                    raycast=True, bp=False, midpoints=False, components_to_flip="ONE"
                )

                for frozen_child, _ in children.items():
                    # Only track and process if we haven't seen this state before
                    if frozen_child not in self.family_tree:
                        # Map child to parent for tree tracking
                        next_gen_candidates[frozen_child] = frozen_parent

                if i % 10 == 0 and i > 0:
                    print(
                        f"Parent {i} processed... children found so far: {len(next_gen_candidates)}"
                    )

            # 1. Update the global family tree with this generation's findings
            self.family_tree.update(next_gen_candidates)

            # 2. Rank candidates by goodness to determine who moves to the next generation
            ranked_candidates = self._rank_candidates(list(next_gen_candidates.keys()))

            # 3. Apply Cutoff logic
            cutoff = self._get_cutoff_index(
                len(ranked_candidates), select_n, select_percent
            )
            self.current_generation = [
                fold for fold, score in ranked_candidates[:cutoff]
            ]
            self.generation_count += 1
            print(
                f"Gen {self.generation_count} complete. {len(next_gen_candidates)} unique children found."
            )
            print(
                f"Top goodness in gen: {ranked_candidates[0][1] if ranked_candidates else 0:.4f}"
            )

    def _rank_candidates(self, frozen_list, criteria="goodness"):
        """Unfreezes, scores, and sorts folds."""
        if criteria == "goodness":
            scored = []
            for frozen in frozen_list:
                f = unfreeze(frozen)
                scored.append((frozen, f.goodness()))
        elif criteria == "efficiency":
            scored = []
            for frozen in frozen_list:
                f = unfreeze(frozen)
                tree, _ = f.get_tree_and_packing(include_packing=False)
                total_length = sum(nx.get_edge_attributes(tree, "length").values())
                scored.append(
                    (frozen, f.goodness() / total_length if total_length > 0 else 0)
                )
        # Sort by goodness descending
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def _get_cutoff_index(self, total, n, percent):
        if n:
            return min(total, n)
        if percent:
            return max(1, math.floor(total * percent))
        return total  # Default to keeping all

    def get_top_folds(self, top_n=100, criteria="goodness"):
        """Returns the absolute best folds found across all generations."""
        ranked = self._rank_candidates(list(self.family_tree.keys()), criteria=criteria)
        return [unfreeze(f) for f, score in ranked[:top_n]]
    

BOUNDARY_CORNERS = {
    Vertex4D(-1, 0, -1, 0),
    Vertex4D(1, 0, -1, 0),
    Vertex4D(1, 0, 1, 0),
    Vertex4D(-1, 0, 1, 0),
}
cp = Cp225(
        vertices=list(BOUNDARY_CORNERS),
        edges=[
            (0, 1, "b"),
            (1, 2, "b"),
            (2, 3, "b"),
            (3, 0, "b"),
        ],
    )
ROOT = cp_to_fold(cp)
SQUARE_SIZE = 2

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    print("===== Start Test =====")


    
    evolver = FoldEvolver(ROOT)
    # Gen 2 : evolve everyone
    evolver.evolve()
    frozen = evolver.family_tree.keys()
    unfrozen = [unfreeze(f) for f in frozen]
    plot_multiple(unfrozen)

    # Gen 3: Only 50% move on to the next gen, based on goodness ranking
    evolver.evolve(num_generations=1, select_percent=0.5)

    # Gen 4: Only 30% move on
    # evolver.evolve(num_generations=1, select_n=300)

    # Gen 5: Only 10% move on
    # evolver.evolve(num_generations=1, select_n=300)

    # Plot final results
    top_folds = evolver.get_top_folds(100)
    plot_multi_state_grid(folds=top_folds)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")  # Sort by cumulative time
    stats.print_stats(20)  # Print the top  functions

    print("===== End Test =====")
