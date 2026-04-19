"""
Mid level (crease patterns) 22.5 functions and classes
"""

from src.engine.math225_core import Vertex4D, Fraction, AplusBsqrt2

BOUNDARY_CORNERS = {
    Vertex4D(-1, 0, -1, 0),
    Vertex4D(1, 0, -1, 0),
    Vertex4D(1, 0, 1, 0),
    Vertex4D(-1, 0, 1, 0),
}


class Cp225:
    """
    A custom variant of the FOLD class to represent crease patterns with 22.5 degree angles.

    This object is mutable ("unfrozen"). Use canonicalize() to freeze it in canonical form, or unfreeze() to convert back to mutable form.
    """

    def __init__(self, vertices, edges):
        self.vertices = vertices  # List of Vertex4D objects
        self.edges = edges  # Set of tuples (v1, v2, line_type) representing edges between vertices
        self.faces = []  # List of faces, where each face is a list of vertex indices
        # self.vertex_neighbors = self.get_vertex_neighbors()  # List of tuples (other_vertex_index, angle_in_22.5_degrees, line_type)

    # def __repr__(self):
    #     return f"Cp225 with {len(self.vertices)} vertices and {len(self.edges)} edges"
    def __str__(self):
        return f"Cp225, vertices: {self.vertices}\nedges:{self.edges}"

    # ============ Housekeeping methods ===========

    def render(self) -> list[tuple[str, float, float, float, float]]:
        """
        Convert similar to .cp file format: a list of edges where each edge is a list that contains the line type 'm','v','b','a' and the two vertices expressed as cartesian x1,y1,x2,y2 floats
        """
        rendered_edges = []
        for v1_idx, v2_idx, line_type in self.edges:
            v1 = self.vertices[v1_idx]
            v2 = self.vertices[v2_idx]
            x1, y1 = v1.to_cartesian()
            x2, y2 = v2.to_cartesian()
            rendered_edges.append((line_type, x1, y1, x2, y2))
        return rendered_edges

    def get_vertex_neighbors(self) -> list[list[tuple[int, int, str]]]:
        """
        Compute the neighbors for each vertex based on the edges.
        Cache result based on edges content.
        """
        # Create a simple cache key from edges
        edges_key = (len(self.edges), id(self.edges))

        if hasattr(self, "_neighbors_cache") and self._cache_key == edges_key:
            self.vertex_neighbors = self._neighbors_cache
            return self._neighbors_cache

        # Pre-allocate list with exact size
        neighbors = [[] for _ in range(len(self.vertices))]

        # Cache vertices list to avoid repeated attribute lookup
        vertices = self.vertices

        for v1_idx, v2_idx, line_type in self.edges:
            # Direct indexing instead of variable assignment when used once
            angle = vertices[v1_idx].angle_to(vertices[v2_idx])

            neighbors[v1_idx].append((v2_idx, angle, line_type))
            neighbors[v2_idx].append((v1_idx, (angle + 8) % 16, line_type))

        # Cache the result
        self._neighbors_cache = neighbors
        self._cache_key = edges_key
        self.vertex_neighbors = neighbors

        return neighbors

    def compute_faces(self) -> list[list[int]]:
        """
        Compute list of faces from vertex neighbor connectivity.
        Each face is represented as a list of edge indices into self.edges.
        """
        self.get_vertex_neighbors()
        faces = []
        visited = set()  # track directed edges (v1, v2) to avoid reprocessing

        # Build a quick lookup from unordered vertex pairs to global edge index
        edge_lookup = {
            frozenset((v1, v2)): idx for idx, (v1, v2, _) in enumerate(self.edges)
        }

        for v1, neighbors in enumerate(self.vertex_neighbors):
            for v2, _, _ in neighbors:
                if (v1, v2) in visited:
                    continue

                face_edge_indices = []
                curr_v, next_v = v1, v2

                while True:
                    visited.add((curr_v, next_v))

                    # Step 0: get global edge index
                    edge_idx = edge_lookup[frozenset((curr_v, next_v))]
                    face_edge_indices.append(edge_idx)

                    # Step 1: at next_v, find angle to curr_v
                    nbrs = self.vertex_neighbors[next_v]
                    nbrs_sorted = sorted(
                        nbrs, key=lambda x: x[1]
                    )  # sort neighbors by angle
                    idx = next(
                        i for i, (nbr, _, _) in enumerate(nbrs_sorted) if nbr == curr_v
                    )

                    # Step 2: move to next neighbor counterclockwise
                    next_idx = (idx + 1) % len(nbrs_sorted)
                    next2_v, _, _ = nbrs_sorted[next_idx]

                    # Step 3: check if face is closed
                    if (next_v, next2_v) == (v1, v2):
                        break

                    curr_v, next_v = next_v, next2_v

                faces.append(face_edge_indices)

        # remove the outer square face if present
        if len(faces) > 0:
            # the outer face contains all four boundary vertices
            outer_face = None
            for face in faces:
                face_vertices = {
                    self.vertices[self.edges[edge_idx][0]] for edge_idx in face
                } | {self.vertices[self.edges[edge_idx][1]] for edge_idx in face}
                if BOUNDARY_CORNERS.issubset(face_vertices):
                    outer_face = face
                    break
            if outer_face:
                faces.remove(outer_face)
        self.faces = faces
        return faces

    # ============ modify cp methods ===========

    def split_edge(self, edge_index: int, new_vertex: Vertex4D) -> None:
        """
        Split an edge at the specified index by adding a new vertex.
        The edge is replaced by two edges connecting to the new vertex.
        """
        if new_vertex in {
            self.vertices[self.edges[edge_index][0]],
            self.vertices[self.edges[edge_index][1]],
        }:
            print(
                "New vertex is identical to an existing endpoint; no split performed."
            )
            return  # no need to split if new vertex is same as existing endpoint
        if not point_on_line(
            self.vertices[self.edges[edge_index][0]],
            self.vertices[self.edges[edge_index][1]],
            new_vertex,
        ):
            raise ValueError("New vertex does not lie on the specified edge.")
        v1_idx, v2_idx, line_type = self.edges[edge_index]
        self.vertices.append(new_vertex)
        new_vertex_idx = len(self.vertices) - 1
        self.edges.pop(edge_index)
        self.edges.append((v1_idx, new_vertex_idx, line_type))
        self.edges.append((new_vertex_idx, v2_idx, line_type))
        self.get_vertex_neighbors()

    def ray_cast(self, start_idx: int, angle: int, operation_count=0) -> tuple[None, None] | tuple["Cp225", int]:
        """
        Cast a ray from vertex `start_idx` along `angle` (int, multiple of 22.5 degrees),
        find the first edge intersection, split that edge, and add a new crease.
        Assumes self.faces stores edge indices (global indices into self.edges).
        """

        self.compute_faces()
        start_vertex = self.vertices[start_idx]

        closest_dist = float("inf")
        hit_edge_idx = None
        hit_point = None

        # 1. Get all faces containing the start vertex
        candidate_faces = [
            face
            for face in self.faces
            if any(start_idx in self.edges[edge_idx][:2] for edge_idx in face)
        ]
        candidate_edges = set(edge_idx for face in candidate_faces for edge_idx in face)

        # Find hit point
        for edge_idx in candidate_edges:
            v1, v2, _ = self.edges[edge_idx]

            if start_idx in (v1, v2):
                continue  # skip edges sharing the same vertex

            p1, p2 = self.vertices[v1], self.vertices[v2]
            angle_edge = p1.angle_to(p2)

            # 2. Compute infinite-line intersection
            P = intersection(start_vertex, p1, angle, angle_edge)
            if P is None:
                continue

            # 3. Check if intersection is within the edge segment
            if not point_on_line(p1, p2, P):
                continue

            # 4. Check if intersection is in front of the ray
            if start_vertex.angle_to(P) != angle:
                continue

            # 5. Track closest intersection
            dist = start_vertex.distance_to(P)
            if dist < closest_dist:
                closest_dist = dist
                hit_edge_idx = edge_idx
                hit_point = P

        if hit_point is None:
            # print("Ray cast found no intersection for direction ", angle)
            return None, operation_count  # no intersection found

        new_vertex = hit_point
        if new_vertex in self.vertices:
            new_idx = self.vertices.index(new_vertex)
        else:
            angle_edge = self.vertices[self.edges[hit_edge_idx][0]].angle_to(
                self.vertices[self.edges[hit_edge_idx][1]]
            )
            self.split_edge(hit_edge_idx, new_vertex)
            new_idx = len(self.vertices) - 1
            if not vertex_on_border(self.vertex_neighbors[new_idx]):
                _, operation_count = self.ray_cast(
                    new_idx, reflect_angle(angle, angle_edge), operation_count
                )
        self.edges.append((start_idx, new_idx, "m"))
        return self, operation_count + 1

    def remove_crease(self, edge_index: int) -> tuple["Cp225", int]:
        """
        Recursively remove a crease (non-boundary edge) at the specified index.
        Chain the removal through any vertices that become collapsible (degree-3
        internal vertices where two creases are 180° apart).
        """
        operations = 0

        if edge_index >= len(self.edges):
            raise IndexError("Edge index out of range.")

        v1_idx, v2_idx, line_type = self.edges[edge_index]
        if line_type == "b":
            raise ValueError("Cannot remove a boundary edge.")

        # Remove the edge
        self.edges.pop(edge_index)
        operations += 1
        self.get_vertex_neighbors()

        def find_edge_index(a, b):
            """Return edge index connecting vertices a,b or None"""
            for i, (x, y, _) in enumerate(self.edges):
                if {x, y} == {a, b}:
                    return i
            return None

        def try_collapse(vertex_idx):
            """Check if a vertex qualifies for crease removal and recurse if so"""
            nbrs = self.vertex_neighbors[vertex_idx]
            if vertex_on_border(nbrs) or len(nbrs) != 3:
                return 0

            for i in range(3):
                for j in range(i + 1, 3):
                    if abs((nbrs[i][1] - nbrs[j][1]) % 16) == 8:
                        k = list({0, 1, 2} - {i, j})[0]
                        other_vertex = nbrs[k][0]

                        idx_to_remove = find_edge_index(vertex_idx, other_vertex)
                        if idx_to_remove is not None:
                            _, operations_ = self.remove_crease(idx_to_remove)
                            return operations_
                        return 0
            return 0

        operations += try_collapse(v1_idx)
        operations += try_collapse(v2_idx)
        # self.get_vertex_neighbors()

        return self, operations

    # =========== Flat foldability methods ===========

    def kawasaki_errors(self) -> list[int]:
        """
        Check if the crease pattern is flat foldable using Kawasaki's theorem.
        MV assignment, layer ordering, and self intersection are outside the scope of this project.
        Return a list of vertex indices that violate Kawasaki's theorem.
        """
        self.get_vertex_neighbors()
        errors = []
        for i, neighbors in enumerate(self.vertex_neighbors):
            if vertex_on_border(neighbors):
                continue  # border vertices are immune to flat foldability errors
            angles = [angle for _, angle, _ in neighbors]
            if not vertex_kawasaki(angles):
                errors.append(i)
        return errors
    
# =========== Helper functions ===========
X = Vertex4D(1, 0, 0, 0)
Y = Vertex4D(0, 1, 0, 0)
Z = Vertex4D(0, 0, 1, 0)
W = Vertex4D(0, 0, 0, 1)
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
HALF = Fraction(1, 2)

def reflect_angle(angle: int, axis: int) -> int:
    """
    Reflect an angle (in 22.5 degree increments) across a given axis (also in 22.5 degree increments).
    """
    if angle == axis:
        print(
            "Warning: Reflecting angle across itself results in the same angle."
        )  # allow this case
        # raise ValueError("Angle cannot be the same as the axis of reflection")
    return (2 * axis - angle + 8) % 16

def point_on_line(v1: Vertex4D, v2: Vertex4D, point: Vertex4D) -> bool:
    """
    Given two vertices v1 and v2, and a point, determine if the point lies on the line segment between v1 and v2
    """
    if point == v1 or point == v2:
        return True
    angle1 = point.angle_to(v1)
    angle2 = point.angle_to(v2)
    if angle1 is None or angle2 is None:
        return False
    return abs(angle1 - angle2) == 8

def intersection(
    v1: Vertex4D, v2: Vertex4D, angle1: int, angle2: int
) -> Vertex4D | None:
    """
    Given two vertices v1 and v2, and the angles (in 22.5 degree units) of the rays originating from them, compute the intersection point of the two rays. If the rays are parallel or do not intersect, return None
    """
    a1 = angle1 % 8
    a2 = angle2 % 8

    if a1 == a2:
        return None  # parallel rays
    if v1 == v2:
        return None  # same starting point, no unique intersection
    # dy and dx are defined as positive in the up/right direction from v1

    # diff = v2 - v1
    # dx = AplusBsqrt2(diff.y - diff.w, diff.x)
    # dy = AplusBsqrt2(diff.y + diff.w, diff.z)
    dx = AplusBsqrt2(v2.x - v1.x, HALF * ((v2.y - v2.w) - (v1.y - v1.w)))
    dy = AplusBsqrt2(v2.z - v1.z, HALF * ((v2.y + v2.w) - (v1.y + v1.w)))
    if a2 == 4:
        return v1 + dx * X + dx * TAN_225[a1] * Z
    if a1 == 4:
        return v1 + (dy - dx * TAN_225[a2]) * Z

    tan1 = TAN_225[a1]
    tan2 = TAN_225[a2]
    # x and y components of the angle2 ray
    angle2x = (dy - dx * tan1) / (tan1 - tan2)
    angle2y = tan2 * angle2x
    return v2 + angle2x * X + angle2y * Z

def vertex_on_border(neighbors: list[tuple[int, int, str]]) -> bool:
    """
    Is border if contains exactly 2 edges of type b (border)
    """
    border_count = sum(1 for _, _, line_type in neighbors if line_type == "b")
    if border_count == 2:
        return True
    if border_count == 0:
        return False
    raise ValueError(f"Vertex has {border_count} border edges")


def vertex_kawasaki(angles: list[int]) -> bool:
    """
    Check if the given list of angles (in 22.5 degree increments) satisfies Kawasaki's theorem.
    Returns true if vertex is flat foldable, false otherwise.
    """
    angles_sorted = sorted(angles)
    alternating_sum = sum(angles_sorted[1::2]) - sum(angles_sorted[0::2])
    return alternating_sum == 8

def freeze(fold: Cp225) -> tuple:
    """
    Return an immutable form of the crease pattern.
    Vertices are lexicographically sorted, edges remapped to new indices.
    """
    # Step 1: convert vertices into 8-tuples (use tuple comprehension, slightly faster)
    vertices = tuple(
        (
            vert.x.numerator,
            vert.x.denominator,
            vert.y.numerator,
            vert.y.denominator,
            vert.z.numerator,
            vert.z.denominator,
            vert.w.numerator,
            vert.w.denominator,
        )
        for vert in fold.vertices
    )

    # Step 2: sort vertices lexicographically and create index mapping
    # Use enumerate on sorted result instead of sorting tuples with indices
    sorted_indices = sorted(range(len(vertices)), key=vertices.__getitem__)

    # Step 3: rebuild vertex tuple in canonical order and create index map simultaneously
    frozen_vertices = tuple(vertices[i] for i in sorted_indices)
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}

    # Step 4: rebuild and sort edges with remapped indices in one pass
    # Pre-allocate list size for small performance gain
    num_edges = len(fold.edges)
    remapped_edges = [None] * num_edges

    for idx, (v1, v2, line_type) in enumerate(fold.edges):
        i1, i2 = index_map[v1], index_map[v2]
        # Use tuple comparison instead of if/swap
        remapped_edges[idx] = (min(i1, i2), max(i1, i2), line_type)

    # Step 5: sort edges canonically
    frozen_edges = tuple(sorted(remapped_edges))

    return (frozen_vertices, frozen_edges)

if __name__ == "__main__":
    pass
