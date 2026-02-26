"""
Mid level (crease patterns) 22.5 functions and classes
"""

from engine.math225_core import Vertex4D

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

    def __repr__(self):
        return f"Cp225 with {len(self.vertices)} vertices and {len(self.edges)} edges"

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


if __name__ == "__main__":
    pass
