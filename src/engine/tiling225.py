import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import os

from src.engine.math225_core import (
    Fraction,
    Vertex4D,
    AplusBsqrt2,
    reflect,
    reflect_group,
    canonicalize_cpp,
    flatten_cpp,
    split_and_rebuild_cpp
)
from src.engine.cp225 import Cp225
from src.engine.fold225 import PLOT_COLORS

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

def rotate(v, rot_index, mirror=False):
    """
    Apply rotation and optional mirror symmetry to a Vertex4D.
    rot_index is in 22.5 degree units (0, 2, 4, 6, 8, 10, 12, 14).
    """
    rot_index %= 16
    # Direct mapping for rotations
    if rot_index == 0:
        p = v
    elif rot_index == 2:
        p = Vertex4D(-v.w, v.x, v.y, v.z)
    elif rot_index == 4:
        p = Vertex4D(-v.z, -v.w, v.x, v.y)
    elif rot_index == 6:
        p = Vertex4D(-v.y, -v.z, -v.w, v.x)
    elif rot_index == 8:
        p = Vertex4D(-v.x, -v.y, -v.z, -v.w)
    elif rot_index == 10:
        p = Vertex4D(v.w, -v.x, -v.y, -v.z)
    elif rot_index == 12:
        p = Vertex4D(v.z, v.w, -v.x, -v.y)
    elif rot_index == 14:
        p = Vertex4D(v.y, v.z, v.w, -v.x)
    else:
        p = v
    
    if mirror:
        mx = p.x
        my = -p.w
        mz = -p.z
        mw = -p.y
        p = Vertex4D(mx, my, mz, mw)
    
    return p


class Tile:
    """
    Tile object: describes the outline of a molecule, with ridge and hinge positions implied. Lengths are default normalized so that the sum of lengths is 1. Defined by a list of edge lengths and the normal angle for each length (pointing away out of the tile), and a center point, scale, rotation, and flip to define the tile's geometric transformation. Rotation is defined such that 0 corresponds to the first having a normal angle of 0. Flip means go ccw if false, else cw
    """
    def __init__(self,lengths: list[AplusBsqrt2], angles: list[int], center: Vertex4D = Vertex4D(0,0,0,0), scale: AplusBsqrt2 = None, rotation: int = 0, flip: bool = False):
        """
        Automatically canonicalize the lengths/angles, and store the scale rotation and flip to match the input
        """
        if len(lengths) != len(angles):
            raise ValueError(f"Lengths and angles must have the same length: {len(lengths)} != {len(angles)}")
        self.n = len(lengths)
        # if sum(angles) != 8 * (self.n - 2):
        #     raise ValueError(f"Invalid sum of angles: {angles}")
        perimeter = sum(lengths)
        
        self.lengths = [l / perimeter for l in lengths]  # Normalize lengths so that sum is 1
        self.angles = angles
        
        self.center = center

        if scale is None:
            self.scale = 1  # Default scale to normalize perimeter to 1. or should it be perimeter?
        else: 
            self.scale = scale

        self.rotation = rotation
        self.flip = flip

        #radius is scaled down to canonical size
        self.radius = sum(self.lengths) / (2*sum([
            TAN_225[(angles[i+1]-angles[i])//2] for i in range(self.n-1)
            ] + [TAN_225[((angles[0]-angles[-1])//2)%8]]
        ))


    def __hash__(self):
        """
        For dictionary lookup. In this context, we don't care about the tile's geometric transformation, just shape.
        """
        return hash(tuple(self.lengths + self.angles))
    def __eq__(self, other):
        pass

one = AplusBsqrt2(1,0)
# 16 unique convex molecules with one interior point
CORE_TILES = [
    Tile(lengths = [one,one,one,one,one,one,one,one], angles = [0,2,4,6,8,10,12,14]),
    Tile(lengths = [AplusBsqrt2(1,Fraction(1,2)),AplusBsqrt2(1,Fraction(1,2)), one, one, one, one, one], angles = [0,4,6,8,10,12,14]),
    Tile(lengths = [AplusBsqrt2(2,1),AplusBsqrt2(2,1),one,one,one,one], angles = [0,6,8,10,12,14]),
    Tile(lengths = [AplusBsqrt2(2,2),AplusBsqrt2(2,1),AplusBsqrt2(2,0),AplusBsqrt2(2,0),AplusBsqrt2(2,0),AplusBsqrt2(2,1)], angles = [0,4,6,8,10,12]),

    Tile(lengths = [AplusBsqrt2(2,1),AplusBsqrt2(2,1),AplusBsqrt2(2,1),AplusBsqrt2(2,1),AplusBsqrt2(2,0),AplusBsqrt2(2,0),], angles = [0,4,6,10,12,14]),
    Tile(lengths = [AplusBsqrt2(2,1),AplusBsqrt2(2,Fraction(3,2)),AplusBsqrt2(1,Fraction(1,2)),one,one], angles = [0,6,10,12,14]),
    Tile(lengths = [one,one,AplusBsqrt2(-2,2),one,one,AplusBsqrt2(-2,2)], angles = [0,4,6,8,12,14]),
    Tile(lengths = [AplusBsqrt2(2,1),AplusBsqrt2(2,1),AplusBsqrt2(1,Fraction(1,2)),AplusBsqrt2(1,Fraction(1,2)),one], angles = [0,6,8,12,14]),

    Tile(lengths = [AplusBsqrt2(2,2),AplusBsqrt2(2,2),AplusBsqrt2(2,1),AplusBsqrt2(2,0),AplusBsqrt2(2,1)], angles = [0,4,8,10,12]),
    Tile(lengths = [AplusBsqrt2(6,4),AplusBsqrt2(4,2),AplusBsqrt2(2,0),AplusBsqrt2(4,2)], angles = [0,6,8,10]),
    Tile(lengths = [AplusBsqrt2(2,2),AplusBsqrt2(2,1),AplusBsqrt2(2,1),AplusBsqrt2(2,1),AplusBsqrt2(2,1)], angles = [0,4,6,10,12]),
    Tile(lengths = [AplusBsqrt2(4,2),AplusBsqrt2(4,3),AplusBsqrt2(2,2),AplusBsqrt2(2,1)], angles = [0,6,10,14]),
    
    Tile(lengths = [AplusBsqrt2(4,3),AplusBsqrt2(4,3),AplusBsqrt2(2,1),AplusBsqrt2(2,1)], angles = [0,6,10,12]),
    Tile(lengths = [one,one,one,one], angles = [0,2,8,10]),
    Tile(lengths = [one,one,one,one], angles = [0,4,8,12]),
    Tile(lengths = [AplusBsqrt2(0,1),one,one], angles = [0,6,10]),
]

class Tiling225:
    """
    Tiling object: describes a tiling. Defined by a list of tiles and their connectivity (adjacency list). The adjacency list contains a list for each tile, where the ith element in list j corresponds to the index of the tile that tile j is connected to via its ith edge. If there is no tile connected to tile j via its ith edge, the value is -1. Note that this implies that the number of edges for each tile must be the same as the length of its adjacency list.
    """
    def __init__(self,tiles: list[Tile]=[] , adjacencies: list[list[int]]= []):
        self.tiles = tiles
        self.adjacencies = adjacencies

    def __str__(self):
        s = "Tiling225:\n"
        for i, tile in enumerate(self.tiles):
            s += f"Tile {i}: lengths={[str(l) for l in tile.lengths]}, angles={tile.angles}, center={tile.center}, scale={tile.scale}, rotation={tile.rotation}, flip={tile.flip}\n"
            s += f"  Adjacencies: {self.adjacencies[i]}\n"
        return s
    
    def join_tile(self, target_t, target_e, new_tile: Tile, new_e, new_flip = False):
        """
        Join a new tile to the tiling by connecting the new tile's new_e to the target tile's target_e

        Does not mutate the input tile, but will mutate self.
        """

        if self.adjacencies[target_t][target_e] != -1:
            return None

        # Disregard the new_tiles' rotation flip and center. recalculate based on the target tile.        
        target_tile = self.tiles[target_t]
        target_length = target_tile.lengths[target_e] * target_tile.scale
        new_s = target_length / new_tile.lengths[new_e]

        new_rot = (target_tile.rotation + target_tile.angles[target_e] - new_tile.angles[new_e] + 8) % 16

        r_angle1= ((target_tile.angles[(target_e+1)%target_tile.n] - target_tile.angles[target_e])//2)%16
        r_angle2= ((new_tile.angles[(new_e)%target_tile.n] - new_tile.angles[new_e-1])//2)%16
        new_center = (
            target_tile.center + 
            rotate(target_tile.radius*target_tile.scale * (X + Z * TAN_225[r_angle1%8]), target_tile.rotation + target_tile.angles[target_e] ) -
            rotate(new_tile.radius*new_s * (X - Z * TAN_225[r_angle2%8]), new_rot + new_tile.angles[new_e])

        )
        new_tile_copy = Tile(
            lengths = new_tile.lengths,
            angles = new_tile.angles,
            center = new_center,
            scale = new_s,
            rotation = new_rot,
            flip = new_flip
        )


        # 1. Add the new tile to the tiling
        new_tile_index = len(self.tiles)
        self.tiles.append(new_tile_copy)

        # 2. Update the adjacency list for the new tile
        while len(self.adjacencies) < len(self.tiles):
            self.adjacencies.append([-1] * self.tiles[len(self.adjacencies)-1].n)
        
        self.adjacencies[new_tile_index][new_e] = target_t
        self.adjacencies[target_t][target_e] = new_tile_index





        # 1. Generate new tile geometry for intersection testing
        # We use a standalone CP to get the exact vertices
        new_cp = tiling_to_cp(Tiling225(tiles=[new_tile_copy], adjacencies=[[-1]*new_tile_copy.n]))
        new_verts = new_cp.vertices[:new_tile_copy.n] # Polygon boundary only
        
        # Generate full tiling geometry for existing tiles
        existing_cp = tiling_to_cp(self)
        existing_verts = existing_cp.vertices

        # 2. Check for Overlap (Point in Polygon)
        # If any existing tile center is inside the new tile, or vice versa
        for t in self.tiles:
            if self._is_point_in_tile(t.center, new_verts):
                return None # Overlap detected
        
        # 3. Detect Secondary Adjacencies (Cycle Closure)
        # We look for edges in other tiles that match the new tile's edges
        auto_adjacencies = [-1] * new_tile_copy.n
        auto_adjacencies[new_e] = target_t # The primary connection

        for e_idx in range(new_tile_copy.n):
            if e_idx == new_e: continue
            
            v1_new = new_verts[e_idx]
            v2_new = new_verts[(e_idx + 1) % new_tile_copy.n]
            
            # Search all existing edges in the tiling
            for other_t_idx, other_tile in enumerate(self.tiles):
                # Get geometry for just this 'other' tile
                other_cp = tiling_to_cp(Tiling225(tiles=[other_tile], adjacencies=[[-1]*other_tile.n]))
                other_verts = other_cp.vertices[:other_tile.n]
                
                for other_e_idx in range(other_tile.n):
                    v1_old = other_verts[other_e_idx]
                    v2_old = other_verts[(other_e_idx + 1) % other_tile.n]
                    
                    # Check if vertices match (allowing for flipped orientation)
                    if (v1_new == v2_old and v2_new == v1_old):
                        # Line up check: Adjacency found!
                        auto_adjacencies[e_idx] = other_t_idx
                        # Update the existing tile's adjacency as well
                        self.adjacencies[other_t_idx][other_e_idx] = len(self.tiles)
                    elif self._lines_intersect(v1_new, v2_new, v1_old, v2_old):
                        # They touch but don't align perfectly (partial overlap/mismatch)
                        return None 

        # 4. Commit changes
        new_tile_index = len(self.tiles)
        self.tiles.append(new_tile_copy)
        self.adjacencies.append(auto_adjacencies)
        self.adjacencies[target_t][target_e] = new_tile_index
        
        return self



def plot_tilings(tilings: list[Tiling225]):
    """
    Plot a list of tilings. For now, just print out the tile shapes and adjacencies.
    Plot without hinges, just axial and ridges
    """
    n = len(tilings)
    rows = math.ceil(math.sqrt(n / 2))
    cols = math.ceil(n / rows)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten() if n > 1 else [axes]
    for i, tiling in enumerate(tilings):
        ax = axes[i]
        cp = tiling_to_cp(tiling)
        for edge in cp.render():
            l_type, x1, y1, x2, y2 = edge
            # Assumes PLOT_COLORS exists in your namespace
            ax.plot([x1, x2], [y1, y2], color=PLOT_COLORS[l_type], linewidth=3)

        ax.set_aspect("equal")
        ax.axis("off")



    renders_dir = "renders"
    os.makedirs(renders_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(renders_dir) if f.endswith(".png")]
    file_count = len(existing_files)
    filename = f"tilings_{file_count}.png"
    filepath = os.path.join(renders_dir, filename)
    plt.tight_layout(pad=0)
    plt.savefig(filepath)
    plt.close(fig)
    print(f"Saved render to {filepath}")


def tiling_to_cp(tiling: Tiling225) -> Cp225:
    """
    With flat foldable hinges. Can be converted into Fold225 afterwards.
    """
    vertices = []
    edges = []
    for t, tile in enumerate(tiling.tiles):
        n = tile.n
        r = tile.radius
        s = tile.scale
        sign = 1 if not tile.flip else -1

        # External angles dictate how much we turn AFTER drawing an edge
        # the ith angle is the angle between the ith and i+1th edge. Ignore the last one
        turns = [tile.angles[i+1] - tile.angles[i] for i in range(n-1)]
        sign = 1 if not tile.flip else -1
        
        start_offset = TAN_225[(tile.angles[1]-tile.angles[0]) // 2]
        
        current_heading = (tile.rotation + 4*sign) % 16
        tile_vertices = [
            tile.center + r*s* rotate(X + Z*start_offset, tile.rotation)
        ]
        
        # 3. Step through the edges
        for i in range(1,n):
            # Draw the current edge based on the current heading
            current_heading = (current_heading + sign*turns[i-1]) % 16
            
            tile_vertices.append(tile_vertices[-1] + rotate(X * tile.lengths[i] * s, current_heading))
            
        tile_v = []
        for vert in tile_vertices:
            if not vert in vertices:
                vertices.append(vert)
                tile_v.append(len(vertices)-1)
            else:
                tile_v.append(vertices.index(vert))
        for i,v_idx in enumerate(tile_v):
            next_v = tile_v[(i+1)%len(tile_v)]
            edge = (min(v_idx,next_v),max(v_idx,next_v),"b")
            if edge in edges:
                edges.remove(edge)
                edges.append( (min(v_idx,next_v),max(v_idx,next_v),"ax") )
            else:
                edges.append(edge)
        
        #tile center should not be in vertices. if it is, something is wrong
        vertices.append(tile.center)
        c = len(vertices)-1
        for v in tile_v:
            edges.append( (min(v,c), max(v,c), "r") )

    return Cp225(vertices=vertices, edges=edges)


def plot_tiling_debug(tilings: list[Tiling225]):
    """
    Enhanced debug plot for Tiling225:
    - Labels local edge indices on the inside of each tile.
    - Draws inscribed circles to verify tangency.
    - Draws and labels adjacency connections between tile centers.
    """
    n = len(tilings)
    rows = math.ceil(math.sqrt(n / 2))
    cols = math.ceil(n / rows)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 8))
    axes = axes.flatten() if n > 1 else [axes]

    for i, tiling in enumerate(tilings):
        ax = axes[i]
        
        # 1. Generate CP and plot base edges
        cp = tiling_to_cp(tiling)
        for edge in cp.render():
            l_type, x1, y1, x2, y2 = edge
            ax.plot([x1, x2], [y1, y2], color=PLOT_COLORS.get(l_type, 'black'), linewidth=1, alpha=0.4)

        # 2. Track connections to avoid double-drawing adjacency lines
        seen_connections = set()

        for t_idx, tile in enumerate(tiling.tiles):
            cx, cy = tile.center.to_cartesian()
            r_val = float(tile.radius * tile.scale)
            
            # Draw Inscribed Circle
            circle = patches.Circle((cx, cy), r_val, color='green', fill=False, linestyle='--', alpha=0.3)
            ax.add_patch(circle)
            ax.scatter([cx], [cy], color='red', s=10)

            # Generate this specific tile's vertices for local labeling
            # We use a dummy adjacency to get the geometry of just this tile
            temp_tiling = Tiling225(tiles=[tile], adjacencies=[[-1]*tile.n])
            temp_cp = tiling_to_cp(temp_tiling)
            
            # The first n vertices are the polygon boundary
            tile_verts = [v.to_cartesian() for v in temp_cp.vertices[:tile.n]]

            # 3. Label Local Edges on the Inside
            for e_idx in range(tile.n):
                v1 = tile_verts[e_idx]
                v2 = tile_verts[(e_idx + 1) % tile.n]
                
                # Edge midpoint
                mx, my = (v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2
                
                # Vector from midpoint to center to push label "inside"
                dx, dy = cx - mx, cy - my
                dist = math.sqrt(dx**2 + dy**2)
                offset = 0.15 * r_val # Adjust label depth inside the tile
                lx, ly = mx + (dx/dist)*offset, my + (dy/dist)*offset
                
                ax.text(lx, ly, str((e_idx+1) % tile.n), color='blue', fontsize=8, 
                        ha='center', va='center', fontweight='bold')

            # 4. Draw Adjacency Connections
            if t_idx < len(tiling.adjacencies):
                for e_idx, neighbor_idx in enumerate(tiling.adjacencies[t_idx]):
                    if neighbor_idx != -1:
                        # Ensure we label from lower index to higher or track carefully
                        conn_id = tuple(sorted((t_idx, neighbor_idx)))
                        
                        # Get neighbor center
                        neighbor_tile = tiling.tiles[neighbor_idx]
                        ncx, ncy = neighbor_tile.center.to_cartesian()
                        
                        # Draw connection line between centers
                        ax.plot([cx, ncx], [cy, ncy], color='orange', linewidth=1.5, linestyle=':', zorder=1)
                        
                        if conn_id not in seen_connections:
                            # Find which edge of the neighbor tile connects back to this tile
                            try:
                                n_edge_idx = tiling.adjacencies[neighbor_idx].index(t_idx)
                                mid_cx, mid_cy = (cx + ncx) / 2, (cy + ncy) / 2
                                label = f"({t_idx},e{e_idx})\n({neighbor_idx},e{n_edge_idx})"
                                ax.text(mid_cx, mid_cy, label, color='darkorange', fontsize=7,
                                        ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
                            except ValueError:
                                pass # Neighbor doesn't list this tile back (error in adjacency list)
                            
                            seen_connections.add(conn_id)

        ax.set_aspect("equal")
        ax.axis("on")
        ax.grid(True, linestyle=':', alpha=0.2)

    # Save logic...
    renders_dir = "renders"
    os.makedirs(renders_dir, exist_ok=True)
    filepath = os.path.join(renders_dir, "tiling_debug.png")
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close(fig)
    print(f"Saved connectivity debug to {filepath}")

if __name__ == "__main__":
    # Example usage
    # square = Tile(lengths=[AplusBsqrt2(1,0), AplusBsqrt2(1,0), AplusBsqrt2(1,0), AplusBsqrt2(1,0)], angles=[0,4,8,12])

    # triangle = Tile(lengths=[AplusBsqrt2(1,0), AplusBsqrt2(1,0), AplusBsqrt2(0,1)], angles=[0,4,10])
    
    # tiling = Tiling225(tiles=[square], adjacencies=[[-1,-1,-1,-1]])
    # tiling.join_tile(0,0, triangle, new_e = 0)
    # tiling.join_tile(1,1,square, new_e=1,)
    # tiling.join_tile(2,2,triangle, new_e = 0)
    # plot_tiling_debug([tiling])
    
    # plot_tiling_debug([Tiling225(tiles = [triangle], adjacencies = [])])

    plot_tiling_debug([
        Tiling225(tiles=[tile]) for tile in CORE_TILES
    ])