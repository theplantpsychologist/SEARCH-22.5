import multiprocessing as mp
import time
import cProfile
import pstats
import math
import itertools
from wakepy import keep

from sqlalchemy import create_engine, Column, Integer, LargeBinary, Boolean, String, text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.sqlite import insert

import networkx as nx
from z3 import Solver, Bool, And, Or, Not, If, Implies, BoolVal, sat, is_true

from engine.topology225 import apply_transform, is_valid_tiling_global, lex_le, plot_multiple_graphs

Base = declarative_base()

# =============================================================================
# DATABASE MODELS
# =============================================================================

class Prefix(Base):
    __tablename__ = 'prefixes'
    id = Column(Integer, primary_key=True)
    bits = Column(String, nullable=False, unique=True) # e.g. '10110'
    is_done = Column(Boolean, default=False)

class State(Base):
    __tablename__ = 'states'
    id = Column(Integer, primary_key=True)
    prefix_id = Column(Integer) 
    binary_state = Column(LargeBinary, nullable=False, unique=True) # <--- ADDED unique=True
# =============================================================================
# BINARY COMPRESSION & EDGE UTILS
# =============================================================================

def get_ordered_internal_edges(N):
    """Returns a strictly sorted list of all possible internal edges."""
    edges = []
    # Orthogonal
    for i in range(N):
        for j in range(1, N):
            edges.append(tuple(sorted(((i, j), (i+1, j)))))
            edges.append(tuple(sorted(((j, i), (j, i+1)))))
    # Diagonals
    for x in range(N):
        for y in range(N):
            edges.append(tuple(sorted(((x, y), (x+1, y+1)))))
            edges.append(tuple(sorted(((x+1, y), (x, y+1)))))
    return sorted(edges)

def get_boundary_edges(N):
    return [tuple(sorted(((i, 0), (i+1, 0)))) for i in range(N)] + \
           [tuple(sorted(((i, N), (i+1, N)))) for i in range(N)] + \
           [tuple(sorted(((0, i), (0, i+1)))) for i in range(N)] + \
           [tuple(sorted(((N, i), (N, i+1)))) for i in range(N)]

def compress_edges(active_internal_edges, ordered_edges):
    """Compresses a list of active edges into a raw byte string."""
    active_set = set(active_internal_edges)
    bit_string = "".join("1" if e in active_set else "0" for e in ordered_edges)
    # Convert bit string to bytes
    num_bytes = (len(ordered_edges) + 7) // 8
    return int(bit_string, 2).to_bytes(num_bytes, byteorder='big')

def decompress_edges(binary_blob, N):
    """Reconstructs the full edge list (including boundaries) from a byte string."""
    ordered_edges = get_ordered_internal_edges(N)
    num_bits = len(ordered_edges)
    
    val = int.from_bytes(binary_blob, byteorder='big')
    bit_string = bin(val)[2:].zfill(num_bits)
    
    edges = []
    for i, bit in enumerate(bit_string):
        if bit == '1':
            edges.append(ordered_edges[i])
            
    edges.extend(get_boundary_edges(N))
    return edges


def extract_topology(state_id,db_name=None, N = 4):
    if db_name is None:
        db_name = f'topologies_{N}_diag.db' # Default to diagonal symmetry for extraction
    engine = create_engine(f'sqlite:///database/tilings/storage/{db_name}')
    Session = sessionmaker(bind=engine)
    session = Session()
    
    state = session.query(State).filter_by(id=state_id).first()
    if not state:
        print(f"No state found with ID {state_id}")
        return None
    
    edges = decompress_edges(state.binary_state, N=N)

    G = nx.Graph()
    G.add_edges_from(edges)
    
    # 3. FIX: Manually re-assign the 'pos' attribute to every node
    # This tells the plotter that node (x, y) is located at (x, y)
    pos = {node: node for node in G.nodes()}
    nx.set_node_attributes(G, pos, 'pos')
    return G

# =============================================================================
# MULTIPROCESSING: Z3 WORKER
# =============================================================================

def z3_worker(task_queue, result_queue, N, symmetry):
    ordered_edges = get_ordered_internal_edges(N)
    boundary_edges = get_boundary_edges(N)
    
    while True:
        task = task_queue.get()
        if task is None: 
            break # Poison pill received
        
        prefix_id, prefix_bits = task
        
        # 1. Initialize Z3 Solver
        s = Solver()
        edge_vars = {e: Bool(f"e_{i}") for i, e in enumerate(ordered_edges)}
        
        # 2. APPLY THE PREFIX (Cube & Conquer)
        for i, bit in enumerate(prefix_bits):
            s.add(edge_vars[ordered_edges[i]] == (bit == '1'))

        # 3. CORE Z3 CONSTRAINTS (Copied from tiling225.py)
        
        # Planarity
        cells = {}
        for x in range(N):
            for y in range(N):
                d1 = tuple(sorted(((x, y), (x+1, y+1))))
                d2 = tuple(sorted(((x+1, y), (x, y+1))))
                s.add(Not(And(edge_vars[d1], edge_vars[d2])))

        # No Dangling Edges (Degree != 1)
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

        # Symmetry linking
        if symmetry != 'none':
            t_type = 6 if symmetry == 'diag' else 4
            for e, var in edge_vars.items():
                re = tuple(sorted((apply_transform(e[0], N, t_type), apply_transform(e[1], N, t_type))))
                if re in edge_vars:
                    s.add(var == edge_vars[re])

        # Strict T-Junctions
        for x in range(1, N):
            for y in range(1, N):
                neighbors = [
                    (x+1, y), (x+1, y+1), (x, y+1), (x-1, y+1),
                    (x-1, y), (x-1, y-1), (x, y-1), (x+1, y-1)
                ]
                e_vars = [edge_vars[tuple(sorted(((x,y), nbr)))] for nbr in neighbors]
                pairs = [(e_vars[0], e_vars[4]), (e_vars[2], e_vars[6]), 
                         (e_vars[1], e_vars[5]), (e_vars[3], e_vars[7])]
                any_straight = Or(*[And(p[0], p[1]) for p in pairs])
                all_paired = And([p[0] == p[1] for p in pairs])
                s.add(Implies(any_straight, all_paired))

        # Native Isomorphism Rejection
        base_vars = [edge_vars[e] for e in ordered_edges]
        for t_type in range(1, 8):
            transformed_vars = []
            for e in ordered_edges:
                tu, tv = apply_transform(e[0], N, t_type), apply_transform(e[1], N, t_type)
                re = tuple(sorted((tu, tv)))
                transformed_vars.append(edge_vars[re] if re in edge_vars else BoolVal(True))
            s.add(lex_le(base_vars, transformed_vars))
        
        # 4. SOLVE AND COMPRESS
        found_states = []
        models_checked = 0
        BATCH_SIZE = 50  # Flush to database frequently
        
        while s.check() == sat:
            model = s.model()
            models_checked += 1
            
            current_active = []
            block_clause = []
            
            # FIX: Safely unpack both the edge tuple (e) and the Z3 variable (v)
            for e, v in edge_vars.items():
                is_active = is_true(model.evaluate(v, model_completion=True))
                if is_active:
                    current_active.append(e)
                
                # Optimized Blocking: Add to the block list
                block_clause.append(v != is_active)
            
            # Apply the blocking clause for the next iteration
            s.add(Or(*block_clause))
            
            # Global Filter
            if is_valid_tiling_global(current_active + boundary_edges):
                found_states.append(compress_edges(current_active, ordered_edges))
                
            # --- Intra-prefix batching and progress printouts ---
            if models_checked % 500 == 0:
                print(f"[Worker] Prefix {prefix_id} active... {models_checked} Z3 states evaluated.")
                
            if len(found_states) >= BATCH_SIZE:
                result_queue.put((prefix_id, found_states, False)) # False = prefix not done yet
                found_states = []
                
        # Send any remaining states and signal that this prefix is COMPLETE
        result_queue.put((prefix_id, found_states, True))
# =============================================================================
# MULTIPROCESSING: DATABASE WRITER
# =============================================================================

def db_writer(db_uri, result_queue, total_prefixes):
    engine = create_engine(db_uri)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    processed = 0
    t_start = time.time()
    
    while True:
        res = result_queue.get()
        if res is None: break # Poison pill
        
        # Unpack the new 3-part message
        prefix_id, states, is_complete = res
        
        # 1. Bulk save NEWLY discovered topologies incrementally
        if states:
            # Use SQLite's INSERT OR IGNORE to prevent duplicates if a prefix is restarted
            stmt = insert(State).values([{"prefix_id": prefix_id, "binary_state": s} for s in states])
            stmt = stmt.on_conflict_do_nothing(index_elements=['binary_state'])
            session.execute(stmt)
            session.commit() # Safely committed to disk immediately!
            
        # 2. Mark Prefix as done ONLY when the worker signals it has exhausted the prefix
        if is_complete:
            session.query(Prefix).filter_by(id=prefix_id).update({"is_done": True})
            session.commit()
            
            processed += 1
            elapsed = time.time() - t_start
            print(f">>> [DB Writer] Progress: [{processed}/{total_prefixes}] Prefixes Exhausted | Time: {elapsed:.1f}s")
# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

if __name__ == "__main__":
    mp.freeze_support() # Safe execution on Windows
    

    # --- CONFIGURATION ---
    N = 4
    symmetry = "none"
    prefix_length = 8 # Number of edges to hardcode (2^8 = 256 parallel chunks)
    # ---------------------
    
    db_uri = f'sqlite:///database/tilings/storage/topologies_{N}_{symmetry}.db'
    engine = create_engine(db_uri)
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    session.execute(text("PRAGMA journal_mode=WAL;"))
    session.execute(text("PRAGMA synchronous=NORMAL;")) # Speeds up WAL writing



    #  Extract 100 random states and render them
    print("Extracting 100 random states for visualization...")
    all_states = session.query(State).all()
    if len(all_states) >= 100:
        import random
        random_states = random.sample(all_states, 100)
    else:
        random_states = all_states
    
    graphs = []
    for state in random_states:
        # 1. Reconstruct edge list from the compressed binary blob
        edges = decompress_edges(state.binary_state, N)
        
        # 2. Create the graph object
        G = nx.Graph()
        G.add_edges_from(edges)
        
        # 3. FIX: Manually re-assign the 'pos' attribute to every node
        # This tells the plotter that node (x, y) is located at (x, y)
        pos = {node: node for node in G.nodes()}
        nx.set_node_attributes(G, pos, 'pos')
        
        graphs.append(G)
    
    print(f"Rendering {len(graphs)} graphs...")
    plot_multiple_graphs(graphs, filename = f"renders/db_sample_n{N}_{symmetry}.png")
    breakpoint()


    
    with keep.running():
        print("===== Start Database Build =====")
        
        # 1. Initialize the Prefixes (if starting fresh)
        if session.query(Prefix).count() == 0:
            print(f"Generating 2^{prefix_length} prefixes for Cube-and-Conquer...")
            combinations = list(itertools.product(['0', '1'], repeat=prefix_length))
            prefix_objects = [Prefix(bits="".join(c)) for c in combinations]
            session.bulk_save_objects(prefix_objects)
            session.commit()
            
        pending_prefixes = session.query(Prefix).filter_by(is_done=False).all()
        total_pending = len(pending_prefixes)
        
        if total_pending == 0:
            print("Database is already complete!")
            exit()
            
        print(f"Resuming operation: {total_pending} prefixes left to process.")
        
        # 2. Setup Queues and Processes
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        
        num_cores = max(1, mp.cpu_count() - 1) # Leave 1 core for DB/OS
        print(f"Spinning up {num_cores} Z3 Worker processes...")
        
        writer_proc = mp.Process(target=db_writer, args=(db_uri, result_queue, total_pending))
        writer_proc.start()
        
        worker_procs = []
        for _ in range(num_cores):
            p = mp.Process(target=z3_worker, args=(task_queue, result_queue, N, symmetry))
            p.start()
            worker_procs.append(p)
            
        # 3. Feed the tasks
        for prefix in pending_prefixes:
            task_queue.put((prefix.id, prefix.bits))
            
        # Send poison pills to workers
        for _ in range(num_cores):
            task_queue.put(None)
            
        # Wait for workers to finish
        for p in worker_procs:
            p.join()
            
        # Send poison pill to writer
        result_queue.put(None)
        writer_proc.join()
        
        final_count = session.query(State).count()
        print("===== End Database Build =====")
        print(f"Total Unique Topologies Generated: {final_count}")
        

