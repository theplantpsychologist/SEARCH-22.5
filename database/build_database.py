"""
Database is built from a queue of states who have no children and look like promising parents. If a parent gets to have children, we add all.

Then, control of the database growth is determined by how we search for promising parents.

"""

import time
import cProfile
import pstats
from wakepy import keep
import os
import concurrent.futures
from multiprocessing import Manager

from sqlalchemy import asc, create_engine, Column, Integer, LargeBinary, ForeignKey, Float, BigInteger, Boolean, cast, desc, text, func
from sqlalchemy.orm import declarative_base,sessionmaker
import numpy as np
from mmh3 import hash
import networkx as nx
import psutil
import faiss

from src.engine.fold225 import Fold225, ROOT, canonicalize, unfreeze, plot_multi_state_grid
from src.engine.tree import extract_eigenvalues, random_tree, plot_trees


Base = declarative_base()

class State(Base):
    __tablename__ = 'states'
    id = Column(Integer, primary_key=True)

    binary_state = Column(LargeBinary, nullable=False)
    hashed_state = Column(BigInteger, nullable=False, index=True)
    embedding = Column(LargeBinary, nullable=False)

    tree_efficiency = Column(Float, nullable=True)
    layer_goodness = Column(Float, nullable=True)
    parent_id = Column(Integer, ForeignKey('states.id'), nullable=True)

    has_children = Column(Boolean, default=False)
    generation = Column(Integer, default=0)
    # metadata = Column(JSON, nullable=True)

engine = create_engine('sqlite:///database/storage/database_3.db')
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()
session.execute(text("PRAGMA journal_mode=WAL;"))

# number of eigenvalues in embedding. Effectively the max number of tree nodes we can accurately match
DIMENSION = 32
index = faiss.IndexFlatL2(DIMENSION)

# We need to map FAISS internal IDs (0, 1, 2...) back to your SQLite IDs
faiss_to_db_id = []
def sync_faiss_with_db():
    global index, faiss_to_db_id
    # Fetch states that aren't in the index yet
    # (You would need a 'is_indexed' column or track the last ID)
    new_states = session.query(State.id, State.embedding).filter(State.id > len(faiss_to_db_id)).all()
    
    if not new_states:
        return

    embeddings = np.vstack([np.frombuffer(s.embedding, dtype=np.float32) for s in new_states])
    
    index.add(embeddings)
    for s in new_states:
        faiss_to_db_id.append(s.id)

def hot_sync_faiss():
    """Builds or updates the FAISS index from SQLite."""
    global faiss_to_db_id
    
    # Find the last ID we indexed (to avoid duplicates)
    last_indexed_id = faiss_to_db_id[-1] if faiss_to_db_id else -1
    
    # Fetch only ID and Embedding for anything new
    new_records = session.query(State.id, State.embedding)\
                         .filter(State.id > last_indexed_id).all()
    
    if new_records:
        embeddings = np.vstack([np.frombuffer(r.embedding, dtype=np.float32) for r in new_records])
        index.add(embeddings)
        faiss_to_db_id.extend([r.id for r in new_records])
        # print(f"FAISS Synced: {index.ntotal} total vectors.")


# 1. Isolated worker function (No database access here)
def compute_children_task(parent_id, binary_parent, generation):
    """
    Computes children and their metadata in parallel.
    Returns a list of data dictionaries to be inserted by the main process.
    """
    fold = unfreeze(np.frombuffer(binary_parent, dtype=np.int16))
    children_data = []
    
    # Compute children (The CPU-intensive part)
    frozen_children = fold.get_children(raycast=True, bp=False, midpoints=True, components_to_flip="ONE")
    
    for frozen_child in frozen_children:
        unfrozen_child = unfreeze(frozen_child)
        tree, _ = unfrozen_child.get_tree_and_packing()
        
        # Extract data
        binary_child = np.array(frozen_child, dtype=np.int16).tobytes()
        try:
            embedding = extract_eigenvalues(tree, dim=DIMENSION)
            tree_eff = sum(nx.get_edge_attributes(tree, 'length').values())
        except Exception:
            embedding = [0.0] * DIMENSION  # fallback for degenerate graphs
            tree_eff = 0.0
        goodness = unfrozen_child.goodness()
        
        children_data.append({
            "binary_state": binary_child,
            "hashed_state": hash(binary_child),
            "embedding": np.array(embedding, dtype=np.float32).tobytes(),
            "parent_id": parent_id,
            "generation": generation + 1,
            "tree_efficiency": tree_eff,
            "layer_goodness": goodness
        })
    return children_data

def main_parallel(time_limit=60, time_per_tree=10, tree_growth_rate = 1):
    t0 = time.time()
    num_workers = os.cpu_count() - 2 
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        # We'll use this to keep track of what is currently "out for delivery"
        processing_ids = set() 
        random_trees = []
        temp = t0 - time_per_tree
        distances = []
        
        with keep.running():
            while time.time() - t0 < time_limit:
                # 1. Keep FAISS fresh with any children added in the previous iteration
                hot_sync_faiss()
                # 1. REFILL LOGIC: Only query if we have room in the kitchen
                # If we have fewer than 'num_workers' tasks running, get more.
                if len(futures) < num_workers:
                    # Tree switching logic
                    if time.time() - temp >= time_per_tree:
                        tree = random_tree(8 + len(random_trees)* np.floor(len(random_trees) * tree_growth_rate).astype(int))
                        print(f"\n--- Switching to a new training tree with {len(tree.nodes())} nodes---")
                        random_trees.append(tree)
                        target_embedding = extract_eigenvalues(tree, dim=DIMENSION)
                        temp = time.time()
                        print(f"Total states: {session.query(State).count()} | Time elapsed: {time.time() - t0:.2f}s |  RAM usage: {current_memory_usage() / 1024 / 1024:.2f} MB")

                    # Find candidates that don't have children and aren't CURRENTLY being processed
                    top_candidates, distances = get_best_candidates_faiss(target_embedding=target_embedding, top_k=num_workers * 2)
                    
                    dispatched = 0
                    for candidate in top_candidates:
                        if candidate.id not in processing_ids:
                            candidate.has_children = True
                            processing_ids.add(candidate.id)
                            
                            fut = executor.submit(compute_children_task, candidate.id, candidate.binary_state, candidate.generation)
                            futures[fut] = candidate.id
                            dispatched += 1
                        
                        if len(futures) >= num_workers:
                            break
                    
                    if dispatched > 0:
                        session.commit() # Save the 'has_children = True' status
                        session.expunge_all() # Clear the candidate objects from RAM

                # 2. HARVEST RESULTS
                # Wait for at least one worker to finish. 
                # If everything is busy, this blocks for 0.5s to prevent "spinning".
                done, _ = concurrent.futures.wait(
                    futures.keys(), 
                    timeout=0.5, 
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                for fut in done:
                    try:
                        child_results = fut.result()
                        parent_id = futures.pop(fut)
                        processing_ids.remove(parent_id)

                        # 1. Batch check existing hashes
                        all_hashes = [data["hashed_state"] for data in child_results]
                        existing_hashes = {
                            h[0] for h in session.query(State.hashed_state)
                            .filter(State.hashed_state.in_(all_hashes)).all()
                        }

                        # 2. Add only new states
                        for data in child_results:
                            if data["hashed_state"] not in existing_hashes:
                                session.add(State(**data))
                        
                        session.commit()
                        session.expunge_all() 
                    except Exception as e:
                        print(f"Error: {e}")

                # 3. ANTI-SPIN: If we are fully loaded, take a tiny nap
                if len(futures) >= num_workers and not done:
                    time.sleep(0.1)


def view_best_matches(n=16, target_embedding=None ):
    """
    View the top n states in the database according to some criteria.
    """
    t0 = time.time()
    best_states, distances = get_best_candidates(target_embedding, top_k=n)
    tf = time.time()
    folds = [unfreeze(np.frombuffer(state.binary_state, dtype=np.int16)) for state in best_states]
    plot_multi_state_grid(folds, packing_instead_of_cp=True, labels = np.round(distances, decimals=3))
    print(f"Plotted top {len(folds)} states. Best distance: {min(distances):.4f} | Search time: {tf - t0:.2f}s")
    return best_states, distances

def current_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

def get_best_candidates(embedding=None, top_k=5):
    # Only fetch ID and Embedding to save IO/RAM
    query = (
        session.query(State.id, State.embedding)
        .filter(State.has_children == False)
        .order_by(desc(
            # State.layer_goodness * 0.5 +
            State.tree_efficiency * (3.0 / func.sqrt(State.generation + 1))
        ))
    )
    
    if embedding is None:
        results = query.limit(top_k).all()
        # Return full objects for the dispatcher
        return session.query(State).filter(State.id.in_([r.id for r in results])).all(), []

    # If searching by embedding, fetch more IDs/Embeddings
    candidates = query.limit(50000).all() 
    
    # Fast NumPy distance calculation on a pre-allocated array
    candidate_embeddings = np.stack([np.frombuffer(c.embedding, dtype=np.float32) for c in candidates])
    distances = np.linalg.norm(candidate_embeddings - embedding, axis=1)
    
    # Get top_k indices
    top_indices = np.argsort(distances)[:top_k]
    best_ids = [candidates[i].id for i in top_indices]
    best_dists = [distances[i] for i in top_indices]
    
    # Fetch full objects ONLY for the winners
    return session.query(State).filter(State.id.in_(best_ids)).all(), best_dists

def get_best_candidates_faiss(target_embedding, top_k=5):
    # Ensure index is up to date
    sync_faiss_with_db()
    
    if index.ntotal == 0:
        return [], []

    # FAISS expects a 2D array [1, DIMENSION]
    query_vec = np.array([target_embedding], dtype=np.float32)
    
    # Search: D is distances, I is the indices in the FAISS index
    # We search more than top_k so we can filter for 'has_children == False'
    D, I = index.search(query_vec, top_k * 5) 
    
    best_states = []
    best_distances = []
    
    for dist, idx in zip(D[0], I[0]):
        db_id = faiss_to_db_id[idx]
        candidate = session.query(State).get(db_id)
        # candidate = session.get(db_id, ident=db_id)  # More efficient than filter_by for primary key
        
        if not candidate.has_children:
            best_states.append(candidate)
            best_distances.append(dist)
            
        if len(best_states) >= top_k:
            break
            
    return best_states, best_distances
def find_closest_matches(tree, top_k=8):
    """
    Optimized: Uses FAISS to find the top_k closest matches across the ENTIRE database.
    """
    target_embedding = extract_eigenvalues(tree, dim=DIMENSION)
    sync_faiss_with_db() # Ensure we are searching the latest data

    if index.ntotal == 0:
        return [], []

    query_vec = np.array([target_embedding], dtype=np.float32).reshape(1, DIMENSION)
    D, I = index.search(query_vec, top_k)

    # Convert FAISS indices back to SQLite objects
    db_ids = [faiss_to_db_id[idx] for idx in I[0] if idx != -1]
    results = session.query(State).filter(State.id.in_(db_ids)).all()
    
    # Re-sort results to match FAISS distance order (SQL 'IN' doesn't preserve order)
    id_to_dist = {faiss_to_db_id[idx]: dist for idx, dist in zip(I[0], D[0])}
    results.sort(key=lambda x: id_to_dist.get(x.id, 999))

    return results, [id_to_dist[r.id] for r in results]



def view_sequence(state_id):
    sequence = []
    current_id = state_id
    
    while current_id is not None:
        state = session.query(State).get(current_id)
        sequence.append(state)
        current_id = state.parent_id
    
    sequence = list(reversed(sequence))
    folds = [unfreeze(np.frombuffer(state.binary_state, dtype=np.int16)) for state in sequence]
    plot_multi_state_grid(folds, packing_instead_of_cp=True)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()


    sqrt2 = np.float32(np.sqrt(2))
    test_tree = nx.Graph()
    test_tree.add_weighted_edges_from([
        (0, 1, 1.0), 
        (0, 2, sqrt2 + 1), 
        (0, 3, sqrt2 + 1),
        (0, 4, sqrt2 + 1),
        (0, 5, sqrt2 + 1),
        (0, 6, sqrt2 + 1),
        (0, 7, 1.0),
        (0, 8, 1.0),
        (1, 9, 1.0),
    ], weight='length')



    best_states, distances = view_best_matches(n=16, target_embedding=extract_eigenvalues(test_tree, dim=DIMENSION))
    plot_trees([test_tree])
    raise

    print("===== Start Test =====")
    size0 = session.query(State).count()
    t0 = time.time()



    size0 = session.query(State).count()
    print("Initial number of states in the database: ", size0)

    binary_root = np.array(canonicalize(ROOT), dtype=np.int16).tobytes()
    hashed_root = hash(binary_root)
    root_id = session.query(State.id).filter_by(hashed_state=hashed_root).first()

    if root_id is None:
        tree,_ = ROOT.get_tree_and_packing()
        embedding = [0] * DIMENSION  
        tree_efficiency = sum(nx.get_edge_attributes(tree, 'length').values())
        layer_goodness = ROOT.goodness()

        session.add(State(
            binary_state = binary_root,
            hashed_state = hashed_root,
            embedding = np.array(embedding, dtype=np.float32).tobytes(),
            parent_id = None,
            generation = 0,
            tree_efficiency = 0,
            layer_goodness = layer_goodness,
            has_children = False
        ))
        session.commit()
        print("Added root to database.")
        hot_sync_faiss()  # Sync FAISS after adding root
    


    main_parallel(time_limit=60 * 60 * 3, time_per_tree = 30, tree_growth_rate = 0.1)  

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")  # Sort by cumulative time
    stats.print_stats(20)  # Print the top  functions

    sizefinal = session.query(State).count()
    print("total number of states in the database: ", sizefinal)
    print(f"Added {sizefinal - size0} states in {time.time() - t0:.2f} seconds. Average states/s: {(sizefinal - size0) / (time.time() - t0):.2f}")

    test_tree = random_tree(15)
    best_states, distances = view_best_matches(n=16, target_embedding=extract_eigenvalues(test_tree, dim=DIMENSION))
    plot_trees([test_tree])

    view_sequence(best_states[0].id)

    print("===== End Test =====")
    

