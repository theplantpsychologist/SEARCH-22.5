"""
Database is built from a queue of states who have no children and look like promising parents. If a parent gets to have children, we add all.

Then, control of the database growth is determined by how we search for promising parents.

"""

from sqlalchemy import asc, create_engine, Column, Integer, LargeBinary, ForeignKey, Float, BigInteger, Boolean, cast, desc, text, func
from sqlalchemy.orm import declarative_base,sessionmaker
import numpy as np
import os
from mmh3 import hash
import sys
import networkx as nx

import time
import cProfile
import pstats

import psutil
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

engine = create_engine('sqlite:///database/storage/database_midpoints.db')
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()
session.execute(text("PRAGMA journal_mode=WAL;"))


def expand_parent(fold: Fold225, parent_id=None):
    """
    Given a state, generate its children and add to the database.
    """
    children = fold.get_children(raycast=True, bp=False, midpoints=False, components_to_flip="ONE")
    if parent_id is not None:
        generation = session.query(State.generation).filter_by(id=parent_id).first()[0] + 1
    else:
        generation = 0

    for frozen_child in children:
        binary_child = np.array(frozen_child, dtype=np.int16).tobytes()
        hashed_child = hash(binary_child)
        if session.query(State).filter_by(hashed_state=hashed_child).first() is not None:
            continue  # already in database

        unfrozen_child = unfreeze(frozen_child)
        tree,_ = unfrozen_child.get_tree_and_packing()
        embedding = extract_eigenvalues(tree)
        tree_efficiency = sum(nx.get_edge_attributes(tree, 'length').values())
        layer_goodness = unfrozen_child.goodness()

        session.add(State(
            binary_state = binary_child,
            hashed_state = hashed_child,
            embedding = np.array(embedding, dtype=np.float32).tobytes(),
            parent_id = parent_id,
            generation = generation,
            tree_efficiency = tree_efficiency,
            layer_goodness = layer_goodness,
            has_children = False
        ))
    session.commit()
    session.expunge_all()
    return children




def view_best_states(n=100, target_embedding=None ):
    """
    View the top n states in the database according to some criteria.
    """
    best_states, distances = get_best_candidates(target_embedding, top_k=n)

    folds = [unfreeze(np.frombuffer(state.binary_state, dtype=np.int16)) for state in best_states]
    plot_multi_state_grid(folds, packing_instead_of_cp=True, labels = np.round(distances, decimals=3))

def current_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

def get_best_candidates(embedding = None, top_k=5):
    top_candidates = (
        session.query(State)
        .filter(State.has_children == False)
        .order_by(desc(
            # State.layer_goodness * 
            State.tree_efficiency * (3.0 / func.sqrt(State.generation + 1)) # SQL-side calculation
        ))
        .limit(3000 if embedding is not None else top_k)  # Grab the top N best parents
        .all()
    )
    if embedding is None:
        return top_candidates
    
    distances = []
    for candidate in top_candidates:
        s_sig = np.frombuffer(candidate.embedding, dtype=np.float32)
        dist = np.linalg.norm(embedding - s_sig) # Euclidean Distance
        distances.append((dist, candidate))

    # Smallest distance is the best match
    distances.sort(key=lambda x: x[0])
    return [candidate for _, candidate in distances[:top_k]], [dist for dist, _ in distances[:top_k]]

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()


    print("===== Start Test =====")
    size0 = session.query(State).count()
    t0 = time.time()
    print("Initial number of states in the database: ", size0)

    binary_root = np.array(canonicalize(ROOT), dtype=np.int16).tobytes()
    hashed_root = hash(binary_root)
    root_id = session.query(State.id).filter_by(hashed_state=hashed_root).first()


    if root_id is None:
        tree,_ = ROOT.get_tree_and_packing()
        embedding = [0] # special
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
    
    tree= random_tree(20)
    target_embedding = extract_eigenvalues(tree)

    random_trees = []
    parents_queue = []
    temp = t0
    distances = [0]
    while time.time() - t0 < 300:
        if time.time() - temp > 30:
            print("Switching to a new training tree")
            # time to switch and train on an increasingly growing new tree
            tree = random_tree(20 + len(random_trees))
            random_trees.append(tree)
            target_embedding = extract_eigenvalues(tree)
        if not parents_queue:
            # Add some parents to the queue
            top_candidates, distances = get_best_candidates(embedding=target_embedding, top_k=5)
            for candidate in top_candidates:
                parents_queue.append((candidate,candidate.binary_state, candidate.id))
            print(f"Total states: {session.query(State).count()} | Time elapsed: {time.time() - t0:.2f}s |  RAM usage: {current_memory_usage() / 1024 / 1024:.2f} MB | Best distance: {distances[0]:.4f}")
        else:
            candidate, binary_parent, parent_id = parents_queue.pop(0)
            children = expand_parent(unfreeze(np.frombuffer(binary_parent, dtype=np.int16)), parent_id=parent_id)
            candidate.has_children = True


    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")  # Sort by cumulative time
    stats.print_stats(20)  # Print the top  functions

    sizefinal = session.query(State).count()
    print("total number of states in the database: ", sizefinal)
    print(f"Added {sizefinal - size0} states in {time.time() - t0:.2f} seconds.")
    print("===== End Test =====")

    view_best_states(n=50, target_embedding=target_embedding)
    plot_trees(random_trees)

