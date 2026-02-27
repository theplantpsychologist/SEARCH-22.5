"""
Database is built from a queue of states who have no children and look like promising parents. If a parent gets to have children, we add all.

Then, control of the database growth is determined by how we search for promising parents.

"""

from sqlalchemy import asc, create_engine, Column, Integer, LargeBinary, ForeignKey, Float, BigInteger, Boolean, cast, desc, text
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

engine = create_engine('sqlite:///database/storage/database.db')
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
        embedding = extract_laplacian_eigenvalues(tree)
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


def extract_laplacian_eigenvalues(G):
    """
    Computes the eigenvalues of the Laplacian matrix of graph G.
    The Laplacian L = D - A, where D is the degree matrix 
    and A is the adjacency matrix.
    """
    L = nx.laplacian_matrix(G).toarray()
    eigenvalues = np.linalg.eigvalsh(L)
    return np.sort(eigenvalues)


def view_best_states(n=100, ):
    """
    View the top n states in the database according to some criteria.
    """
    best_states = (
        session.query(State)
        .order_by(desc(State.layer_goodness * State.tree_efficiency))
        .limit(n)
        .all()
    )


    folds = [unfreeze(np.frombuffer(state.binary_state, dtype=np.int16)) for state in best_states]
    plot_multi_state_grid(folds, packing_instead_of_cp=True)

def current_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

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

    # view_best_states(n=100, criteria="tree_efficiency", worst=False)
    parents_queue = []
    while time.time() - t0 < 30:
        if not parents_queue:
            # Add some parents to the queue
            top_candidates = (
                session.query(State)
                .filter(State.has_children == False)
                .order_by(desc(State.layer_goodness * State.tree_efficiency))
                .limit(10)  # Grab the top N best parents
                .all()
            )
            for candidate in top_candidates:
                parents_queue.append((candidate,candidate.binary_state, candidate.id))
        else:
            candidate, binary_parent, parent_id = parents_queue.pop(0)
            children = expand_parent(unfreeze(np.frombuffer(binary_parent, dtype=np.int16)), parent_id=parent_id)
            candidate.has_children = True
        print(f"Queue size: {len(parents_queue)} | Total states: {session.query(State).count()} | Time elapsed: {time.time() - t0:.2f}s | Memory usage: {current_memory_usage() / 1024 / 1024:.2f} MB")


    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")  # Sort by cumulative time
    stats.print_stats(20)  # Print the top  functions

    sizefinal = session.query(State).count()
    print("total number of states in the database: ", sizefinal)
    print(f"Added {sizefinal - size0} states in {time.time() - t0:.2f} seconds.")
    print("===== End Test =====")

    view_best_states(n=100)
    breakpoint()

