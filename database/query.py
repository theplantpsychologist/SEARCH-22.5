import time
import cProfile
import pstats

from sqlalchemy import func
import numpy as np
import networkx as nx
import faiss
import torch

from src.engine.fold225 import unfreeze, plot_multi_state_grid
from src.engine.tree import extract_eigenvalues, random_tree, plot_trees

from database.build import DIMENSION, sync_faiss_with_db, session, index, faiss_to_db_id, State
from database.learning import AncestryEnsemble # Import your architecture


def view_best_matches(n=16, target_embedding=None ):
    """
    View the top n states in the database according to some criteria.
    """
    t0 = time.time()
    best_states, distances = find_closest_matches(target_embedding, top_k=n)
    tf = time.time()
    folds = [unfreeze(np.frombuffer(state.binary_state, dtype=np.int16)) for state in best_states]
    plot_multi_state_grid(folds, packing_instead_of_cp=True, labels = np.round(distances, decimals=3))
    print(f"Plotted top {len(folds)} states. Search time: {tf - t0:.2f}s")
    return best_states, distances



def find_closest_matches(target_embedding, top_k=8):
    """
    For final query. Returns the closest matches in the database to the given tree
    """
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


class AncestryPredictor:
    def __init__(self, model_path, embeddings, faiss_index):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Model
        self.model = AncestryEnsemble().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # 2. Load FAISS Index (Assuming you have one, or build it from DB)
        # For now, let's assume we load a raw numpy array of all embeddings in DB
        self.all_embeddings = embeddings
        self.index = faiss_index

    def predict(self, embedding):
        target_t = torch.tensor(embedding).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Predicted Ancestors: [1, 5, 32]
            return self.model(target_t).cpu().numpy()[0]

if __name__ == "__main__":
    
    print("===== Start database query =====")

    sync_faiss_with_db()
    ancestry_predictor = AncestryPredictor(model_path='database/storage/ancestry_ensemble.pth', embeddings=faiss_to_db_id, faiss_index=index)


    # Bird base or frog base
    # sqrt2 = np.float32(np.sqrt(2))
    # test_tree = nx.Graph()
    # test_tree.add_weighted_edges_from([
    #     (0, 1, 1.0), 
    #     (0, 2, sqrt2 + 1), 
    #     (0, 3, sqrt2 + 1),
    #     (0, 4, sqrt2 + 1),
    #     (0, 5, sqrt2 + 1),

    #     # (0, 6, sqrt2 + 1),
    #     # (0, 7, 1.0),
    #     # (0, 8, 1.0),
    #     # (1, 9, 1.0),
    # ], weight='length')
    # test_tree = random_tree(n=10)

    # best_states, distances = view_best_matches(n=16, target_embedding=extract_eigenvalues(test_tree, dim=DIMENSION))
    # plot_trees([test_tree])

    random_states = session.query(State)\
        .filter(State.generation == 11)\
        .order_by(func.random())\
        .limit(16)\
        .all()
    folds = [unfreeze(np.frombuffer(state.binary_state, dtype=np.int16)) for state in random_states]
    # plot_multi_state_grid(folds, packing_instead_of_cp=True)

    target_embedding = np.frombuffer(random_states[0].embedding, dtype=np.float32)
    ancestors = ancestry_predictor.predict(target_embedding)

    view_sequence(random_states[0].id)
    ancestor_matches = [random_states[0]]  # Start with the original state as the first "ancestor" for visualization
    for anc in ancestors:
        matches, distances = find_closest_matches(anc, top_k=1)
        ancestor_matches.append(matches[0])
        print(f"Ancestor match found with distance {distances[0]:.4f}")
        
    folds = [unfreeze(np.frombuffer(state.binary_state, dtype=np.int16)) for state in ancestor_matches]
    plot_multi_state_grid(folds, packing_instead_of_cp=True, labels=["Original","predicted ancestor 1", "predicted ancestor 2", "predicted ancestor 3", "predicted ancestor 4", "predicted ancestor 5"])
    print("===== End database query =====")
    