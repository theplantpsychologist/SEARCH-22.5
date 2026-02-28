"""
Helper functions related to tree handling
"""

import networkx as nx
import numpy as np
import os
import matplotlib.pyplot as plt
import math

def random_tree(n):
    tree = nx.random_labeled_tree(n)
    for edge in tree.edges():
        tree.edges[edge]["length"] = 1
        tree.edges[edge]["weight"] = 1.0 / tree.edges[edge]["length"]
    merge_edges(tree)[0]

    if n >=4 and len(tree.edges())==1:
        # want tree to be more than just 1 edge
        return random_tree(n)
    return tree

def merge_edges(tree):
    """
    Remove nodes of degree 2 by merging their two edges into one.
    Optimized to O(N) by calculating the merge list exactly once.
    """
    removed_nodes = set()
    
    # 1. Find all degree 2 nodes in a single pass
    to_merge = [n for n, d in tree.degree() if d == 2]
    
    # 2. Iterate through them without re-scanning the graph
    for n in to_merge:
        # Get neighbors before removing the node
        neighbors = list(tree.neighbors(n))
        
        # Safety check (optional but good practice)
        if len(neighbors) != 2:
            continue
            
        u, v = neighbors
        new_len = tree[u][n]["length"] + tree[n][v]["length"]
        
        # Mutate the graph
        tree.remove_node(n)
        tree.add_edge(u, v, length=new_len, weight=1.0 / new_len)
        
        removed_nodes.add(n)
        
    return tree, removed_nodes

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

def normalize_weights(tree):
    """
    Normalize edge lengths so that the total tree efficiency is 1.
    """
    total_length = sum(nx.get_edge_attributes(tree, 'length').values())
    if total_length == 0:
        return tree  # avoid division by zero
    for u, v in tree.edges():
        # tree.edges[u, v]['length'] /= total_length
        tree.edges[u, v]['weight'] = total_length / tree.edges[u, v]['length']
    return tree

def extract_eigenvalues(G, dim=8):
    """
    Computes the eigenvalues of the Laplacian matrix of graph G.
    The Laplacian L = D - A, where D is the degree matrix 
    and A is the adjacency matrix.
    """
    normalize_weights(G)
    L = nx.laplacian_matrix(G).toarray()
    eigenvalues = np.linalg.eigvalsh(L)
    raw_eigenvalues = np.sort(eigenvalues)[1:]  # exclude the zero eigenvalue
    if len(raw_eigenvalues) >= dim:
        return raw_eigenvalues[:dim]
    if len(raw_eigenvalues) < dim:
        padding = np.zeros(dim - len(raw_eigenvalues))
        return np.concatenate([raw_eigenvalues, padding])

def plot_trees(trees):
    n = len(trees)
    rows = math.ceil(math.sqrt(n / 2))
    cols = math.ceil(n / rows)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten() if n > 1 else [axes]
    for i, ax in enumerate(axes):
        ax.axis("off")
    for i, tree in enumerate(trees):
        ax = axes[i]
        pos = get_proportional_tree_pos(tree)
        nx.draw(tree, pos, with_labels=True, node_color='lightblue', edge_color='gray', ax=ax)
        ax.set_title(f"Tree {i}")
        ax.axis('equal')


    renders_dir = "renders"
    os.makedirs(renders_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(renders_dir) if f.endswith(".png")]
    file_count = len(existing_files)
    filename = f"trees_{file_count}.png"
    filepath = os.path.join(renders_dir, filename)
    plt.tight_layout(pad=0)
    plt.savefig(filepath)
    plt.close(fig)
    print(f"Saved render to {filepath}")