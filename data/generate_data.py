import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
import os

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

def generate_graph_data(n_samples=2000, n_features=50, n_classes=3, seed=42):
    """
    Generates a graph using Stochastic Block Model (SBM) to ensure strong homophily.
    Node features are noisy versions of the class center, making structure crucial.
    """
    print(f"Generating graph data (SBM) with {n_samples} nodes...")
    np.random.seed(seed)
    
    # 1. Define SBM Parameters
    # Strong diagonal (intra-class) probability, weak off-diagonal
    p_in = 0.05  # Probability of edge within same class
    p_out = 0.005 # Probability of edge between different classes
    
    sizes = [n_samples // n_classes] * n_classes
    # Adjust last class size for remainder
    sizes[-1] += n_samples - sum(sizes)
    
    probs = np.full((n_classes, n_classes), p_out)
    np.fill_diagonal(probs, p_in)
    
    # Generate Graph
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    
    # Get labels from SBM partition
    # Node indices are 0 to n_samples-1
    # We need to map them to classes based on the 'block' attribute if likely, 
    # but nx.stochastic_block_model assigns partitions in order of `sizes`.
    # Let's reconstruct labels explicitly.
    labels = []
    current_idx = 0
    for class_id, size in enumerate(sizes):
        labels.extend([class_id] * size)
    
    # 2. Generate Noisy Node Features
    # Create random centers for each class
    class_centers = np.random.randn(n_classes, n_features)
    
    X = np.zeros((n_samples, n_features))
    for i, label in enumerate(labels):
        # Feature = Class Center + High Gaussian Noise
        # High noise forces reliance on neighbor smoothing (GNN)
        noise_level = 10.0 
        X[i] = class_centers[label] + np.random.randn(n_features) * noise_level

    # 3. Create DataFrame
    feature_names = [f'feat_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = labels
    df['id'] = range(n_samples)
    
    # Reorder
    cols = ['id'] + feature_names + ['target']
    df = df[cols]
    
    # 4. Extract Edges
    edges = list(G.edges())
    edges_df = pd.DataFrame(edges, columns=['source_id', 'target_id'])
    
    print(f"Graph generated: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    return df, edges_df

def save_splits(df, edges_df):
    """
    Splits nodes into train/test. Edges are provided fully (transductive or inductive allowed).
    For simplicity in this challenge:
    - Train/Test split is on NODES.
    - Edges are provided for ALL nodes (standard semi-supervised/transductive setting).
    """
    # 70% Train, 30% Test (Need more test to see GNN benefits clearly? 70/30 is standard)
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    
    # Save Train (Nodes + Labels)
    train_path = os.path.join('data', 'train.csv')
    train_df.to_csv(train_path, index=False)
    print(f"Saved training nodes to {train_path}")
    
    # Save Test (Nodes only, no Labels)
    test_public = test_df.drop(columns=['target'])
    test_path = os.path.join('data', 'test.csv')
    test_public.to_csv(test_path, index=False)
    print(f"Saved test nodes to {test_path}")
    
    # Save Test Labels (Hidden)
    test_labels = test_df[['id', 'target']]
    labels_path = os.path.join('data', 'test_labels.csv')
    test_labels.to_csv(labels_path, index=False)
    print(f"Saved hidden test labels to {labels_path}")
    
    # Save Edges (Full Graph)
    edges_path = os.path.join('data', 'edges.csv')
    edges_df.to_csv(edges_path, index=False)
    print(f"Saved edge list to {edges_path}")

if __name__ == "__main__":
    df, edges = generate_graph_data()
    save_splits(df, edges)
