import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import os

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def load_data():
    print("Loading data for GNN...")
    # 1. Load Nodes
    train_df = pd.read_csv('../data/train.csv')
    test_df = pd.read_csv('../data/test.csv')
    
    # Concatenate to get full node features
    # Note: test_df has NO target column.
    # We must ensure IDs are aligned.
    
    full_nodes = pd.concat([train_df, test_df], sort=False).sort_values('id').reset_index(drop=True)
    
    # Features
    feat_cols = [c for c in full_nodes.columns if c.startswith('feat_')]
    x = torch.tensor(full_nodes[feat_cols].values, dtype=torch.float)
    
    # Labels (only available for train nodes)
    # We need a mask.
    train_ids = train_df['id'].values
    test_ids = test_df['id'].values
    
    # Create mask based on ID indices
    # Assumes IDs are 0..N-1
    train_mask = torch.zeros(len(full_nodes), dtype=torch.bool)
    train_mask[train_ids] = True
    
    test_mask = torch.zeros(len(full_nodes), dtype=torch.bool)
    test_mask[test_ids] = True
    
    # Validation split from Train
    # We can't use sklearn split easily on masks, let's just take last 20% of train IDs for val
    num_train = len(train_ids)
    num_val = int(num_train * 0.2)
    val_ids = train_ids[-num_val:]
    real_train_ids = train_ids[:-num_val]
    
    train_mask = torch.zeros(len(full_nodes), dtype=torch.bool)
    train_mask[real_train_ids] = True
    
    val_mask = torch.zeros(len(full_nodes), dtype=torch.bool)
    val_mask[val_ids] = True
    
    # Targets
    # Fill unknown targets with -1
    y = torch.full((len(full_nodes),), -1, dtype=torch.long)
    
    # Map targets from train_df
    # We can iterate or use efficient pandas mapping
    # train_df index is not node ID.
    for _, row in train_df.iterrows():
        y[int(row['id'])] = int(row['target'])

    # 2. Load Edges
    edges_df = pd.read_csv('../data/edges.csv')
    edge_index = torch.tensor(edges_df.values.T, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.test_ids = test_ids
    
    return data

def run_gnn():
    data = load_data()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    # Model
    num_features = data.num_features
    # Number of classes: find max in y (ignoring -1)
    num_classes = data.y.max().item() + 1
    
    model = GCN(num_features, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    print("Training GCN...")
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            # Validation
            model.eval()
            pred = out.argmax(dim=1)
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
            print(f'Epoch {epoch}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')
            model.train()

    print("Evaluating on Test...")
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    # Extract test predictions
    # pred[data.test_mask] returns predictions for test nodes sorted by their index (ID)
    # because full_nodes was sorted by ID.
    test_preds = pred[data.test_mask].cpu().numpy()
    
    # We need to map these to the IDs. 
    # Since data.test_mask corresponds to sorted IDs, we just need the sorted test IDs.
    test_ids = sorted(data.test_ids)
    
    # Save Submission
    os.makedirs('../submissions', exist_ok=True)
    submission_path = os.path.join('..', 'submissions', 'gnn_submission.csv')
    
    submission_df = pd.DataFrame({
        'id': test_ids,
        'target': test_preds
    })
    
    # Sort by ID to be safe
    submission_df = submission_df.sort_values('id')
    
    submission_df.to_csv(submission_path, index=False)
    print(f"GNN Submission saved to {submission_path}")

if __name__ == "__main__":
    run_gnn()
