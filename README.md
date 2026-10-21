# GNN Rising Stars

A node-classification teaching challenge on a synthetic graph. Compare a feature-only Random Forest with a graph convolutional network, then submit predictions for automated scoring.

The repository includes the public node features and edges, starter baselines, a scoring script, and a leaderboard workflow.

## The task

Predict `target` for the nodes in `data/test.csv`. The graph comes from a stochastic block model with noisy node features and stronger connections within classes.

| File | Contents |
| :--- | :--- |
| [data/train.csv](data/train.csv) | Labeled training nodes and features |
| [data/test.csv](data/test.csv) | Test node IDs and features |
| [data/edges.csv](data/edges.csv) | Edges across the full graph |
| [starter_code/baseline.py](starter_code/baseline.py) | Random Forest baseline |
| [starter_code/baseline_gnn.py](starter_code/baseline_gnn.py) | Two-layer GCN baseline |

The starter GCN uses a transductive setup: it sees all node features and graph edges, while training uses only the designated training labels. Its validation nodes come from the labeled training set.

## Run a baseline

```bash
git clone https://github.com/MostafaAbdelmegeed/gnn-mini-comp.git
cd gnn-mini-comp
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on macOS/Linux. Then:

```bash
python -m pip install -r starter_code/requirements.txt
cd starter_code
python baseline.py
python baseline_gnn.py
```

Run the scripts from `starter_code/` because their input paths are relative to that directory. They write predictions into the repository's `submissions/` directory.

## Submit predictions

Fork the repository and commit a CSV such as `submissions/YourName_ModelName.csv`:

```csv
id,target
2,0
7,1
```

Include every published test ID exactly once. Predictions must use the same class labels as the training data. Open a pull request containing your CSV to trigger the scoring workflow.

The scoring script sorts IDs, checks them against the test labels, and computes **macro F1**. Hosted scoring depends on the maintainer's `GNN_CHALLENGE_SEED` configuration. See the [workflow](.github/workflows/gnn_scoring.yml) for its regeneration and submission paths.

## Recorded results

[LEADERBOARD.md](LEADERBOARD.md) contains the recorded scores and dates. The [visual leaderboard](leaderboard.html) is a standalone HTML page; download it and open it in a browser to view it. The README keeps the results in those files rather than maintaining a second score table.

## Local experiments

Keep the checked-in data intact when preparing a challenge submission. `data/generate_data.py` writes new training data, test data, edges, and local ground truth into `data/`. Its default seed is 42; a regenerated dataset may differ from the published challenge.

To explore the generator, use a separate checkout. The included `verify_challenge.ps1` is an experiment script that regenerates data before running baselines; it is not a read-only verification of the published files.

## Scope

This is an educational synthetic dataset, rather than a benchmark for claims about real-world graph performance. Its public generation procedure and node ordering are part of that limitation. Report validation settings and keep test-label reconstruction out of model comparisons.
