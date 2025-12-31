# GNN Mini-Competition: Synthetic Node Classification

## Overview
Welcome to the GNN Mini-Competition! Your goal is to achieve the highest Macro F1 Score on a synthetic node classification task.

While the data is provided in CSV format (nodes with features), it implies a graph structure. You are encouraged to infer or generate a graph structure (e.g., k-NN graph) to apply Graph Neural Networks (GNNs) for better performance, though standard ML methods (Random Forest, MLP) are provided as a baseline.

## Dataset
- `data/train.csv`: Training data with features `feat_0` to `feat_99` and `target` labels.
- `data/test.csv`: Test data with features only. You must predict the `target` for these IDs.

## Getting Started

1. **Setup Environment**:
   It is recommended to use a virtual environment.
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

   **Note**: Do NOT run `data/generate_data.py` unless you want to explore data generation. Doing so will overwrite the official competition data with the default local seed (42), making your checks against `test` invalid. If this happens, `git checkout data/` to restore.

2. **Install Dependencies**:
   ```bash
   pip install -r starter_code/requirements.txt
   ```

3. **Run Baselines**:
   The baseline script trains a Random Forest model and generates a submission file.
   ```bash
   cd starter_code
   python baseline.py
   ```
   This will create `submissions/sample_submission.csv`.

2b. **Run GNN Baseline** (Recommended):
   This script uses PyTorch Geometric to leverage the graph structure (`data/edges.csv`) and typically achieves higher scores.
   ```bash
   python baseline_gnn.py
   ```
   This will create `submissions/gnn_submission.csv`.

3. **Submit via GitHub**:
   This competition uses **GitHub Actions** for automated scoring and leaderboard updates.
   
   **Steps to Enter:**
   1. **Fork** this repository.
   2. **Create a Branch** for your submission (e.g., `feat/my-gnn-model`).
   3. **Generate your Prediction**: Save your valid CSV prediction (must have `id` and `target` columns) to the `submissions/` folder.
      - Naming Convention: `submissions/your_name_model.csv`.
   4. **Commit & Push**:
      ```bash
      git add submissions/your_name_model.csv
      git commit -m "Add submission for [Your Name]"
      git push origin feat/my-gnn-model
      ```
   5. **Open a Pull Request**:
      - Go to the original repository.
      - Click "Compare & pull request".
      - **Wait for the Bot**: A GitHub Action will run automatically.
      - **Check Comments**: The bot will post your **Macro F1 Score** as a comment on your PR!
   6. **Leaderboard**: If your score is good, merge the PR (or ask the maintainer to), and the `LEADERBOARD.md` will automatically update.

## Maintenance: Rotating the Secret Seed
If you need to change the ground truth (e.g., if the seed leaks), follow these 3 steps:

1.  **Regenerate Locally**:
    ```bash
    # Choose a NEW_SEED (e.g., 999999)
    # Windows PowerShell
    $env:GNN_CHALLENGE_SEED="999999"; python data/generate_data.py
    # Linux/Mac
    export GNN_CHALLENGE_SEED=999999; python data/generate_data.py
    ```
2.  **Push Changes**:
    Commit and push the updated `data/` files to GitHub.
    ```bash
    git add data/
    git commit -m "Rotate competition seed"
    git push
    ```
3.  **Update GitHub Secret**:
    Go to `Settings > Secrets and variables > Actions` in your repo and update `GNN_CHALLENGE_SEED` to match your new seed (`999999`).

## Scoring
To score your submission locally:
```bash
python scoring_script.py submissions/sample_submission.csv
```

## Rules
- You may use any library (PyTorch, TensorFlow, DGL, PyG, sklearn).
- You may generate your own graph structures from the node features.
