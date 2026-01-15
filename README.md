# 🏆 GNN Rising Stars: Node Classification Challenge

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-orange)
![Status](https://img.shields.io/badge/Status-Active-success)
![Leaderboard](https://img.shields.io/badge/Leaderboard-Live-gold)

Welcome to the **GNN Rising Stars Mini-Competition**! Your mission is to master graph-based learning by outperforming standard baselines on a challenging synthetic dataset.

---

## 📑 Table of Contents
- [Overview](#-overview)
- [The Challenge](#-the-challenge)
- [Dataset](#-dataset)
- [Leaderboard](#-leaderboard)
- [Getting Started](#-getting-started)
- [Submission Guide](#-submission-guide)
- [Rules](#-rules)

---

## 🎯 Overview
This competition is designed to test your ability to leverage **structural information** in data. While tabular models (Random Forest, XGBoost) treat each data point in isolation, **Graph Neural Networks (GNNs)** use the connections between points to learn richer representations.

**Your Goal:** Achieve the highest **Macro F1 Score** on the hidden test set.

## 🧩 The Challenge
You are provided with a network of entities (nodes).
*   **Input:** Noisy feature vectors ($X$) and an Adjacency Matrix ($A$).
*   **Task:** Predict the class label ($Y$) for the unlabeled test nodes.
*   **Difficulty:** Features are purposely noisy ($\sigma=10.0$). Relying solely on features will result in poor performance (~40% F1). You **must** use the edges!

---

## 💾 Dataset
The dataset is generated via a **Stochastic Block Model (SBM)** with high homophily.

| File | Description |
| :--- | :--- |
| `data/train.csv` | **Features + Labels**. Use this to train your model. |
| `data/edges.csv` | **Graph Structure**. Contains source and destination node IDs. |
| `data/test.csv` | **Features Only**. You must predict labels for these nodes. |

> **⚠️ Warning:** Do NOT run `data/generate_data.py`. This uses a local default seed (42) which is DIFFERENT from the secret server seed used for scoring. If you overwrite your data, use `git checkout data/` to restore it.

---

## 🏆 Leaderboard
Check the **[Live Visual Leaderboard](leaderboard.html)** for the most up-to-date rankings with fancy visuals! 🎨

Or see the text version in [LEADERBOARD.md](LEADERBOARD.md).

| Rank | Model | Author | Macro F1 | Date |
| :--- | :--- | :--- | :--- | :--- |
| 🥇 | **GNN Baseline** | Starter Code | **0.902** | 2026-01-01 |
| 🥈 | Random Forest | Baseline | 0.415 | 2026-01-01 |

*(The leaderboard updates automatically when a Pull Request is merged!)*

---

## 🚀 Getting Started

### 1. Clone & Setup
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/gnn-mini-comp.git
cd gnn-mini-comp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install dependencies
pip install -r starter_code/requirements.txt
```

### 2. Run Baselines
We provide two starter scripts in `starter_code/`:
*   **`baseline.py`**: A simple Random Forest (no graph). Fast but weak.
*   **`baseline_gnn.py`**: A Graph Convolutional Network (GCN) using PyTorch Geometric. **Start here!**

```bash
cd starter_code
python baseline_gnn.py
# Output: submissions/gnn_submission.csv
```

---

## 📤 Submission Guide

We use **GitHub Actions** for automated scoring. Follow this workflow:

1.  **Fork** this repository.
2.  **Create a Branch**: `git checkout -b submission/my-new-model`
3.  **Generate Prediction**: Save your CSV to `submissions/`.
    *   **Format**: Must have `id` and `target` columns.
    *   **Naming**: `submissions/YourName_ModelName.csv`
4.  **Push & PR**:
    ```bash
    git add submissions/YourName_ModelName.csv
    git commit -m "New submission by YourName"
    git push origin submission/my-new-model
    ```
5.  **View Score**: Open a **Pull Request** to the `main` branch.
    *   🤖 The **GitHub Bot** will run your model against the hidden ground truth.
    *   Check the **PR Comments** to see your Macro F1 Score!

---

## 📜 Rules
1.  **No Cheating:** Do not try to reverse-engineer the seed.
2.  **Open Source:** All submissions must be committed to the repo.
3.  **Creativity:** You can use any library (PyG, DGL, NetworkX) or method (Link Prediction, Node Classification).

**Happy Modeling! 🧪**
