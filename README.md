# HNSW Insertion Strategy Research

Research project investigating the effect of vector insertion ordering on HNSW 
graph quality for approximate nearest neighbor search. Five strategies are 
compared — random ordering, K-Means clustering, Hilbert curve ordering, LID-based 
ordering, and a K-Means LID approximation — evaluated on BEIR benchmark datasets 
using Recall@10 and NDCG@10 as primary quality metrics.

## Project Structure
```
project/
├── embeddings/              # Cached embeddings per dataset (gitignored)
├── datasets/                # BEIR downloads (gitignored)
├── results/                 # Experiment results per dataset and run (gitignored)
├── reports/                 # Generated HTML reports
├── experiments/
│   ├── run_random.py           # Random insertion strategy (baseline)
│   ├── run_kmeans.py           # K-Means clustering insertion strategy
│   ├── run_hilbertcurve.py     # Hilbert curve insertion strategy
│   ├── run_lids.py             # LID-based insertion strategy
│   ├── run_approx_density.py   # K-Means LID approximation insertion strategy
│   ├── evaluate.py             # Computes Recall@10 and NDCG@10
│   └── run_experiments.bat     # Batch script to run all experiments
├── data_visualizer/
│   └── visualize.py            # Generates HTML report from results
├── embed.py                 # Computes and caches embeddings
└── requirements.txt         # Python dependencies
```

## Requirements

Python 3.10+ is recommended. Install dependencies with:

```bash
pip install -r requirements.txt
```

Key libraries:

- `hnswlib` — HNSW index construction and search
- `beir` — dataset loading and evaluation
- `sentence-transformers` — document and query embedding
- `scikit-learn` — K-Means clustering
- `hilbertcurve` — Hilbert curve position computation
- `matplotlib` — report chart generation

## Usage

### 1. Compute Embeddings

Run once per dataset before any experiments. Downloads the dataset and
computes and caches embeddings to `embeddings/{dataset}/`.

```bash
python embed.py --dataset scifact
python embed.py --dataset fiqa
```

### 2. Run Experiments

Run all five insertion strategies, evaluate, and generate a report in one 
command. Tests 200 and 32 ef, 1 report each. Run from the `experiments/` directory:

```bash
run_experiments.bat scifact 5
run_experiments.bat fiqa 5
```

Arguments:
- `dataset` — BEIR dataset name (e.g. `scifact`, `fiqa`, `nfcorpus`)
- `trials` — number of trials per strategy (recommended: 5-10)

To run a single strategy manually from the `experiments/` directory:

```bash
python run_random.py --dataset scifact --trials 5 --efconstruction 32
python run_kmeans.py --dataset scifact --trials 5 --efconstruction 32
python run_hilbertcurve.py --dataset scifact --trials 5 --efconstruction 32
python run_lids.py --dataset scifact --trials 5 --efconstruction 32
python run_density.py --dataset scifact --trials 5 --efconstruction 32
```

### 3. Evaluate Quality Metrics

Computes Recall@10 and NDCG@10 for all strategies in the latest run.
Run from the `experiments/` directory:

```bash
python evaluate.py --dataset scifact
```

To target a specific run:

```bash
python evaluate.py --dataset scifact --run-id 2026-04-25_11-53-08
```

### 4. Generate Report

Generates an HTML report averaging metrics across all trials.
Run from the `data_visualizer/` directory:

```bash
python visualize.py --dataset scifact
```

Report is saved to `reports/{dataset}_{run-id}.html`. Open in any browser.

## HNSW Parameters

The following parameters were held constant across all experiments to isolate 
insertion order as the independent variable:

| Parameter | Value |
|---|---|
| M | 16 |
| ef_construction | 200 or 32 |
| ef_search | 50 |
| Space | Cosine |
| Top-K | 10 |

## Datasets

Datasets are downloaded automatically by BEIR on first run.

| Dataset | Corpus | Queries | Domain |
|---|---|---|---|
| SciFact | ~5,000 | 300 | Scientific claims |
| FiQA | ~57,000 | 648 | Financial QA |

## Insertion Strategies

| Strategy | Description |
|---|---|
| Random | Vectors inserted in random order. Serves as the baseline. |
| K-Means | Cluster centroids inserted first, followed by cluster members. |
| Hilbert Curve | Vectors sorted by position along a Hilbert space-filling curve. |
| LID | Vectors sorted by Local Intrinsic Dimensionality descending using MLE with k=100 neighbors. High LID vectors inserted first. |
| Density Approximation | Vectors sorted by minimum distance to any k-means centroid descending. Approximates LID without O(n²) cost. |

## Results Summary

Random insertion consistently outperformed K-Means and Hilbert curve ordering 
on Recall@10 across all conditions. The recall gap widened under larger corpus 
size and lower ef_construction. NDCG@10 and retrieval time showed no meaningful 
differences across strategies. LID-based and density approximation results 
pending final experimental run.

## References

- Malkov & Yashunin (2018). Efficient and Robust Approximate Nearest Neighbor 
  Search Using Hierarchical Navigable Small World Graphs. IEEE TPAMI.
- Elliott et al. (2024). The Impacts of Data, Ordering, and Intrinsic 
  Dimensionality on Recall in Hierarchical Navigable Small Worlds. ACM SIGIR ICTIR.
- Thakur et al. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation 
  of Information Retrieval Models. NeurIPS.