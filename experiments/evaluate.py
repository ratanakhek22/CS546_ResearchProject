"""
Computes quality metrics (Recall@k, NDCG@k) for all strategies in a run directory.
Recall@k uses a brute force index as ground truth.
NDCG@k uses BEIR qrels via pytrec_eval.

Usage:
    python evaluate.py --dataset scifact
    python evaluate.py --dataset scifact --run-id 2026-04-25_11-53-08
"""
import os
import json
import argparse
import numpy as np
import hnswlib
import pytrec_eval
from beir.datasets.data_loader import GenericDataLoader

EMBED_DIR = "../embeddings/"
RESULTS_DIR = "../results/"
DATA_DIR = "../datasets/"
TOP_K = 10

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="scifact")
    parser.add_argument("--run-id",  type=str, default=None, help="Timestamp dir name. Defaults to latest run.")
    return parser.parse_args()

def resolve_run_dir(dataset, run_id):
    base = os.path.join(RESULTS_DIR, dataset)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"No results found for dataset '{dataset}' at {base}")
    if run_id:
        path = os.path.join(base, run_id)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Run directory not found: {path}")
        return path
    runs = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    if not runs:
        raise FileNotFoundError(f"No run directories found in {base}")
    return os.path.join(base, runs[-1])

def load_embeddings(dataset):
    embed_dir = os.path.join(EMBED_DIR, dataset)
    corpus_embeddings = np.load(os.path.join(embed_dir, "corpus_embeddings.npy"))
    query_embeddings  = np.load(os.path.join(embed_dir, "query_embeddings.npy"))
    with open(os.path.join(embed_dir, "corpus_ids.json")) as f:
        corpus_ids = json.load(f)
    with open(os.path.join(embed_dir, "query_ids.json")) as f:
        query_ids = json.load(f)
    return corpus_embeddings, corpus_ids, query_embeddings, query_ids

def load_qrels(dataset):
    data_path = os.path.join(DATA_DIR, dataset)
    _, _, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return qrels

def build_brute_force(corpus_embeddings, corpus_ids):
    """
    Builds an exact brute force index over the corpus.
    Used as ground truth for Recall@k.
    """
    dim = corpus_embeddings.shape[1]
    n   = corpus_embeddings.shape[0]
    bf  = hnswlib.BFIndex(space="cosine", dim=dim)
    bf.init_index(max_elements=n)
    bf.add_items(corpus_embeddings, list(range(n)))
    return bf

def compute_recall(hnsw_results, bf_index, query_embeddings, query_ids, corpus_ids, k):
    """
    For each query, compares HNSW top-k results against brute force top-k.
    Recall@k = |hnsw_results ∩ brute_force_results| / k
    """
    recalls = []
    corpus_id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}

    for i, qid in enumerate(query_ids):
        # brute force ground truth
        bf_labels, _ = bf_index.knn_query(query_embeddings[i], k=k)
        bf_set = set(corpus_ids[label] for label in bf_labels[0])

        # hnsw results for this query
        hnsw_docs = set(hnsw_results.get(qid, {}).keys())

        if not bf_set:
            continue

        recall = len(hnsw_docs & bf_set) / k
        recalls.append(recall)

    return round(float(np.mean(recalls)), 4)

def compute_ndcg(retrieval_results, qrels, k):
    """
    Computes NDCG@k using pytrec_eval against BEIR qrels.
    """
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {f"ndcg_cut.{k}"}
    )
    scores = evaluator.evaluate(retrieval_results)
    ndcg_values = [v[f"ndcg_cut_{k}"] for v in scores.values()]
    return round(float(np.mean(ndcg_values)), 4)

def load_trial_results(run_dir):
    """
    Groups all trial JSONs by strategy.
    Returns {strategy: [trial_data, ...]}
    """
    grouped = {}
    for fname in os.listdir(run_dir):
        if not fname.endswith(".json") or fname.startswith("eval_"):
            continue
        with open(os.path.join(run_dir, fname)) as f:
            data = json.load(f)
        strategy = data.get("strategy", fname.split("_")[0])
        key = f"{strategy}_ef{data.get('ef_construction', 'unknown')}"
        grouped.setdefault(key, []).append(data)
    return grouped

def merge_trial_results(trials):
    """
    Merges retrieval_results across trials by taking the union,
    averaging scores for docs that appear in multiple trials.
    """
    merged = {}
    counts = {}
    for trial in trials:
        for qid, docs in trial.get("retrieval_results", {}).items():
            if qid not in merged:
                merged[qid] = {}
                counts[qid] = {}
            for doc_id, score in docs.items():
                merged[qid][doc_id]  = merged[qid].get(doc_id, 0) + score
                counts[qid][doc_id] = counts[qid].get(doc_id, 0) + 1

    # average scores across trials
    for qid in merged:
        for doc_id in merged[qid]:
            merged[qid][doc_id] /= counts[qid][doc_id]

    return merged

if __name__ == "__main__":
    args    = parse_args()
    run_dir = resolve_run_dir(args.dataset, args.run_id)
    run_id  = os.path.basename(run_dir)
    print(f"Evaluating run: {run_dir}")

    print("Loading embeddings...")
    corpus_embeddings, corpus_ids, query_embeddings, query_ids = load_embeddings(args.dataset)

    print("Loading qrels...")
    qrels = load_qrels(args.dataset)

    print("Building brute force index...")
    bf_index = build_brute_force(corpus_embeddings, corpus_ids)

    print("Loading trial results...")
    grouped = load_trial_results(run_dir)
    print(f"Strategies found: {list(grouped.keys())}")

    all_metrics = {}
    for strategy, trials in grouped.items():
        print(f"\nEvaluating {strategy} ({len(trials)} trial(s))...")

        merged_results = merge_trial_results(trials)

        recall = compute_recall(
            merged_results, bf_index,
            query_embeddings, query_ids, corpus_ids, k=TOP_K
        )
        ndcg = compute_ndcg(merged_results, qrels, k=TOP_K)

        metrics = {
            "strategy":   strategy,
            "num_trials": len(trials),
            f"recall@{TOP_K}": recall,
            f"ndcg@{TOP_K}":   ndcg,
        }
        all_metrics[strategy] = metrics
        print(f"  Recall@{TOP_K}: {recall}")
        print(f"  NDCG@{TOP_K}:   {ndcg}")

        out_path = os.path.join(run_dir, f"eval_{strategy}.json")
        with open(out_path, "w") as f:
            json.dump(metrics, f)

    print(f"\nAll metrics saved to {run_dir}")