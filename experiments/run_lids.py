"""
Experiment script for replicating LID based inerstion.
Saves the vector DB created for querying later (to test quality)

Usage:
    python run_lids.py --dataset scifact
    python run_lids.py --dataset scifact --trials 5
"""
import os
import json
import time
import numpy as np
import hnswlib
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="scifact")
parser.add_argument("--efconstruction", type=int, default=32)
parser.add_argument("--trials", type=int, default=1)
parser.add_argument("--run-id", type=str, default=None) # for organizing tests into batches
args = parser.parse_args()

DATASET = args.dataset
NUM_TRIALS = args.trials

# default configs
DATA_DIR = "datasets/"
EMBED_DIR = f"../embeddings/{DATASET}/"
RUN_ID = args.run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RESULTS_DIR = f"../results/{DATASET}/{RUN_ID}/"
STRATEGY = "lids"
SEED = 42

HNSW_SPACE = "cosine"
HNSW_EF_CONSTRUCTION = args.efconstruction
HNSW_M = 16
HNSW_EF_SEARCH = 50
TOP_K = 10
LID_K = 100

def load_embeddings():
    """
    Loads the vectors from cache to work with. 
    """
    corpus_embeddings = np.load(os.path.join(EMBED_DIR, "corpus_embeddings.npy"))
    query_embeddings  = np.load(os.path.join(EMBED_DIR, "query_embeddings.npy"))
    with open(os.path.join(EMBED_DIR, "corpus_ids.json")) as f:
        corpus_ids = json.load(f)
    with open(os.path.join(EMBED_DIR, "query_ids.json")) as f:
        query_ids = json.load(f)
    return corpus_embeddings, corpus_ids, query_embeddings, query_ids

def build_brute_force(corpus_embeddings, corpus_ids):
    """
    Builds an exact brute force index over the corpus.
    Used as ground truth for Recall@k.
    """
    start = time.perf_counter()
    dim = corpus_embeddings.shape[1]
    n = corpus_embeddings.shape[0]
    bf = hnswlib.BFIndex(space="cosine", dim=dim)
    bf.init_index(max_elements=n)
    bf.add_items(corpus_embeddings, list(range(n)))
    preprocess_time = time.perf_counter() - start

    return bf, preprocess_time

def rank_LIDs(corpus_embeddings, bf_index):
    """
    Find the density of all vectors. Order them by high to low density and return the ordering. 
    """
    start = time.perf_counter()
    n = corpus_embeddings.shape[0]

    lid_scores = np.zeros(n) # for calculated scores
    for i in range(n):
        labels, distances = bf_index.knn_query(corpus_embeddings[i], k=101)
        distances = distances[0][1:]
        distances = np.maximum(distances, 1e-10)
        r_max = distances[-1]

        # calculate the density score for this vector center
        lid_scores[i] = -1.0 / np.mean(np.log(distances / r_max))

    # get sorted order
    order = np.argsort(lid_scores)[::-1]
    preprocess_time = time.perf_counter() - start

    return order, preprocess_time

def build_index(corpus_embeddings, order):
    """
    Builds the HNSW indexing by inserting in order (LID ranks) and logs the time
    it take to complete the insertion.
    """
    dim = corpus_embeddings.shape[1]
    n = corpus_embeddings.shape[0]

    index = hnswlib.Index(space=HNSW_SPACE, dim=dim) # alloc the vector space
    index.init_index(max_elements=n, ef_construction=HNSW_EF_CONSTRUCTION, M=HNSW_M)
    index.set_ef(HNSW_EF_SEARCH)

    start = time.perf_counter()
    for idx in order:
        index.add_items(corpus_embeddings[idx], idx) # vector + id
    build_time = time.perf_counter() - start

    return index, build_time

def run_queries(index, query_embeddings, query_ids, corpus_ids):
    """
    Goes through all possible queries (one per vector) and checks the time it takes.
    This is a test to measure the retrieval time metric and compare it to what BEIR
    expects in their baseline.
    """
    results = {}
    retrieval_times = []

    for i, qid in enumerate(query_ids):
        start = time.perf_counter()
        labels, distances = index.knn_query(query_embeddings[i], k=TOP_K)
        retrieval_times.append(time.perf_counter() - start)

        # each queries gets k closest documents formatted into similarit range
        results[qid] = {
            corpus_ids[label]: float(1 - distances[0][rank])
            for rank, label in enumerate(labels[0])
        }

    return results, retrieval_times

def save_results(trial_num, results, build_time, retrieval_times, preprocess_time):
    """
    Records the results of both the build and retrieval metrics to file.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "strategy": STRATEGY,
        "ef_construction": HNSW_EF_CONSTRUCTION,
        "preprocess_time_seconds": round(preprocess_time, 4),
        "build_time_seconds": round(build_time, 4),
        "total_time_seconds": round(preprocess_time + build_time, 4),
        "avg_retrieval_ms": round(np.mean(retrieval_times) * 1000, 4),
        "retrieval_results": results,
    }
    out_path = os.path.join(RESULTS_DIR, f"{STRATEGY}_{trial_num}.json")
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"Results saved to {out_path}\n")

if __name__ == "__main__":
    print("Loading embeddings...")
    corpus_embeddings, corpus_ids, query_embeddings, query_ids = load_embeddings()

    # calculate brute force once to save time, can do per trial in commented out lines
    bf_index, bf_preprocess_time = build_brute_force(corpus_embeddings, corpus_ids)

    for i in range(0, NUM_TRIALS):
        print(f"Starting Trial {i + 1}")
        SEED += 1
    
        print("Determining insertion order...")
        # bf_index, bf_preprocess_time = build_brute_force(corpus_embeddings, corpus_ids) # per trial version of bf_index step
        order, preprocess_time = rank_LIDs(corpus_embeddings, bf_index)
        print(f"Preprocess time: {(preprocess_time + bf_preprocess_time):.4f}s")

        print("Building HNSW index...")
        index, build_time = build_index(corpus_embeddings, order)
        print(f"Build time: {build_time:.4f}s")

        print("Running queries...")
        results, retrieval_times = run_queries(index, query_embeddings, query_ids, corpus_ids)
        print(f"Avg retrieval time: {np.mean(retrieval_times)*1000:.4f}ms")

        save_results(i, results, build_time, retrieval_times, preprocess_time)