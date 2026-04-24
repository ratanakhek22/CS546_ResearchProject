"""
Experiment script for hilbert curve insertions.
Saves the vector DB created for querying later (to test quality)
"""
import os
import json
import time
import numpy as np
import hnswlib
import argparse
from datetime import datetime
from hilbertcurve.hilbertcurve import HilbertCurve

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="scifact")
parser.add_argument("--trials", type=int, default=1)
args = parser.parse_args()

DATASET = args.dataset
NUM_TRIALS = args.trials

# default configs
DATASET = "scifact"
DATA_DIR = "datasets/"
EMBED_DIR = f"../embeddings/{DATASET}/"
RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RESULTS_DIR = f"../results/{DATASET}/{RUN_ID}/"
STRATEGY = "hilbert_curve"
SEED = 42
TARGET_DIM = 16 # 16 bit integers
TARGET_SIZE = pow(2, TARGET_DIM) - 1

HNSW_SPACE = "cosine"
HNSW_EF_CONSTRUCTION = 200
HNSW_M = 16
HNSW_EF_SEARCH = 50
TOP_K = 10

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

def fit_hilbert_curve(corpus_embeddings):
    """
    Maps each vector to the hilbert curve and sorts them in that order to 
    prep for the insertion step.
    """
    start = time.perf_counter()
    min_dim_vector = corpus_embeddings.min(axis=0) # find the min for each dim in vector space
    max_dim_vector = corpus_embeddings.max(axis=0) # find the max for each dim in vector space

    normalized_ratios = (corpus_embeddings - min_dim_vector) / (max_dim_vector - min_dim_vector) # ratios of all vectors

    # reformat vectors: float -> int
    int_corpus_embeddings = (normalized_ratios * TARGET_SIZE).astype(np.uint16)

    # feed data into hilbert curve
    hilbert_curve = HilbertCurve(TARGET_DIM, corpus_embeddings.shape[1]) # curve init
    distances = hilbert_curve.distances_from_points(int_corpus_embeddings.tolist())

    # get insertion order via sorting (return indices in insertion order)
    order = np.argsort(np.array(distances)) 
    preprocess_time = time.perf_counter() - start

    return order, preprocess_time

def build_index(corpus_embeddings, order):
    """
    Builds the HNSW indexing by inserting in order (hilbert curve) and logs the time
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

    for i in range(0, NUM_TRIALS):
        print(f"Starting Trial {i + 1}")
        SEED += 1

        print("Determining insertion order...")
        order, preprocess_time = fit_hilbert_curve(corpus_embeddings)
        print(f"Preprocess time: {preprocess_time:.4f}s")

        print("Building HNSW index...")
        index, build_time = build_index(corpus_embeddings, order)
        print(f"Build time: {build_time:.4f}s")

        print("Running queries...")
        results, retrieval_times = run_queries(index, query_embeddings, query_ids, corpus_ids)
        print(f"Avg retrieval time: {np.mean(retrieval_times)*1000:.4f}ms")

        save_results(i, results, build_time, retrieval_times, preprocess_time)