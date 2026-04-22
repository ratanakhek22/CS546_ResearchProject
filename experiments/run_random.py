"""
Experiment script for randomized insertion.
Saves the vector DB created for querying later (to test quality)
"""
import os
import json
import time
import numpy as np
import hnswlib

DATASET = "scifact"
DATA_DIR = "datasets/"
EMBED_DIR = f"embeddings/{DATASET}/"
RESULTS_DIR = "results/"
STRATEGY = "random"
SEED = 42

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

def get_insertion_order(n):
    """
    Randomizer for insertion. Done by simple suffle of index positions which can
    be used whe inserting by indexing into the original embeded data.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed=SEED)
    order = np.arange(n)
    rng.shuffle(order)
    preprocess_time = time.perf_counter() - start
    return order, preprocess_time

def build_index(corpus_embeddings, order):
    """
    Builds the HNSW indexing by inserting in order (randomly) and logs the time
    take to complete the insertion part.
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

def save_results(results, build_time, retrieval_times, preprocess_time):
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
    out_path = os.path.join(RESULTS_DIR, f"{STRATEGY}.json")
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    print("Loading embeddings...")
    corpus_embeddings, corpus_ids, query_embeddings, query_ids = load_embeddings()

    print("Determining insertion order...")
    order, preprocess_time = get_insertion_order(len(corpus_ids))
    # print(f"Build time: {preprocess_time:.4f}s") # uncomment incase preprocessing is relevant
    print(f"Build time: {0:.4f}s") # hard-coded preprocessing time to 0 for random

    print("Building HNSW index...")
    index, build_time = build_index(corpus_embeddings, order)
    print(f"Build time: {build_time:.4f}s")

    print("Running queries...")
    results, retrieval_times = run_queries(index, query_embeddings, query_ids, corpus_ids)
    print(f"Avg retrieval time: {np.mean(retrieval_times)*1000:.4f}ms")

    save_results(results, build_time, retrieval_times, 0) # hard-coded preprocessing time to 0 for random