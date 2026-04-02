"""
Sets up vectors for testing and experiements.

Initially loads a dataset from BEIR (unless its already cached)
and converts them into vectors for later operations on using a LM 
(specified by MODEL_NAME), storing them into the local cache.
"""

import os
import json
import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer

# Configs
DATASET = "scifact"
DATA_DIR = "datasets/"
EMBED_DIR = f"embeddings/{DATASET}/"
MODEL_NAME = "all-MiniLM-L6-v2"

def load_dataset():
    """
    Fetches the dataset we want to use/convert into vectors and store.
    """
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET}.zip"
    data_path = util.download_and_unzip(url, DATA_DIR)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels

def compute_and_cache_embeddings(corpus, queries):
    """
    Converts text from dataset to vectors to store, and records a mapping
    to keep track of document ID -> created vector.

    Also skips recreating vectors if the dataset was ran through
    the process already (is in our cache). 
    """
    os.makedirs(EMBED_DIR, exist_ok=True)

    corpus_embed_path = os.path.join(EMBED_DIR, "corpus_embeddings.npy")
    corpus_ids_path = os.path.join(EMBED_DIR, "corpus_ids.json")
    query_embed_path = os.path.join(EMBED_DIR, "query_embeddings.npy")
    query_ids_path = os.path.join(EMBED_DIR, "query_ids.json")

    model = SentenceTransformer(MODEL_NAME)

    if os.path.exists(corpus_embed_path):
        print("Loading cached corpus embeddings...")
        corpus_embeddings = np.load(corpus_embed_path)
        with open(corpus_ids_path) as f:
            corpus_ids = json.load(f)
    else:
        print("Computing corpus embeddings...")
        corpus_ids = list(corpus.keys())
        corpus_texts = [
            (corpus[cid]["title"] + " " + corpus[cid]["text"]).strip()
            for cid in corpus_ids
        ]
        corpus_embeddings = model.encode(
            corpus_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
        )
        np.save(corpus_embed_path, corpus_embeddings)
        with open(corpus_ids_path, "w") as f:
            json.dump(corpus_ids, f)
        print(f"Saved corpus embeddings: {corpus_embeddings.shape}")

    if os.path.exists(query_embed_path):
        print("Loading cached query embeddings...")
        query_embeddings = np.load(query_embed_path)
        with open(query_ids_path) as f:
            query_ids = json.load(f)
    else:
        print("Computing query embeddings...")
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        query_embeddings = model.encode(
            query_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
        )
        np.save(query_embed_path, query_embeddings)
        with open(query_ids_path, "w") as f:
            json.dump(query_ids, f)
        print(f"Saved query embeddings: {query_embeddings.shape}")

    return corpus_embeddings, corpus_ids, query_embeddings, query_ids

if __name__ == "__main__":
    corpus, queries, qrels = load_dataset()
    corpus_embeddings, corpus_ids, query_embeddings, query_ids = compute_and_cache_embeddings(corpus, queries)

    print(f"\nCorpus: {len(corpus_ids)} docs, embedding dim: {corpus_embeddings.shape[1]}")
    print(f"Queries: {len(query_ids)}")
    print(f"Qrels:   {len(qrels)} queries with relevance judgments")