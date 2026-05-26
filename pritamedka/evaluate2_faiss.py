import os
import json
import time
import ast
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

#Config

EMBEDDING_DIR   = "embeddings2"
RESULTS_DIR     = "results2"
K_VALUES        = [10, 20, 50, 100]
EMBEDDING_TYPES = ["title", "abstract", "combined"]
SAMPLE_SIZE     = None
SEARCH_BATCH    = 64
MODEL_NAME      = "pritamdeka/S-PubMedBert-MS-MARCO"

os.makedirs(RESULTS_DIR, exist_ok=True)

#Load documents 

with open("pubmed_dataset.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

pmid_to_idx = {str(doc["pmid"]): i for i, doc in enumerate(documents)}
all_pmids   = [str(doc["pmid"]) for doc in documents]
N_DOCS      = len(documents)
print(f"Documents: {N_DOCS}")

#Load questions

ds = load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages")

questions = []
for split in ds:
    for row in ds[split]:
        q_text    = row["question"]
        rel_pmids = [str(p) for p in ast.literal_eval(row["relevant_passage_ids"])]
        rel_pmids = [p for p in rel_pmids if p in pmid_to_idx]
        if rel_pmids:
            questions.append({"question": q_text, "relevant_pmids": rel_pmids})

if SAMPLE_SIZE:
    questions = questions[:SAMPLE_SIZE]

print(f"Questions: {len(questions)}")

#Load model & embed questions

print(f"\nLoading {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)

print("Embedding questions ...")
q_embs = model.encode(
    [q["question"] for q in questions],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype("float32")

#Metrics helpers

def precision_at_k(ranked_pmids, relevant, k):
    return sum(1 for p in ranked_pmids[:k] if p in relevant) / k

def recall_at_k(ranked_pmids, relevant, k):
    hits = sum(1 for p in ranked_pmids[:k] if p in relevant)
    return hits / len(relevant) if relevant else 0.0

def get_relevant_positions(ranked_pmids, relevant_set):
    return [i + 1 for i, pmid in enumerate(ranked_pmids) if pmid in relevant_set]

#Main loop

print("\n" + "=" * 60)
print("  FAISS EVALUATION — S-PubMedBert-MS-MARCO")
print("=" * 60)

all_summaries = []

for emb_type in EMBEDDING_TYPES:
    print(f"\n─── {emb_type} ───")
    doc_embs = np.load(
        os.path.join(EMBEDDING_DIR, f"{emb_type}_embeddings.npy")
    ).astype("float32")

    # Approximate IVF index
    dim      = doc_embs.shape[1]
    nlist    = 64
    nprobe   = 8
    quantizer = faiss.IndexFlatIP(dim)
    index     = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(doc_embs)
    index.add(doc_embs)
    index.nprobe = nprobe
    print(f"  FAISS IVF index built (nlist={nlist}, nprobe={nprobe})")

    # Batched search
    all_indices = np.empty((len(questions), N_DOCS), dtype=np.int64)
    t0 = time.perf_counter()
    for start in range(0, len(questions), SEARCH_BATCH):
        end = min(start + SEARCH_BATCH, len(questions))
        _, idxs = index.search(q_embs[start:end], N_DOCS)
        all_indices[start:end] = idxs
        print(f"  {end}/{len(questions)} ...", end="\r")
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) / len(questions) * 1000
    print(f"\n  Done in {t1-t0:.1f} s  ({avg_ms:.2f} ms/question)")

    metrics_per_k = {k: {"precision": [], "recall": []} for k in K_VALUES}
    all_rankings  = []

    for qi, q in enumerate(questions):
        ranked_pmids = [all_pmids[i] for i in all_indices[qi] if i >= 0]
        relevant     = set(q["relevant_pmids"])
        positions    = get_relevant_positions(ranked_pmids, relevant)

        for k in K_VALUES:
            metrics_per_k[k]["precision"].append(precision_at_k(ranked_pmids, relevant, k))
            metrics_per_k[k]["recall"].append(recall_at_k(ranked_pmids, relevant, k))

        all_rankings.append({
            "question":           q["question"],
            "relevant_pmids":     q["relevant_pmids"],
            "relevant_positions": positions
        })

    summary = {}
    for k in K_VALUES:
        summary[k] = {
            "precision": float(np.mean(metrics_per_k[k]["precision"])),
            "recall":    float(np.mean(metrics_per_k[k]["recall"])),
        }
    all_summaries.append({"emb_type": emb_type, "avg_ms": avg_ms, "summary": summary})

    del all_indices, index

    with open(os.path.join(RESULTS_DIR, f"metrics_{emb_type}_faiss.json"), "w") as f:
        json.dump({
            "embedding_type": emb_type,
            "model":          MODEL_NAME,
            "method":         "faiss_ivf",
            "n_questions":    len(questions),
            "K_values":       K_VALUES,
            "avg_search_ms":  avg_ms,
            "metrics":        {str(k): v for k, v in summary.items()},
        }, f, indent=4)

    with open(
        os.path.join(RESULTS_DIR, f"rankings_{emb_type}_faiss.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(all_rankings, f, ensure_ascii=False, indent=2)

    print(f"  Saved to {RESULTS_DIR}/")

print("\n Done!")
