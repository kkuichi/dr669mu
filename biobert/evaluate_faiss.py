import os
import json
import time
import ast
import numpy as np
import faiss
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch


#Config 

EMBEDDING_DIR    = "embeddings"
RESULTS_DIR      = "results"
K_VALUES         = [10, 20, 50, 100]
EMBEDDING_TYPES  = ["title", "abstract", "combined"]
SAMPLE_SIZE      = None
BATCH_SIZE       = 32      # BioBERT embedding batch
SEARCH_BATCH     = 64      # questions per FAISS search call — reduce if OOM

os.makedirs(RESULTS_DIR, exist_ok=True)


#Load documents

with open("pubmed_dataset.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

pmid_to_idx = {str(doc["pmid"]): i for i, doc in enumerate(documents)}
all_pmids   = [str(doc["pmid"]) for doc in documents]
N_DOCS      = len(documents)
print(f"Documents loaded: {N_DOCS}")


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

print(f"Questions ready: {len(questions)}")


#Load BioBERT

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model     = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to(device)
model.eval()
print(f"BioBERT loaded on: {device}")


def embed_texts(texts):
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch  = texts[i : i + BATCH_SIZE]
            inputs = tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            ).to(device)
            out  = model(**inputs)
            embs = out.last_hidden_state[:, 0, :]
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu().numpy())
    return np.vstack(all_embs)


#Metrics helpers

def precision_at_k(ranked_pmids, relevant, k):
    return sum(1 for p in ranked_pmids[:k] if p in relevant) / k


def recall_at_k(ranked_pmids, relevant, k):
    hits = sum(1 for p in ranked_pmids[:k] if p in relevant)
    return hits / len(relevant) if relevant else 0.0


def get_relevant_positions(ranked_pmids, relevant_set):
    """1-based positions of relevant docs in the full ranking."""
    return [i + 1 for i, pmid in enumerate(ranked_pmids) if pmid in relevant_set]


#Embed all questions once

print("\nEmbedding questions ...")
q_embs = embed_texts([q["question"] for q in questions]).astype("float32")
print(f"  q_embs shape: {q_embs.shape}")


#Main loop

print("\n" + "=" * 60)
print("  FAISS RETRIEVAL EVALUATION")
print("=" * 60)

all_summaries = []

for emb_type in EMBEDDING_TYPES:
    print(f"\n─── {emb_type} ───")
    doc_embs = np.load(
        os.path.join(EMBEDDING_DIR, f"{emb_type}_embeddings.npy")
    ).astype("float32")

    dim   = doc_embs.shape[1]
    nlist = 64        
    nprobe = 8      

    quantizer = faiss.IndexFlatIP(dim)
    index     = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(doc_embs)
    index.add(doc_embs)
    index.nprobe = nprobe
    print(f"  FAISS IVF index built  ({N_DOCS} vectors, dim={dim}, nlist={nlist}, nprobe={nprobe})")

    all_indices = np.empty((len(questions), N_DOCS), dtype=np.int64)

    t0 = time.perf_counter()
    for start in range(0, len(questions), SEARCH_BATCH):
        end   = min(start + SEARCH_BATCH, len(questions))
        batch = q_embs[start:end]
        _, idxs = index.search(batch, N_DOCS)
        all_indices[start:end] = idxs
        print(f"  Searched {end}/{len(questions)} questions ...", end="\r")
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) / len(questions) * 1000
    print(f"\n  Search done in {t1-t0:.1f} s  ({avg_ms:.2f} ms/question)")

    # Compute metrics
    metrics_per_k = {k: {"precision": [], "recall": []} for k in K_VALUES}
    all_rankings  = []

    for qi, q in enumerate(questions):
        idxs         = all_indices[qi]
        ranked_pmids = [all_pmids[i] for i in idxs if i >= 0]
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

    all_summaries.append({
        "emb_type": emb_type,
        "avg_ms":   avg_ms,
        "summary":  summary,
    })

    # Free the large index before next iteration
    del all_indices
    del index

    # Save
    with open(os.path.join(RESULTS_DIR, f"metrics_{emb_type}_faiss.json"), "w") as f:
        json.dump({
            "embedding_type": emb_type,
            "method":         "faiss_flatip",
            "n_questions":    len(questions),
            "K_values":       K_VALUES,
            "avg_search_ms":  avg_ms,
            "metrics":        {str(k): v for k, v in summary.items()},
        }, f, indent=4)

    with open(
        os.path.join(RESULTS_DIR, f"rankings_{emb_type}_faiss.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(all_rankings, f, ensure_ascii=False, indent=2)

    print(f"  Saved: metrics_{emb_type}_faiss.json  |  rankings_{emb_type}_faiss.json")


#Print comparison table

print("\n" + "=" * 60)
print(f"  {'Type':>12}  {'K':>6}  {'Precision@K':>12}  {'Recall@K':>10}  {'ms/q':>8}")
print(f"  {'-'*58}")

for entry in all_summaries:
    for k in K_VALUES:
        p = entry["summary"][k]["precision"]
        r = entry["summary"][k]["recall"]
        print(f"  {entry['emb_type']:>12}  {k:>6}  {p:>12.4f}  {r:>10.4f}  {entry['avg_ms']:>8.2f}")
    print()

best = max(all_summaries, key=lambda e: e["summary"][100]["recall"])
print("=" * 60)
print(f"  Best by Recall@100 : {best['emb_type']}")
print(f"  Recall@100         : {best['summary'][100]['recall']:.4f}")
print(f"  Precision@100      : {best['summary'][100]['precision']:.4f}")
print("=" * 60)

print("\n Done!")