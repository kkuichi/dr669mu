import os
import json
import ast
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

#Config

EMBEDDING_DIR   = "embeddings2"
RESULTS_DIR     = "results2"
K_VALUES        = [10, 20, 50, 100]
EMBEDDING_TYPES = ["title", "abstract", "combined"]
SAMPLE_SIZE     = None
MODEL_NAME      = "pritamdeka/S-PubMedBert-MS-MARCO"

os.makedirs(RESULTS_DIR, exist_ok=True)

#Load documents

print("Loading pubmed_dataset.json ...")
with open("pubmed_dataset.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

pmid_to_idx = {str(doc["pmid"]): i for i, doc in enumerate(documents)}
print(f"  {len(documents)} documents loaded")

#Load questions

print("Loading RAG-mini-BioASQ questions ...")
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

print(f"  {len(questions)} questions ready")

#Load model

print(f"\nLoading {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)

#Metrics helpers

def precision_at_k(ranked_pmids, relevant, k):
    return sum(1 for p in ranked_pmids[:k] if p in relevant) / k

def recall_at_k(ranked_pmids, relevant, k):
    hits = sum(1 for p in ranked_pmids[:k] if p in relevant)
    return hits / len(relevant) if relevant else 0.0

def get_relevant_positions(ranked_pmids, relevant_set):
    return [i + 1 for i, pmid in enumerate(ranked_pmids) if pmid in relevant_set]

#Main evaluation loop

for emb_type in EMBEDDING_TYPES:
    print(f"\n═══ Evaluating: {emb_type} ═══")
    doc_embs = np.load(os.path.join(EMBEDDING_DIR, f"{emb_type}_embeddings.npy"))

    # Embed questions
    print("  Embedding questions ...")
    q_embs = model.encode(
        [q["question"] for q in questions],
        batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )

    # Similarities
    print("  Computing similarities ...")
    sim_matrix = q_embs @ doc_embs.T

    metrics_per_k = {k: {"precision": [], "recall": []} for k in K_VALUES}
    all_rankings  = []

    for qi, q in enumerate(questions):
        ranked_i     = np.argsort(sim_matrix[qi])[::-1]
        ranked_pmids = [str(documents[i]["pmid"]) for i in ranked_i]
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

    # Print
    print(f"\n  Results for [{emb_type}]:")
    print(f"  {'K':<8} {'Precision@K':>12} {'Recall@K':>12}")
    print(f"  {'-'*34}")

    summary = {}
    for k in K_VALUES:
        p = float(np.mean(metrics_per_k[k]["precision"]))
        r = float(np.mean(metrics_per_k[k]["recall"]))
        summary[k] = {"precision": p, "recall": r}
        print(f"  {k:<8} {p:>12.4f} {r:>12.4f}")

    # Save
    with open(os.path.join(RESULTS_DIR, f"metrics_{emb_type}.json"), "w") as f:
        json.dump({
            "embedding_type": emb_type,
            "model":          MODEL_NAME,
            "n_questions":    len(questions),
            "K_values":       K_VALUES,
            "metrics":        {str(k): v for k, v in summary.items()}
        }, f, indent=4)

    with open(os.path.join(RESULTS_DIR, f"rankings_{emb_type}.json"), "w", encoding="utf-8") as f:
        json.dump(all_rankings, f, ensure_ascii=False, indent=2)

    print(f"  Saved to {RESULTS_DIR}/")

print("\n Evaluation complete!")
