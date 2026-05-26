import os
import json
import ast
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from datasets import load_dataset


#Config

EMBEDDING_DIR   = "embeddings"
RESULTS_DIR     = "results"
K_VALUES        = [10, 20, 50, 100]
EMBEDDING_TYPES = ["title", "abstract", "combined"]
SAMPLE_SIZE     = None   # None = all questions; set e.g. 200 for quick test
BATCH_SIZE      = 32

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

rel_counts = [len(q["relevant_pmids"]) for q in questions]
print(f"  relevant docs per question — "
      f"min={min(rel_counts)}  max={max(rel_counts)}  "
      f"mean={np.mean(rel_counts):.1f}  median={np.median(rel_counts):.1f}")


#Load BioBERT

print("\nLoading BioBERT ...")
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model     = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to(device)
model.eval()
print(f"  device: {device}")


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return L2-normalised CLS embeddings, shape (N, hidden)."""
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
    """Return 1-based positions (ranks) of all relevant docs in the full ranking."""
    return [i + 1 for i, pmid in enumerate(ranked_pmids) if pmid in relevant_set]


def precision_at_recall_level(positions, total_relevant, recall_level):
    """
    Interpolated precision at a given recall level.
    positions   : sorted 1-based ranks of relevant docs in the full ranking
    recall_level: float in (0, 1]
    """
    if not positions:
        return 0.0
    positions    = sorted(positions)
    target_hits  = int(np.ceil(total_relevant * recall_level))
    if target_hits > len(positions):
        return 0.0
    k = positions[target_hits - 1]
    return target_hits / k


#Main evaluation loop

for emb_type in EMBEDDING_TYPES:
    print(f"═══ Evaluating: {emb_type} embeddings ═══")
    doc_embs = np.load(os.path.join(EMBEDDING_DIR, f"{emb_type}_embeddings.npy"))

    print("  Embedding questions ...")
    q_texts = [q["question"] for q in questions]
    q_embs  = embed_texts(q_texts)

    print("  Computing similarities ...")

    sim_matrix = q_embs @ doc_embs.T  
    metrics_per_k = {k: {"precision": [], "recall": []} for k in K_VALUES}
    all_rankings  = []

    print("  Evaluating metrics ...")
    for qi, q in enumerate(questions):
        scores_q     = sim_matrix[qi]
        ranked_i     = np.argsort(scores_q)[::-1]         
        ranked_pmids = [str(documents[i]["pmid"]) for i in ranked_i]
        relevant     = set(q["relevant_pmids"])

        positions = get_relevant_positions(ranked_pmids, relevant)

        for k in K_VALUES:
            metrics_per_k[k]["precision"].append(precision_at_k(ranked_pmids, relevant, k))
            metrics_per_k[k]["recall"].append(recall_at_k(ranked_pmids, relevant, k))

        all_rankings.append({
            "question":           q["question"],
            "relevant_pmids":     q["relevant_pmids"],
            "relevant_positions": positions     
        })

    #Print results
    print(f"\n  Results for [{emb_type}]:")
    print(f"  {'K':<8} {'Precision@K':>12} {'Recall@K':>12}")
    print(f"  {'-'*34}")

    summary = {}
    for k in K_VALUES:
        p = float(np.mean(metrics_per_k[k]["precision"]))
        r = float(np.mean(metrics_per_k[k]["recall"]))
        summary[k] = {"precision": p, "recall": r}
        print(f"  {k:<8} {p:>12.4f} {r:>12.4f}")

    print()

    #Save results
    metrics_path  = os.path.join(RESULTS_DIR, f"metrics_{emb_type}.json")
    rankings_path = os.path.join(RESULTS_DIR, f"rankings_{emb_type}.json")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"embedding_type": emb_type,
                   "n_questions":    len(questions),
                   "K_values":       K_VALUES,
                   "metrics":        {str(k): v for k, v in summary.items()}},
                  f, indent=4)

    with open(rankings_path, "w", encoding="utf-8") as f:
        json.dump(all_rankings, f, ensure_ascii=False, indent=2)

    print(f"  Saved: {metrics_path}")
    print(f"  Saved: {rankings_path}\n")

print(" Brute-force evaluation complete!")
