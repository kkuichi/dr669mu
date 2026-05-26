import json
import os
import numpy as np
import matplotlib.pyplot as plt

PLOTS_DIR = "plots"
K_VALUES  = [10, 20, 50, 100]
EMB_TYPE  = "combined"
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_metrics(path):
    with open(path, "r") as f:
        return json.load(f)


# BioBERT
biobert_bf   = load_metrics(f"results/metrics_{EMB_TYPE}.json")
biobert_faiss= load_metrics(f"results/metrics_{EMB_TYPE}_faiss.json")

# S-PubMedBert
pubmed_bf    = load_metrics(f"results2/metrics_{EMB_TYPE}.json")
pubmed_faiss = load_metrics(f"results2/metrics_{EMB_TYPE}_faiss.json")


def get_values(metrics, key):
    return [metrics["metrics"][str(k)][key] for k in K_VALUES]


x     = np.arange(len(K_VALUES))
width = 0.2

for metric_key, ylabel, title_str, filename in [
    ("precision", "Precision@K", "Precision@K — Model Comparison", "comparison_precision"),
    ("recall",    "Recall@K",    "Recall@K — Model Comparison",    "comparison_recall"),
]:
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - 1.5*width, get_values(biobert_bf,    metric_key), width, label="BioBERT — Brute Force",        color="#1f77b4")
    ax.bar(x - 0.5*width, get_values(biobert_faiss, metric_key), width, label="BioBERT — FAISS",              color="#1f77b4", alpha=0.5, hatch="//")
    ax.bar(x + 0.5*width, get_values(pubmed_bf,     metric_key), width, label="S-PubMedBert — Brute Force",   color="#ff7f0e")
    ax.bar(x + 1.5*width, get_values(pubmed_faiss,  metric_key), width, label="S-PubMedBert — FAISS",         color="#ff7f0e", alpha=0.5, hatch="//")

    ax.set_xlabel("K", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title_str + f" ({EMB_TYPE} embeddings)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, f"{filename}_{EMB_TYPE}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

print("\n Comparison plots saved!")
