import json
import os
import numpy as np
import matplotlib.pyplot as plt

#Config
RESULTS_DIR = "results"
PLOTS_DIR   = "plots"
EMB_TYPE    = "combined"   # change to "title" or "abstract" if needed
K_VALUES    = [10, 20, 50, 100]
RECALL_LEVELS = np.linspace(0.1, 1.0, 10)

os.makedirs(PLOTS_DIR, exist_ok=True)


#Load rankings
def load_rankings(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


brute_rankings = load_rankings(os.path.join(RESULTS_DIR, f"rankings_{EMB_TYPE}.json"))
faiss_rankings = load_rankings(os.path.join(RESULTS_DIR, f"rankings_{EMB_TYPE}_faiss.json"))

brute_metrics  = load_metrics(os.path.join(RESULTS_DIR, f"metrics_{EMB_TYPE}.json"))
faiss_metrics  = load_metrics(os.path.join(RESULTS_DIR, f"metrics_{EMB_TYPE}_faiss.json"))


#Precision-Recall curve helper
def precision_at_recall_level(positions, total_relevant, recall_level):
    """
    Interpolated precision at a given recall level.
    positions    : sorted 1-based ranks of relevant docs in the full ranking
    recall_level : float in (0, 1]
    """
    if not positions:
        return 0.0
    positions   = sorted(positions)
    target_hits = int(np.ceil(total_relevant * recall_level))
    if target_hits > len(positions):
        return 0.0
    k = positions[target_hits - 1]
    return target_hits / k


def compute_pr_curve(rankings):
    """Compute mean precision at each recall level for a set of rankings."""
    avg_precisions = []
    for r in RECALL_LEVELS:
        vals = []
        for item in rankings:
            positions  = item["relevant_positions"]
            total_rel  = len(item["relevant_pmids"])
            vals.append(precision_at_recall_level(positions, total_rel, r))
        avg_precisions.append(np.mean(vals))
    return RECALL_LEVELS, np.array(avg_precisions)


#Extract Precision@K and Recall@K from saved metrics

def extract_pk_rk(metrics_dict):
    precisions = [metrics_dict["metrics"][str(k)]["precision"] for k in K_VALUES]
    recalls    = [metrics_dict["metrics"][str(k)]["recall"]    for k in K_VALUES]
    return precisions, recalls


brute_p, brute_r = extract_pk_rk(brute_metrics)
faiss_p, faiss_r = extract_pk_rk(faiss_metrics)


#Compute Precision-Recall curves

r_levels_b, p_curve_b = compute_pr_curve(brute_rankings)
r_levels_f, p_curve_f = compute_pr_curve(faiss_rankings)


#Plot 1 — Precision@K

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(K_VALUES, brute_p, marker="o", linewidth=2, label="Brute Force")
ax.plot(K_VALUES, faiss_p, marker="s", linewidth=2, linestyle="--", label="FAISS")
ax.set_xlabel("K", fontsize=12)
ax.set_ylabel("Precision@K", fontsize=12)
ax.set_title(f"Precision@K — {EMB_TYPE} embeddings (BioBERT)", fontsize=13)
ax.set_xticks(K_VALUES)
ax.set_ylim(bottom=0)
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
path1 = os.path.join(PLOTS_DIR, f"precision_at_k_{EMB_TYPE}.png")
plt.savefig(path1, dpi=150)
plt.close()
print(f"Saved: {path1}")


#Plot 2 — Recall@K

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(K_VALUES, brute_r, marker="o", linewidth=2, label="Brute Force")
ax.plot(K_VALUES, faiss_r, marker="s", linewidth=2, linestyle="--", label="FAISS")
ax.set_xlabel("K", fontsize=12)
ax.set_ylabel("Recall@K", fontsize=12)
ax.set_title(f"Recall@K — {EMB_TYPE} embeddings (BioBERT)", fontsize=13)
ax.set_xticks(K_VALUES)
ax.set_ylim(bottom=0)
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
path2 = os.path.join(PLOTS_DIR, f"recall_at_k_{EMB_TYPE}.png")
plt.savefig(path2, dpi=150)
plt.close()
print(f"Saved: {path2}")


#Plot 3 — Precision-Recall Curve

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(r_levels_b * 100, p_curve_b, marker="o", linewidth=2, label="Brute Force")
ax.plot(r_levels_f * 100, p_curve_f, marker="s", linewidth=2, linestyle="--", label="FAISS")
ax.set_xlabel("Recall (%)", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title(f"Precision–Recall Curve — {EMB_TYPE} embeddings (BioBERT)", fontsize=13)
ax.set_xticks(RECALL_LEVELS * 100)
ax.set_ylim(bottom=0)
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
path3 = os.path.join(PLOTS_DIR, f"precision_recall_curve_{EMB_TYPE}.png")
plt.savefig(path3, dpi=150)
plt.close()
print(f"Saved: {path3}")

print("\n All plots saved to:", PLOTS_DIR)



#S-PubMedBert-MS-MARCO plots (results2/)

RESULTS_DIR2 = "results2"
MODEL2_LABEL = "S-PubMedBert"

brute_rankings2 = load_rankings(os.path.join(RESULTS_DIR2, f"rankings_{EMB_TYPE}.json"))
faiss_rankings2 = load_rankings(os.path.join(RESULTS_DIR2, f"rankings_{EMB_TYPE}_faiss.json"))
brute_metrics2  = load_metrics(os.path.join(RESULTS_DIR2, f"metrics_{EMB_TYPE}.json"))
faiss_metrics2  = load_metrics(os.path.join(RESULTS_DIR2, f"metrics_{EMB_TYPE}_faiss.json"))

brute_p2, brute_r2   = extract_pk_rk(brute_metrics2)
faiss_p2, faiss_r2   = extract_pk_rk(faiss_metrics2)
r_levels_b2, p_curve_b2 = compute_pr_curve(brute_rankings2)
r_levels_f2, p_curve_f2 = compute_pr_curve(faiss_rankings2)


#Plot 4 — Precision@K (S-PubMedBert)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(K_VALUES, brute_p2, marker="o", linewidth=2, label="Brute Force")
ax.plot(K_VALUES, faiss_p2, marker="s", linewidth=2, linestyle="--", label="FAISS")
ax.set_xlabel("K", fontsize=12)
ax.set_ylabel("Precision@K", fontsize=12)
ax.set_title(f"Precision@K — {EMB_TYPE} embeddings ({MODEL2_LABEL})", fontsize=13)
ax.set_xticks(K_VALUES)
ax.set_ylim(bottom=0)
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
path4 = os.path.join(PLOTS_DIR, f"precision_at_k_{EMB_TYPE}_{MODEL2_LABEL}.png")
plt.savefig(path4, dpi=150)
plt.close()
print(f"Saved: {path4}")


#Plot 5 — Recall@K (S-PubMedBert)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(K_VALUES, brute_r2, marker="o", linewidth=2, label="Brute Force")
ax.plot(K_VALUES, faiss_r2, marker="s", linewidth=2, linestyle="--", label="FAISS")
ax.set_xlabel("K", fontsize=12)
ax.set_ylabel("Recall@K", fontsize=12)
ax.set_title(f"Recall@K — {EMB_TYPE} embeddings ({MODEL2_LABEL})", fontsize=13)
ax.set_xticks(K_VALUES)
ax.set_ylim(bottom=0)
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
path5 = os.path.join(PLOTS_DIR, f"recall_at_k_{EMB_TYPE}_{MODEL2_LABEL}.png")
plt.savefig(path5, dpi=150)
plt.close()
print(f"Saved: {path5}")


#Plot 6 — Precision-Recall Curve (S-PubMedBert)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(r_levels_b2 * 100, p_curve_b2, marker="o", linewidth=2, label="Brute Force")
ax.plot(r_levels_f2 * 100, p_curve_f2, marker="s", linewidth=2, linestyle="--", label="FAISS")
ax.set_xlabel("Recall (%)", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title(f"Precision–Recall Curve — {EMB_TYPE} embeddings ({MODEL2_LABEL})", fontsize=13)
ax.set_xticks(RECALL_LEVELS * 100)
ax.set_ylim(bottom=0)
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
path6 = os.path.join(PLOTS_DIR, f"precision_recall_curve_{EMB_TYPE}_{MODEL2_LABEL}.png")
plt.savefig(path6, dpi=150)
plt.close()
print(f"Saved: {path6}")

print("\n All S-PubMedBert plots saved to:", PLOTS_DIR)