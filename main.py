from datasets import load_dataset
from Bio import Entrez, Medline
import json
import time
import ast

Entrez.email   = "email"
Entrez.tool    = "bioasq_dataset_builder"
Entrez.api_key = "api_key"

ds = load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages")

# Collect unique PMIDs across ALL splits
all_pmids = set()
for split in ds:
    for row in ds[split]:
        pmids = ast.literal_eval(row["relevant_passage_ids"])
        for pmid in pmids:
            all_pmids.add(str(pmid))

print(f"Unique PMIDs to fetch: {len(all_pmids)}")


def fetch_pubmed_record(pmid):
    try:
        handle  = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
        records = list(Medline.parse(handle))
        handle.close()
        if not records:
            return None
        record = records[0]
        return {
            "pmid":     pmid,
            "title":    record.get("TI", ""),
            "abstract": record.get("AB", "")
        }
    except Exception as e:
        print(f"  Error with PMID {pmid}: {e}")
        return None


# Fetch each unique PMID once
pubmed_data = {}
for i, pmid in enumerate(sorted(all_pmids)):
    print(f"[{i+1}/{len(all_pmids)}] PMID {pmid}")
    record = fetch_pubmed_record(pmid)
    if record:
        pubmed_data[pmid] = record
    time.sleep(0.3)

# Build final list — one entry per unique PMID
final_dataset = list(pubmed_data.values())

print(f"\nTotal unique documents saved: {len(final_dataset)}")

with open("pubmed_dataset.json", "w", encoding="utf-8") as f:
    json.dump(final_dataset, f, ensure_ascii=False, indent=4)

print("Saved: pubmed_dataset.json")