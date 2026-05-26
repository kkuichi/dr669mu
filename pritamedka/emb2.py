import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer


SAVE_DIR   = "embeddings2"
MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"

os.makedirs(SAVE_DIR, exist_ok=True)

#Load documents

with open("pubmed_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} records")

#Load model

print(f"\nLoading {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded.")

#Prepare texts

texts_title    = [item.get("title", "")                                    for item in data]
texts_abstract = [item.get("abstract", "")                                 for item in data]
texts_combined = [item.get("title", "") + " " + item.get("abstract", "")  for item in data]

#Compute embeddings
#SentenceTransformer handles batching, padding, normalisation internally

print("\nEncoding titles ...")
title_embeddings = model.encode(
    texts_title, batch_size=64, show_progress_bar=True,
    normalize_embeddings=True
)

print("\nEncoding abstracts ...")
abstract_embeddings = model.encode(
    texts_abstract, batch_size=64, show_progress_bar=True,
    normalize_embeddings=True
)

print("\nEncoding combined ...")
combined_embeddings = model.encode(
    texts_combined, batch_size=64, show_progress_bar=True,
    normalize_embeddings=True
)

print(f"\nTitle shape    : {title_embeddings.shape}")
print(f"Abstract shape : {abstract_embeddings.shape}")
print(f"Combined shape : {combined_embeddings.shape}")

#Save

np.save(os.path.join(SAVE_DIR, "title_embeddings.npy"),    title_embeddings)
np.save(os.path.join(SAVE_DIR, "abstract_embeddings.npy"), abstract_embeddings)
np.save(os.path.join(SAVE_DIR, "combined_embeddings.npy"), combined_embeddings)

print("\nEmbeddings saved to:", SAVE_DIR)
