import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel



save_dir = "embeddings"
os.makedirs(save_dir, exist_ok=True)


with open("pubmed_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} records")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1"
)

model = AutoModel.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1"
)

model.to(device)
model.eval()


def get_embeddings(texts, batch_size=16):
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            outputs = model(**inputs)

            cls_embeddings = outputs.last_hidden_state[:, 0, :]

            cls_embeddings = torch.nn.functional.normalize(
                cls_embeddings,
                p=2,
                dim=1
            )

            all_embeddings.append(
                cls_embeddings.cpu().numpy()
            )

    return np.vstack(all_embeddings)

texts_title = []
texts_abstract = []
texts_combined = []

for item in data:
    title = item.get("title", "")
    abstract = item.get("abstract", "")
    combined = title + " " + abstract

    texts_title.append(title)
    texts_abstract.append(abstract)
    texts_combined.append(combined)


title_embeddings = get_embeddings(texts_title)
abstract_embeddings = get_embeddings(texts_abstract)
combined_embeddings = get_embeddings(texts_combined)

print("Title shape:", title_embeddings.shape)
print("Abstract shape:", abstract_embeddings.shape)
print("Combined shape:", combined_embeddings.shape)


np.save(os.path.join(save_dir, "title_embeddings.npy"), title_embeddings)
np.save(os.path.join(save_dir, "abstract_embeddings.npy"), abstract_embeddings)
np.save(os.path.join(save_dir, "combined_embeddings.npy"), combined_embeddings)

print("Embeddings saved successfully!")