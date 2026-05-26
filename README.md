# Generatívne metódy pre vyhľadávanie informácií v medicínskych textoch

Tento projekt implementuje sémantický vyhľadávací pipeline nad biomedicínskymi textami z databázy PubMed. Cieľom je porovnať výkonnosť dvoch predtrénovaných jazykových modelov — **BioBERT** a **S-PubMedBERT-MS-MARCO** — pri vyhľadávaní relevantných dokumentov na základe prirodzenojazykových otázok. Vyhodnotenie prebieha na benchmarku [RAG-mini-BioASQ](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq) s využitím dvoch metód vyhľadávania: Brute-force a aproximatívneho vyhľadávania pomocou knižnice FAISS.



## 📁 Štruktúra projektu

```
.
├── pubmed_dataset.json          # Stiahnuté záznamy z PubMed (názov + abstrakt)
├── main.py                      # Krok 1 – sťahovanie záznamov z PubMed
│
├── embeddings/                  # Vygenerované embeddingy – BioBERT
│   ├── title_embeddings.npy
│   ├── abstract_embeddings.npy
│   └── combined_embeddings.npy
├── embeddings2/                 # Vygenerované embeddingy – S-PubMedBERT
│   ├── title_embeddings.npy
│   ├── abstract_embeddings.npy
│   └── combined_embeddings.npy
│
├── results/                     # Výsledky vyhodnotenia – BioBERT
├── results2/                    # Výsledky vyhodnotenia – S-PubMedBERT
├── plots/                       # Vizualizácie a grafy
│
├── biobert/                     # Pipeline modelu BioBERT
│   ├── emb.py                   # Výpočet embeddingov
│   ├── evaluate.py              # Vyhodnotenie – Brute-force
│   └── evaluate_faiss.py        # Vyhodnotenie – FAISS
│
└── pritamdeka/                  # Pipeline modelu S-PubMedBERT-MS-MARCO
    ├── emb2.py                  # Výpočet embeddingov
    ├── evaluate2.py             # Vyhodnotenie – Brute-force
    └── evaluate2_faiss.py       # Vyhodnotenie – FAISS
```



## 🗄️ Dataset

**[RAG-mini-BioASQ](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq)** — vyhľadávací benchmark odvodený z BioASQ, dostupný na HuggingFace:
`rag-datasets/rag-mini-bioasq` (split: `question-answer-passages`)

Každý riadok obsahuje biomedicínsku otázku a zoznam identifikátorov relevantných PubMed článkov (PMID). Záznamy (názov + abstrakt) boli stiahnuté z PubMed cez Entrez API a uložené do súboru `pubmed_dataset.json`. Do vyhodnotenia sa zahŕňajú iba otázky, pre ktoré boli všetky odkazované PMID úspešne stiahnuté.



## 🤖 Modely embeddingov

Pre každý dokument sa vypočítajú tri varianty embeddingov:

| Variant    | Vstupný text         |
|------------|----------------------|
| `title`    | Iba názov článku     |
| `abstract` | Iba abstrakt článku  |
| `combined` | Názov + abstrakt     |

Keďže všetky porovnávané prístupy používali rovnaké otázky, rovnaký korpus dokumentov a rovnaké referenčné PMID, výsledky bolo možné hodnotiť v rovnakých experimentálnych podmienkach. 

### BioBERT (`dmis-lab/biobert-base-cased-v1.1`)

Predtrénovaný BERT model doladený na biomedicínskych textoch. CLS-token embeddingy sú extrahované manuálnou dávkovou slučkou cez Transformers `AutoModel`. 

Skripty: `biobert/emb.py` → výstup uložený do `embeddings/`

### S-PubMedBERT-MS-MARCO (`pritamdeka/S-PubMedBert-MS-MARCO`)

Sentence-BERT model predtrénovaný na PubMed textoch a doladený na MS-MARCO pre úlohy sémantického vyhľadávania. Embeddingy sú vypočítané cez `SentenceTransformer.encode()` so vstavaným dávkovaním a normalizáciou. 

Skripty: `pritamdeka/emb2.py` → výstup uložený do `embeddings2/`



## 🔍 Metódy vyhľadávania

Otázky sú embeddované rovnakým modelom ako dokumenty. Podobnosť sa vypočíta ako skalárny súčin normalizovaných vektorov a dokumenty sú zoradené zostupne podľa skóre.

### Brute-force

Porovnávanie vektoru otázky so všetkými dokumentmi, presné výsledky, výpočtová náročnosť pri veľkom množstve dát.

- `biobert/evaluate.py`
- `pritamdeka/evaluate2.py`

### FAISS (aproximatívne vyhľadávanie)

Efektívnejšie vyhľadávanie podobných vektorov, rýchlejšie pri veľkom korpuse, výsledky môžu mierne odlišovať podľa indexovania.

- `biobert/evaluate_faiss.py`
- `pritamdeka/evaluate2_faiss.py`



## 📊 Vyhodnotenie

Modely sú vyhodnotené pri K ∈ {10, 20, 50, 100} pomocou nasledujúcich metrík:

- **Presnosť@K (Precision@K)** — podiel relevantných dokumentov spomedzi top-K výsledkov
- **Pokrytie@K (Recall@K)** — podiel nájdených relevantných dokumentov spomedzi top-K výsledkov

### Výstupné súbory

Každý vyhodnocovací skript produkuje dva súbory na každý typ embeddingu (napr. pre `combined`):

| Súbor                    | Obsah                                                                  |
|--------------------------|------------------------------------------------------------------------|
| `metrics_combined.json`  | Priemerná Presnosť@K a Pokrytie@K pre všetky hodnoty K                 |
| `rankings_combined.json` | Zoradený zoznam PMID na každú otázku a pozície relevantných dokumentov |

Výsledky sa uložia do `results/` (BioBERT) a `results2/` (S-PubMedBERT) v koreňi projektu.



## 📈 Vizualizácie

Grafy sú uložené v priečinku `plots/` a generované skriptami `plot_comparison.py` a `plot_precision_recall.py`.

| Súbor                                             | Obsah                                            |
|---------------------------------------------------|--------------------------------------------------|
| `comparison_precision_combined.png`               | Porovnanie Precision@K – BioBERT vs S-PubMedBERT |
| `comparison_recall_combined.png`                  | Porovnanie Recall@K – BioBERT vs S-PubMedBERT    |
| `precision_at_k_combined.png`                     | Precision@K – combined embeddingy BioBert        |
| `precision_at_k_combined_S-PubMedBert.png`        | Precision@K – combined embeddingy S-PubMedBert   |
| `recall_at_k_combined.png`                        | Recall@K – combined embeddingy BioBERT           |
| `precision_at_k_combined_S-PubMedBert.png`        | Recall@K – combined embeddingy S-PubMedBert      |
| `precision_recall_curve_combined.png`             | Precision-Recall krivka BioBERT                  |
| `precision_recall_curve_combined_S-PubMedBert.png`| Precision-Recall krivka S-PubMedBert             |



## 🛠️ Použité nástroje a knižnice

### Inštalácia

```bash
pip install datasets biopython sentence-transformers transformers torch faiss-cpu numpy matplotlib
```


### Prehľad knižníc

| Knižnica                  | Účel                                                                |
|---------------------------|---------------------------------------------------------------------|
| `datasets`                | Načítanie datasetu RAG-mini-BioASQ z HuggingFace                    |
| `biopython`               | Prístup k PubMed cez Entrez API (`Bio.Entrez`, `Bio.Medline`)       | 
| `transformers`            | Načítanie BioBERT tokenizéra a modelu (`AutoTokenizer`, `AutoModel`)|
| `torch`                   | Výpočet CLS-token embeddingov a L2-normalizácia                     |
| `sentence-transformers`   | Výpočet embeddingov pomocou S-PubMedBERT-MS-MARCO                   |
| `faiss-cpu` / `faiss-gpu` | Aproximatívne vyhľadávanie najbližších susedov (IVF-Flat index)     |
| `numpy`                   | Ukladanie embeddingov (`.npy`), maticové operácie, výpočet metrík   |
| `json`                    | Načítanie a ukladanie datasetu a výsledkov                          |
| `time`                    | Meranie doby behu a spomaľovanie požiadaviek na NCBI                |
| `os`                      | Práca so súborovým systémom                                         |
| `ast`                     | Parsovanie zoznamu PMID zo stĺpca datasetu                          |
| `matplotlib`              | Vizualizácia výsledkov – grafy Precision@K, Recall@K                |




## ⚡ Spustenie

```bash
# 1. Stiahnutie dokumentov z PubMed (spustiť raz)
python main.py

# 2. Výpočet embeddingov
python biobert/emb.py           # BioBERT      → embeddings/
python pritamdeka/emb2.py       # S-PubMedBERT → embeddings2/

# 3. Vyhodnotenie – BioBERT
python biobert/evaluate.py              # Brute-force
python biobert/evaluate_faiss.py        # FAISS

# 4. Vyhodnotenie – S-PubMedBERT
python pritamdeka/evaluate2.py          # Brute-force
python pritamdeka/evaluate2_faiss.py    # FAISS

#5.  Generovanie grafov
python plots/plot_comparison.py
python plots/plot_precision_recall.py
```

> **Poznámka:** Pred spustením `main.py` nastavte `Entrez.email` a `Entrez.api_key` na vlastné hodnoty. Požiadavky sú spomalené na 0,3 s na PMID, aby sa dodržali limity NCBI.
