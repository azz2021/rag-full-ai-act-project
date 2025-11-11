import json, math, numpy as np, pandas as pd, faiss
from sentence_transformers import SentenceTransformer

META_PATH   = "meta.csv"
CHUNKS_PATH = "chunks.jsonl"
OUT_META    = "meta_small.csv"
OUT_CHUNKS  = "chunks_small.jsonl"
OUT_FAISS   = "index_small.faiss"

SEED       = "EU AI Act obligations for high-risk systems"   # <<< change if you want Transformer topic
KEEP_FRAC  = 0.20
BATCH_SIZE = 256

meta = pd.read_csv(META_PATH)  # expects: doc,chunk_id,page,preview
assert {"doc","chunk_id","page","preview"}.issubset(meta.columns), "meta.csv columns missing."
n = len(meta)
print("Total chunks:", n)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
qv = model.encode([SEED], convert_to_numpy=True, normalize_embeddings=True)[0]

scores = np.empty(n, dtype=np.float32)
for start in range(0, n, BATCH_SIZE):
    end = min(n, start + BATCH_SIZE)
    texts = meta.loc[start:end-1, "preview"].astype(str).tolist()
    X = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    scores[start:end] = X @ qv

k = max(1, int(math.floor(n * KEEP_FRAC)))
top_idx = np.argpartition(-scores, k-1)[:k]
top_idx = top_idx[np.argsort(-scores[top_idx])]
subset  = meta.iloc[top_idx].copy().reset_index(drop=True)
print(f"Kept {len(subset)} chunks (~{KEEP_FRAC*100:.0f}%) for seed: {SEED}")

subset.to_csv(OUT_META, index=False)

keep_ids = set(subset["chunk_id"].astype(str).tolist())
kept = 0
with open(CHUNKS_PATH, "r", encoding="utf-8") as fin, open(OUT_CHUNKS, "w", encoding="utf-8") as fout:
    for line in fin:
        rec = json.loads(line)
        if str(rec.get("chunk_id","")) in keep_ids:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
print("Wrote", kept, "records to", OUT_CHUNKS)

X_small = model.encode(subset["preview"].astype(str).tolist(),
                       convert_to_numpy=True, normalize_embeddings=True).astype("float32")
d = X_small.shape[1]
index = faiss.IndexFlatIP(d)
index.add(X_small)
faiss.write_index(index, OUT_FAISS)
print("Saved:", OUT_META, OUT_CHUNKS, OUT_FAISS, "vectors:", len(subset))

print("\nLabel distribution:")
print(subset["doc"].value_counts())
