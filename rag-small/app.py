import os, numpy as np, pandas as pd, faiss
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

FAISS_PATH = "index_small.faiss"
META_PATH  = "meta_small.csv"

meta = pd.read_csv(META_PATH)  # expects: doc,chunk_id,page,preview
index = faiss.read_index(FAISS_PATH)

_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder

def search(query: str, k: int = 5):
    qv = get_embedder().encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, k)
    out = meta.iloc[I[0]].copy()
    out["score"] = D[0]
    return out.reset_index(drop=True)

def build_answer(query: str, rows: pd.DataFrame):
    # Simple grounded extractive answer with citations (no heavy LLM)
    ctx = " ".join(str(t) for t in rows["preview"].tolist())
    if not ctx.strip():
        return {"answer":"I don't know based on the provided context.", "citations":[]}
    # pick 1–2 sentences containing most query keywords
    parts = [s.strip() for s in ctx.replace(";", ".").split(".") if s.strip()]
    qwords = set(query.lower().split())
    scored = sorted(((len(qwords & set(p.lower().split())), p) for p in parts), reverse=True)
    chosen = " ".join([s for _, s in scored[:2]]) if scored else "I don't know based on the provided context."
    cits = [f"[C{i+1}] {r.doc} | {r.chunk_id} | score={rows.iloc[i]['score']:.3f}" for i, r in enumerate(rows.itertuples())]
    return {"answer": chosen, "citations": cits}

class Query(BaseModel):
    query: str
    k: int = 5

app = FastAPI(title="RAG Small (FAISS + FastAPI)")

@app.get("/")
def root():
    return {"ok": True, "message": "POST /ask with {query, k}"}

@app.post("/ask")
def ask(q: Query):
    rows = search(q.query, q.k)
    out  = build_answer(q.query, rows)
    out["context"] = [f"[C{i+1}] {rows.iloc[i]['preview']}" for i in range(len(rows))]
    return out

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
