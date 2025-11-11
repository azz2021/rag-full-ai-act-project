import os, gradio as gr, faiss, pandas as pd, numpy as np, torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

FAISS_PATH = "index.faiss"
META_PATH  = "meta.csv"
CHUNKS_PATH= "chunks.jsonl"

# ---- Load retriever ----
index = faiss.read_index(FAISS_PATH)
meta  = pd.read_csv(META_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

def embed(qs):
    return embedder.encode(qs, convert_to_numpy=True, normalize_embeddings=True, device=device)

def dense_search(query, k=5):
    q = embed([query]).astype("float32")
    D, I = index.search(q, k)
    out = meta.iloc[I[0]].copy()
    out["score"] = D[0]
    return out[["doc","page","score","preview","chunk_id"]].reset_index(drop=True)

# ---- Small open LLM (works on CPU) ----
model_name = "google/flan-t5-base"
tok  = AutoTokenizer.from_pretrained(model_name)
mdl  = AutoModelForSeq2SeqLM.from_pretrained(model_name)
gen  = pipeline("text2text-generation", model=mdl, tokenizer=tok,
                device=0 if torch.cuda.is_available() else -1)

def make_ctx(df, max_chars=2800):
    blocks, total = [], 0
    for i, r in df.iterrows():
        p = int(r["page"]) if pd.notna(r["page"]) else -1
        head = f"[C{i+1}] (doc={r['doc']}, page={p}, id={r['chunk_id']})"
        txt  = (r["preview"] or "") if pd.notna(r["preview"]) else ""
        block = f"{head}\n{txt}\n"
        if total + len(block) > max_chars: break
        blocks.append(block); total += len(block)
    return "\n".join(blocks)

def build_prompt(q, ctx):
    return f"""Use ONLY the CONTEXT to answer. If missing, reply:
"I don't know based on the provided context."

CONTEXT:
{ctx}

QUESTION:
{q}

ANSWER (cite chunks like [C1], [C2]):"""

def ask(query, k=5):
    df  = dense_search(query, k=k)
    ctx = make_ctx(df)
    out = gen(build_prompt(query, ctx), max_new_tokens=256, do_sample=False, temperature=0.2)[0]["generated_text"]
    lines = [f"Answer:\n{out}\n", "Context chunks:"]
    for i, r in df.iterrows():
        p = int(r["page"]) if pd.notna(r["page"]) else -1
        lines.append(f"- [C{i+1}] {r['doc']} p.{p} | {r['chunk_id']}")
    return "\n".join(lines)

with gr.Blocks(title="RAG Demo (Cloud Run)") as demo:
    gr.Markdown("### RAG Demo\nAsk a question; the answer is grounded in retrieved context.")
    q   = gr.Textbox(label="Question", placeholder="Explain scaled dot-product attention.")
    k   = gr.Slider(3, 10, value=5, step=1, label="Top-k")
    out = gr.Textbox(label="Answer + Context", lines=18)
    gr.Button("Ask").click(fn=ask, inputs=[q, k], outputs=out)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    demo.launch(server_name="0.0.0.0", server_port=port)
