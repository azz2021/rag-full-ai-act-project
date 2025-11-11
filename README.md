# rag-full-ai-act-project
This project builds a scalable Retrieval-Augmented Generation (RAG) system that integrates text embedding, semantic retrieval, and answer generation, deployed on Google Cloud.

Task 1. Data Preparation
---------------------------------------


Two core documents were used — the EU AI Act summary (DOCX) and “Attention Is All You Need” (PDF) — representing both regulatory and technical text.

Cleaning & Normalisation: Text was extracted with python-docx and PyPDF2, then standardised using unidecode and Unicode NFC. Extra spaces, line breaks, and page numbers were removed, and hyphenated words were merged.

Chunking: Each text was split into ~600-token segments with 100-token overlap via tiktoken, balancing contextual completeness with retrieval precision.

Streaming Pipeline: A memory-safe, streaming process cleaned and chunked pages on-the-fly, writing results directly to chunks.jsonl to prevent Colab RAM issues.

Metadata: Each chunk includes doc, chunk_id, and page for traceable retrieval.

 Outcome: A clean, structured dataset of tokenised text chunks (chunks.jsonl) ready for embedding and FAISS indexing in the subsequent RAG pipeline.
 
Task 2. Test Queries
---------------------------------------


A balanced set of ten test queries was created — five from the EU AI Act and five from the Transformer paper — to evaluate the system’s ability to handle both regulatory and technical language.
The queries span definitions, mechanisms, timelines, lists, and factual details, ensuring diversity in linguistic style and retrieval complexity. This mix tests the RAG system’s robustness across short factual responses (e.g., hyperparameters) and multi-sentence conceptual explanations (e.g., attention mechanisms).
All queries were designed to have verifiable answers traceable to the original documents, preventing ambiguity and hallucination. The same query set (queries.jsonl) was used consistently throughout all retrieval, generation, and evaluation experiments for fair comparison.

Task 3. Retrieval Component
-----------------------------------------

Methods implemented

Dense: FAISS + MiniLM (cosine/IP) for semantic similarity.

Lexical: BM25 for exact-term/number matching.

Hybrid (RRF): Reciprocal Rank Fusion of Dense + BM25 (optionally with MMR for diversity).

How we show relevance

Helper funcs show(q, k) (dense) and show_hybrid(q, k) (hybrid) print Top-k results as
[DOC page] score=… preview…, demonstrating that the retrieved chunks directly answer the query.

Example queries

EU AI Act: “What are the AI risk categories defined by the EU AI Act?” → dense Top-5 from the Act.

Transformer: “Explain scaled dot-product attention.” → hybrid Top-5 from the paper.


Trade-offs:

Dense: + understands meaning/synonyms; − can miss exact tokens/numbers.

BM25: + great for exact terms/figures; − no semantic understanding.

Hybrid (RRF): + best overall precision/recall; − a bit slower (two searches) and needs fusion tuning.

Task 4. Generation Component
---------------------------------------------
In this stage, the system combines the retrieved document chunks with the user query to generate grounded answers using a Large Language Model (LLM). The query is embedded with the MiniLM model, and FAISS retrieves the top-k most relevant chunks (optionally fused with BM25 results). These chunks are inserted into a structured prompt containing the context, question, and an instruction for the model to respond only from the given information. The prompt is processed by an open-source LLM (e.g., FLAN-T5) to produce a concise, evidence-based answer. The output includes both the answer and the cited chunks, demonstrating that the response is context-driven and non-hallucinatory.



Task 5. Generation Component
---------------------------------------------

Evaluation. We evaluated three criteria on a fixed query set: (1) Retrieval relevance using Recall@k and MRR based on each query’s doc_hint; (2) Answer accuracy via keyword checks (expected_keywords) in the model’s answer; and (3) Groundedness / hallucination by verifying that any claimed keywords also appear in the retrieval context, or that the model correctly says “I don’t know based on the provided context” when unsupported. We report per-query results and overall averages (Recall@5, MRR, Accuracy, Groundedness).



Task 6. Deployment
---------------------------------------------

The goal of this stage was to deploy the full RAG system on Google Cloud Run for real-time inference. The initial deployment using the complete dataset encountered a major constraint: the container exceeded Cloud Run’s free-tier memory and time limits, resulting in repeated “deadline exceeded” and RAM exhaustion errors during the build and startup phases. This issue arose because loading the full FAISS index and sentence-transformer model simultaneously required more memory and compute than available within the standard (2 GiB / 15 min) limits.

To address this, an alternative lightweight solution was implemented. A preprocessing step was added to select approximately 20% of the most relevant and representative data, ensuring that the subset covered both documents (EU AI Act + Transformer paper) proportionally. This reduced memory load while preserving semantic diversity. The reduced dataset was used to build a smaller FAISS index and deploy a simplified FastAPI-based retrieval service, allowing successful deployment on Cloud Run without timeouts.

This approach demonstrated a practical trade-off between system completeness and deployability—maintaining meaningful retrieval performance while fitting within free-tier resource constraints.

