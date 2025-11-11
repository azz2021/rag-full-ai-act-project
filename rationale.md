# Rationale for Query Selection

- **Breadth across domains:** 5 queries from the EU AI Act (policy/regulation) and 5 from “Attention Is All You Need” (technical ML) to test retrieval on heterogeneous language and structure.
- **Varied answer types:** definitions (q05, q07), lists (q01, q02), mechanisms/how (q08–q09), timelines (q04), and numeric facts (q10).
- **Retrieval difficulty mix:** short exact answers (q10) vs. multi-sentence explanations (q03, q08–q09) to exercise chunking and top-k retrieval.
- **Verifiability:** each query targets a specific, checkable passage in the selected documents, reducing ambiguity and hallucination risk.
- **Consistency:** this exact file (`eval/queries.jsonl`) will be used unchanged for all experiments and reports in later tasks.
