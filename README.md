# rag-full-ai-act-project
This project builds a scalable Retrieval-Augmented Generation (RAG) system that integrates text embedding, semantic retrieval, and answer generation, deployed on Google Cloud.










--------------------------------------------
Task 6. Deployment

The goal of this stage was to deploy the full RAG system on Google Cloud Run for real-time inference. The initial deployment using the complete dataset encountered a major constraint: the container exceeded Cloud Run’s free-tier memory and time limits, resulting in repeated “deadline exceeded” and RAM exhaustion errors during the build and startup phases. This issue arose because loading the full FAISS index and sentence-transformer model simultaneously required more memory and compute than available within the standard (2 GiB / 15 min) limits.

To address this, an alternative lightweight solution was implemented. A preprocessing step was added to select approximately 20% of the most relevant and representative data, ensuring that the subset covered both documents (EU AI Act + Transformer paper) proportionally. This reduced memory load while preserving semantic diversity. The reduced dataset was used to build a smaller FAISS index and deploy a simplified FastAPI-based retrieval service, allowing successful deployment on Cloud Run without timeouts.

This approach demonstrated a practical trade-off between system completeness and deployability—maintaining meaningful retrieval performance while fitting within free-tier resource constraints.
