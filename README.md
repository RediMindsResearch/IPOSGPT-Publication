# IPOSGPT - Trustworthy AI for the Ocean: Bridging the Science-Policy Divide

[![DOI](https://zenodo.org/badge/1001043644.svg)](https://doi.org/10.5281/zenodo.16988264)

This repository provides the code, data, and example workflows for **IPOSGPT**, a large language model (LLM) system designed to answer ocean science questions in a **trustworthy, transparent, and policy-relevant** manner.  

<!-- The project supports our manuscript submission to *Nature Sustainability* and is openly shared to enable exploration, reproducibility, and community feedback.   -->

---

## 📂 Repository Structure  

### `code/` – Core IPOSGPT components  
The main pipeline can be followed in this order:  

1. **`hierarchical_retrieval.py`** – Multi-stage retrieval from the custom ocean science database.  
2. **`response_generation.py`** – Response generation and source tracking pipeline.  
3. **`post_generation_processing.py`** – Post-response pipeline for **source traceability and verification**.  
4. **`conversational_memory.py`** – Maintains context and memory across user queries.  
5. **`news_api.py`** – Retrieves recent ocean-related news articles for enrichment (optional).  

### `data/` – Evaluation & case studies  
- **Evaluation queries** – Benchmark questions used in the assessment of IPOSGPT.  
- **Hyperparameter optimization** – Results and configuration files used for tuning retrieval and model parameters.  
- **Seychelles case study** – Materials supporting the case study demonstration.  

---

## 📊 Evaluation Materials  

- [List of Evaluation Queries](./eval-queries.md)  
- [Evaluation Summary – RAG strategies](./eval1.md)  
- [Evaluation Summary – LLM models](./eval2.md)  

---

## 🚀 Getting Started  

> **Note**: This repository provides scripts and data for **reproducibility, exploration, and community feedback**, not a production-ready system.  
