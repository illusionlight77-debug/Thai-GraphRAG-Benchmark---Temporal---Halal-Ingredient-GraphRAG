# Research references

Grounding for the design and the claims this project sets out to test.

## GraphRAG vs RAG / benchmarks
- Han et al., **RAG vs. GraphRAG: A Systematic Evaluation and Key Insights** — arXiv:2502.11371.
  Finding this project builds on: GraphRAG's advantage concentrates on **multi-hop** queries
  (e.g. +0.325 F1 on MuSiQue, p < 0.001) while vanilla RAG stays competitive on single-hop.
- **A Systematic Review of Key RAG Systems: Progress, Gaps, Future Directions** — arXiv:2507.18910.
- **Awesome-GraphRAG** (surveys, benchmarks, open-source) — github.com/DEEP-PolyU/Awesome-GraphRAG.
- **A Survey on Knowledge-Oriented Retrieval-Augmented Generation** — arXiv:2503.10677.

## Graph construction / efficient GraphRAG
- **NodeRAG: Structuring Graph-based RAG with Heterogeneous Nodes** — arXiv:2504.11544.
- **LightRAG** (lightweight graph + dual-level retrieval) — see Awesome-GraphRAG.
- **AutoGraph-R1: End-to-End RL for Knowledge Graph Construction** — arXiv:2510.15339.
- **You Don't Need Pre-built Graphs for RAG (adaptive reasoning structures)** — arXiv:2508.06105.

## Temporal (extension B)
- **RAG Meets Temporal Graphs: Time-Sensitive Modeling and Retrieval** (TG-RAG, ECT-QA) — arXiv:2510.13590.
- **ATOM: Adaptive & Optimized dynamic temporal KG construction using LLMs** — arXiv:2510.22590.
- **A Temporal Knowledge Graph Generation Dataset Distantly Supervised by LLMs** — Nature Sci. Data, s41597-025-05062-0.
- Temporal RE corpora: TimeBank, TempLAMA, TempReason.

## Thai NLP / models
- **OpenThaiGPT 1.5: A Thai-Centric Open Source LLM** — arXiv:2411.07238.
- **Mangosteen: An Open Thai Corpus for LM Pretraining** — arXiv:2507.14664.
- **PyThaiNLP** — arXiv:2312.04649. WangchanBERTa for Thai NER / relation extraction.

## KG + LLM (general)
- **KG-LLM-Papers** — github.com/zjukg/KG-LLM-Papers.
- **LLM-KG4QA** — github.com/machuangtao/LLM-KG4QA.

## Multi-hop QA benchmarks (English, for cross-checking the harness)
- HotpotQA · 2WikiMultiHopQA · MuSiQue · MultiHop-RAG.
