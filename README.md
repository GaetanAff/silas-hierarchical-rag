# Silas V2 - Hierarchical RAG Agent 🧠

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange)
![Ollama](https://img.shields.io/badge/Backend-Ollama-black)
![License](https://img.shields.io/badge/License-MIT-green)

**Silas V2** is a document analysis agent utilizing a **Hierarchical RAG** (Retrieval-Augmented Generation) architecture. Unlike classic RAG approaches that rely on vector search (embeddings), Silas performs surgical text analysis by cutting, scanning, and selecting relevant "chunks".

Designed to run **100% locally** with [Ollama](https://ollama.com/).

## 🚀 Why "Hierarchical"?

Standard RAG often retrieves entire documents or imprecise fragments. Silas mimics the human reading process through successive filtering stages:

1.  **Chunking**: Intelligent text segmentation based on natural boundaries.
2.  **Scanning**: Rapid summary of every segment (Fast Model).
3.  **Selection**: Selection of relevant segments based on summaries (Reasoning Model).
4.  **Extraction**: Deep reading of *only* the selected segments.
5.  **Synthesis**: Drafting the final answer with citations.

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com/) installed and running.

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR_USERNAME/silas-hierarchical-rag.git](https://github.com/YOUR_USERNAME/silas-hierarchical-rag.git)
cd silas-hierarchical-rag
