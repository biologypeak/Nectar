<div align="center">

# 🍯 Nectar

### Local RAG Engine for Scientific Literature — Built for Researchers, Designed for Health

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain.com)
[![Gradio](https://img.shields.io/badge/Gradio-UI-FF6B6B?style=flat-square&logo=gradio&logoColor=white)](https://gradio.app)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-6B4FBB?style=flat-square)](https://trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

**Interroga la letteratura scientifica in linguaggio naturale. 100% locale. Zero API key. Zero cloud.**

[Quickstart](#-quickstart) · [Architecture](#-architecture) · [Roadmap](#-roadmap) · [Health Use Case](#-nectar-in-healthcare)

---

</div>

## What is Nectar?

Nectar is a **fully local Retrieval-Augmented Generation (RAG) pipeline** that lets you upload one or more scientific PDFs and query them in natural language. No data leaves your machine. No subscription required.

Built on top of **Qwen2.5-0.5B-Instruct** (LLM) and **mixedbread-ai/mxbai-embed-large-v1** (embeddings), Nectar is optimized for domain-specific scientific literature — with a strong focus on **biomedical and health research**.

> Think of it as a private, offline ChatGPT that has actually read your papers.

---

## Core Stack

| Component | Technology |
|---|---|
| **LLM** | `Qwen/Qwen2.5-0.5B-Instruct` — lightweight, instruction-tuned |
| **Embeddings** | `mixedbread-ai/mxbai-embed-large-v1` — state-of-the-art retrieval |
| **Vector Store** | ChromaDB (in-memory, ephemeral) |
| **Document Loader** | LangChain `PyPDFLoader` |
| **Text Splitter** | `RecursiveCharacterTextSplitter` (1000 tokens, 100 overlap) |
| **Interface** | Gradio (localhost:7860) |
| **Compute** | CUDA (GPU) or CPU fallback |
| **Prompt Format** | Qwen native ChatML (`<|im_start|>` / `<|im_end|>`) |

---

## Quickstart

### 1. Clone & setup environment

```bash
git clone https://github.com/biologypeak/Nectar.git
cd Nectar
conda env create -f environment.yml
conda activate nectar
```

### 2. Run

```bash
python nectar.py
```

### 3. Open the UI

Navigate to `http://localhost:7860` (or use the public Gradio share link printed in terminal).

Upload a PDF → ask a question → get a grounded answer.

---

## How It Works

```
PDF Input
    │
    ▼
PyPDFLoader → RecursiveCharacterTextSplitter (1000t / 100 overlap)
    │
    ▼
mxbai-embed-large-v1 → ChromaDB (in-memory vector store)
    │
    ▼
User Query → top-k=3 relevant chunks retrieved
    │
    ▼
ChatML Prompt (context + question) → Qwen2.5-0.5B-Instruct
    │
    ▼
Grounded Answer (streamed to Gradio UI)
```

The LLM is explicitly instructed to answer **only from the provided context** — no hallucination from pre-training knowledge.

---

## Roadmap

The following features are actively planned. Contributions and feedback welcome.

### Multi-Paper Support

- Load and index multiple PDFs simultaneously into a unified vector store
- Per-paper metadata tagging (title, authors, DOI, year) stored as ChromaDB document metadata
- Cross-paper retrieval: queries span the entire corpus, not a single document
- Source attribution in answers: every chunk cites its paper of origin

### Persistent Vector Database

- Replace in-memory ChromaDB with a **persistent ChromaDB instance** backed by local disk (`./chroma_db/`)
- Papers are ingested once and survive across sessions — no re-embedding on restart
- Incremental updates: add new papers without rebuilding the entire index
- Collection management: create, delete, and inspect named collections from the UI

### Retrieval Depth Control

- A **slider in the left panel** to control `k` — the number of relevant chunks retrieved per query (range: 1–20)
- Retrieved chunks are **displayed in the left panel** alongside their source paper, page number, and similarity score
- Users can inspect exactly what context the LLM sees before it generates an answer
- Manual chunk deselection: remove a specific chunk from context before submitting

### PyArrow / Lance VDB Migration (Evaluation Track)

ChromaDB abstracts away low-level vector operations. To unlock finer control and visualization capabilities, Nectar will evaluate a migration to **PyArrow + Lance** as the underlying vector database:

- Direct columnar access to embedding vectors (float32 arrays via PyArrow)
- Lance format: versioned, memory-mapped, zero-copy reads — significantly faster for large corpora
- Full metadata schema definition (typed columns: `paper_id`, `chunk_id`, `page`, `embedding`, `text`)
- Foundation for custom ANN indexing (IVF, HNSW) without opaque wrappers
- Native compatibility with Pandas, Polars, and DuckDB for ad-hoc analysis

### Vector Space Visualization

- **Interactive 3D scatterplot** of the entire embedding space (UMAP / t-SNE dimensionality reduction)
- Real-time query visualization: when you submit a query, its embedding is projected into the space and rendered live alongside the corpus
- Retrieved chunks highlighted with distance rings
- Cluster coloring by paper origin, topic, or date
- Zoom, rotate, filter — built with Plotly or Three.js

### Paper Mesh & Auto-Annotation

- **Mesh layer per paper**: semantic structure automatically extracted on ingest (Abstract → Methods → Results → Discussion → Conclusions)
- Section-aware retrieval: optionally restrict queries to specific sections (e.g., "search only in Methods")
- Auto-tagging pipeline: LLM-generated keywords, MeSH terms, and entity extraction (genes, diseases, compounds) stored per chunk
- Visual paper graph: nodes = papers, edges = semantic similarity — auto-computed on corpus updates

---

## 🩺 Nectar in Healthcare

Nectar is purpose-built for the pace and volume of biomedical literature.

**Future implementation:**

- **Systematic review acceleration** — query hundreds of papers for PICO elements (Population, Intervention, Comparison, Outcome) without reading each one
- **Microbiome & gut health research** — cross-reference findings across cohort studies, RCTs, and meta-analyses on the microbiota-gut-brain axis
- **Clinical trial monitoring** — ingest trial PDFs and extract adverse events, endpoints, and inclusion criteria by natural language query
- **Drug-gene-disease mapping** — ask relational questions across pharmacology papers ("Which studies link *Lactobacillus rhamnosus* to anxiety reduction?")
- **Medical education** — residents and students can query curated paper sets for evidence-based answers with source traceability

The persistent vector store and mesh annotation system make Nectar especially powerful for **longitudinal research workflows** — your knowledge base grows with your reading list.

> Nectar does not replace clinical judgment. Answers must always be verified against primary sources.

---

## Project Structure

```
Nectar/
├── nectar.py           # Main application — RAG pipeline + Gradio UI
├── environment.yml     # Conda environment (Python 3.10+, CUDA-aware)
└── README.md
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace LLM |
| `EMBED_MODEL_ID` | `mixedbread-ai/mxbai-embed-large-v1` | Embedding model |
| `DEVICE` | auto (CUDA / CPU) | Compute backend |
| `chunk_size` | 1000 | Token window per chunk |
| `chunk_overlap` | 100 | Overlap between adjacent chunks |
| `k` (retriever) | 3 | Number of chunks retrieved per query |
| `max_new_tokens` | 512 | Maximum LLM output length |
| `temperature` | 0.5 | LLM sampling temperature |

---

## Contributing

Nectar is in active development. If you want to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/persistent-chroma`)
3. Open a Pull Request with a clear description of the change

Issues, ideas, and paper-domain-specific feedback are welcome.

---

## License

MIT — use freely, credit appreciated.

---

<div align="center">

Built by [@biologypeak](https://github.com/biologypeak) · Powered by open-source models · No data leaves your machine

</div>