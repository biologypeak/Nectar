<div align="center">

# 🍯 Nectar

### Local RAG Engine for Scientific Literature — Built for Researchers, Designed for Health

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain.com)
[![Gradio](https://img.shields.io/badge/Gradio-UI-FF6B6B?style=flat-square&logo=gradio&logoColor=white)](https://gradio.app)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Persistent-6B4FBB?style=flat-square)](https://trychroma.com)
[![Version](https://img.shields.io/badge/version-2.0-F59E0B?style=flat-square)](#)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

**Interroga la letteratura scientifica in linguaggio naturale. 100% locale. Zero API key. Zero cloud.**

[What's New in v2](#-whats-new-in-v2) · [Quickstart](#-quickstart) · [How It Works](#-how-it-works) · [Roadmap](#-roadmap) · [Health Use Case](#-nectar-in-healthcare)

---

</div>

## What is Nectar?

Nectar is a **fully local Retrieval-Augmented Generation (RAG) pipeline** that lets you upload a corpus of scientific PDFs and query them in natural language. No data leaves your machine. No subscription required.

Built on top of **Qwen2.5-0.5B-Instruct** (LLM) and **mixedbread-ai/mxbai-embed-large-v1** (embeddings), Nectar is optimized for domain-specific scientific literature — with a strong focus on **biomedical and health research**.

> Think of it as a private, offline research assistant that has actually read your entire paper library.

---

## ✨ What's New in v2

Version 2 ships four major features, all implemented and production-ready.

### [1] Multi-PDF Ingestion

Upload any number of PDFs in a single batch. Each paper is processed independently and merged into a unified, queryable corpus. Every chunk is tagged with its source paper name and page number at ingest time, so provenance is never lost.

- `gr.File(file_count="multiple")` — native multi-file upload in Gradio
- Each chunk carries `paper_id` (MD5 of filepath) and `paper_name` metadata
- Duplicate detection: a paper already present in the DB is skipped automatically — no re-embedding, no wasted compute
- A dedicated **"Show papers in DB"** button lists the full indexed corpus at any time

### [2] Persistent ChromaDB

The vector store now survives restarts. Embeddings are written to `./nectar_chroma_db/` on disk at ingest time and loaded back transparently on the next launch. You build your knowledge base once; Nectar remembers it.

- Backed by `Chroma(persist_directory="./nectar_chroma_db/")` with collection name `nectar_papers`
- Incremental: new papers are appended without rebuilding the existing index
- The **"Svuota DB"** button calls `delete_collection()` for a clean slate when needed
- LLM and embedding models are loaded as singletons — no redundant loading between queries

### [3] Retrieval Depth Slider (k)

A slider in the left panel controls `k` — the number of chunks retrieved from the vector store for each query. Range: 1 to 20, default 3.

- Wired directly to `similarity_search_with_relevance_scores(query, k=k)`
- Increase `k` for broad, exploratory queries across a large corpus; decrease for precision on a focused question
- The slider value is passed live to every query — no restart required

### [4] Chunk Explorer Panel

Every query now surfaces a full **Chunk Retrieval Panel** in the left column. For each retrieved chunk you see:

- 📄 **Source PDF name** — exact filename of the paper the chunk comes from
- **Page number** — the page within that PDF
- **Affinity score** — cosine similarity ∈ [0, 1] as both a percentage and an ASCII bar (`█░`) for instant visual comparison
- **Preview** — first 400 characters of the chunk text

This gives you complete transparency into what context the LLM actually sees before generating its answer.

---

## Core Stack

| Component | Technology |
|---|---|
| **LLM** | `Qwen/Qwen2.5-0.5B-Instruct` — lightweight, instruction-tuned |
| **Embeddings** | `mixedbread-ai/mxbai-embed-large-v1` — state-of-the-art retrieval |
| **Vector Store** | ChromaDB — **persistent**, disk-backed (`./nectar_chroma_db/`) |
| **Document Loader** | LangChain `PyPDFLoader` |
| **Text Splitter** | `RecursiveCharacterTextSplitter` (1000 chars, 100 overlap) |
| **Similarity Search** | `similarity_search_with_relevance_scores` (cosine, normalized) |
| **Interface** | Gradio Blocks (localhost:7860) |
| **Compute** | CUDA (GPU) or CPU fallback — auto-detected |
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

```
Upload PDFs → click "Indicizza PDF" → ask a question → read the answer + chunk panel
```

The DB persists between sessions. You only need to ingest each paper once.

---

## 🖥️ UI Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  📂 Upload PDFs (multi)     │  📚 Show papers in DB             │
│  ⚡ Indicizza PDF           │  🗑️  Svuota DB                    │
│  [ingestion status]         │  [DB status]                      │
├─────────────────────────────────────────────────────────────────┤
│  🔍 Chunk Retrieval Panel   │  💬 Query input                   │
│                             │  🚀 Chiedi                        │
│  📊 Slider k  [1 ——●—— 20] │  📝 Answer output                 │
│                             │                                   │
│  Chunk 1 — 📄 paper_A.pdf  │                                   │
│  Pagina: 4 | Affinità: 87% │                                   │
│  ████████████████░░░░       │                                   │
│  > "...testo del chunk..."  │                                   │
│                             │                                   │
│  Chunk 2 — 📄 paper_B.pdf  │                                   │
│  Pagina: 11 | Affinità: 73%│                                   │
│  ██████████████░░░░░░       │                                   │
│  > "...testo del chunk..."  │                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ How It Works

```
PDF Upload (1 or N files)
    │
    ▼
PyPDFLoader → tag metadata (paper_id, paper_name, page)
    │
    ▼
RecursiveCharacterTextSplitter (1000 chars / 100 overlap)
    │
    ▼
mxbai-embed-large-v1 → ChromaDB (persistent on disk)
    │
    │   [on restart: DB reloaded from disk — no re-embedding]
    │
    ▼
User Query + k (from slider)
    │
    ▼
similarity_search_with_relevance_scores → top-k (doc, score) pairs
    │
    ├──→ Chunk Panel: paper name + page + affinity bar + preview
    │
    ▼
ChatML Prompt (context + question) → Qwen2.5-0.5B-Instruct
    │
    ▼
Grounded Answer → Gradio UI
```

The LLM is explicitly instructed to answer **only from the provided context** — no hallucination from pre-training knowledge. Answers are always traceable back to a specific chunk and page in a specific paper.

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace LLM |
| `EMBED_MODEL_ID` | `mixedbread-ai/mxbai-embed-large-v1` | Embedding model |
| `CHROMA_DIR` | `./nectar_chroma_db` | Persistent DB path |
| `COLLECTION` | `nectar_papers` | ChromaDB collection name |
| `DEVICE` | auto (CUDA / CPU) | Compute backend |
| `chunk_size` | 1000 | Characters per chunk |
| `chunk_overlap` | 100 | Overlap between adjacent chunks |
| `k` (slider) | 3 | Retrieved chunks per query (range 1–20) |
| `max_new_tokens` | 512 | Maximum LLM output length |
| `temperature` | 0.5 | LLM sampling temperature |
| `repetition_penalty` | 1.1 | Penalizes token repetition |

---

## Project Structure

```
Nectar/
├── nectar.py              # Main application — RAG pipeline + Gradio Blocks UI
├── environment.yml        # Conda environment (Python 3.10+, CUDA-aware)
├── nectar_chroma_db/      # Persistent vector store (auto-created on first ingest)
│   └── nectar_papers/     # ChromaDB collection
└── README.md
```

> `nectar_chroma_db/` is created automatically on first run. Add it to `.gitignore` if you do not want to commit your embeddings.

---

## Roadmap

The following features are in active development for v3.

### PyArrow / Lance VDB Migration

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

Nectar is purpose-built for the pace and volume of biomedical literature. With multi-paper ingestion and a persistent knowledge base, a researcher can build a private, local corpus of hundreds of papers and interrogate it like a database.

**Typical use cases:**

- **Systematic review acceleration** — query your entire corpus for PICO elements (Population, Intervention, Comparison, Outcome) across hundreds of papers without reading each one
- **Microbiome & gut health research** — cross-reference findings from cohort studies, RCTs, and meta-analyses on the microbiota-gut-brain axis; the chunk panel shows exactly which paper and page supports each claim
- **Clinical trial monitoring** — ingest trial PDFs and extract adverse events, endpoints, and inclusion criteria by natural language query
- **Drug-gene-disease mapping** — ask relational questions across pharmacology papers ("Which studies link *Lactobacillus rhamnosus* to anxiety reduction?")
- **Medical education** — residents and students can query curated paper sets for evidence-based answers with full source traceability down to the page level

The persistent vector store means your knowledge base grows with your reading list and is always ready — no warmup, no re-indexing, no waiting.

> Nectar does not replace clinical judgment. All answers must be verified against the primary sources cited in the chunk panel.

---

## Contributing

Nectar is in active development. If you want to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/lance-migration`)
3. Open a Pull Request with a clear description of the change

Issues, ideas, and domain-specific feedback (especially from healthcare researchers) are welcome.

---

## License

MIT — use freely, credit appreciated.

---

<div align="center">

Built by [@biologypeak](https://github.com/biologypeak) · Powered by open-source models · No data leaves your machine

**v1** — single PDF · in-memory ChromaDB &nbsp;→&nbsp; **v2** — multi-PDF · persistent DB · k-slider · chunk explorer

</div>