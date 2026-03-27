<div align="center">

# 🍯 Nectar

### Local RAG Engine for Scientific Literature — Built for Researchers, Designed for Health

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain.com)
[![Gradio](https://img.shields.io/badge/Gradio-UI-FF6B6B?style=flat-square&logo=gradio&logoColor=white)](https://gradio.app)
[![LanceDB](https://img.shields.io/badge/LanceDB-columnar%20VDB-4F86C6?style=flat-square)](https://lancedb.com)
[![PyArrow](https://img.shields.io/badge/PyArrow-schema-E25A1C?style=flat-square)](https://arrow.apache.org/docs/python)
[![Version](https://img.shields.io/badge/version-3.0-F59E0B?style=flat-square)](#)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

**Interroga la letteratura scientifica in linguaggio naturale. 100% locale. Zero API key. Zero cloud.**

[What's New](#-whats-new) · [Quickstart](#-quickstart) · [How It Works](#️-how-it-works) · [Index Builder](#️-ivf_pq-index-builder) · [Roadmap](#-roadmap) · [Health Use Case](#-nectar-in-healthcare)

---

</div>

## What is Nectar?

Nectar is a **fully local Retrieval-Augmented Generation (RAG) pipeline** that lets you upload a corpus of scientific PDFs and query them in natural language. No data leaves your machine. No subscription required.

Built on top of **Qwen2.5-0.5B-Instruct** (LLM), **mixedbread-ai/mxbai-embed-large-v1** (embeddings), and **LanceDB** (columnar vector store), Nectar is optimized for domain-specific scientific literature — with a strong focus on **biomedical and health research**.

> Think of it as a private, offline research assistant that has actually read your entire paper library — and remembers it across restarts.

---

## ✨ What's New

### v3 — LanceDB + PyArrow + IVF_PQ Index Builder

#### [A] LanceDB replaces ChromaDB

The vector store has been migrated from ChromaDB to **LanceDB**, a columnar database built on the open Lance file format. LanceDB is disk-persistent by design — no `persist_directory` configuration needed, embeddings are written atomically to `./nectar_lancedb/` on every ingest.

- `lancedb.connect(LANCE_DIR)` + `db.create_table()` / `db.open_table()` — zero boilerplate persistence
- Lance columnar format: memory-mapped, versioned, zero-copy reads — significantly faster for large corpora
- Native PyArrow interop: the entire table is queryable as a typed Arrow table at any time
- SQL-style filter predicates for duplicate detection: `where("paper_id = '...'", prefilter=True)`

#### [B] Explicit PyArrow Schema

Every row in the LanceDB table conforms to a strict typed schema defined with `pa.schema`:

```python
SCHEMA = pa.schema([
    pa.field("paper_id",   pa.string()),
    pa.field("paper_name", pa.string()),
    pa.field("page",       pa.int32()),
    pa.field("chunk",      pa.string()),
    pa.field("vector",     pa.list_(pa.float32(), 1024)),
])
```

Ingestion builds a fully typed `pa.table()` before writing — no implicit type coercion, no silent schema drift. The three primary columns exposed to the retriever are `paper_name`, `chunk`, and `vector`.

#### [C] IVF_PQ Index Builder with Metric Selector

A dedicated **Index Builder** section in the UI lets you create an Approximate Nearest Neighbor (ANN) index over the vector column. This replaces exhaustive scan with a two-stage IVF_PQ search, making retrieval significantly faster as the corpus grows.

- **Dropdown** with three distance metrics — selecting one dynamically updates an explanation panel
- **Auto-tuned parameters**: `num_partitions = max(1, n // 4096)`, `num_sub_vectors = 1024 // 8 = 128`
- **Row count guard**: index creation is blocked if the corpus has fewer than 256 chunks, returning a clear message with the current count and the minimum required
- `replace=True` — re-running the builder safely overwrites the previous index

---

### v2 — Multi-PDF · Persistent DB · k-Slider · Chunk Explorer ✅

All v2 features are fully preserved in v3.

**[1] Multi-PDF Ingestion** — upload any number of PDFs in a single batch; each chunk is tagged with `paper_id` (MD5), `paper_name`, and `page`; duplicate detection skips already-indexed papers automatically.

**[2] Persistent Vector Store** — embeddings survive restarts; new papers are appended incrementally; the **"Svuota DB"** button drops and recreates the table for a clean slate.

**[3] Retrieval Depth Slider (k)** — a slider (range 1–20, default 3) controls how many chunks are retrieved per query; wired live to every search call.

**[4] Chunk Explorer Panel** — after every query, the left panel shows each retrieved chunk with its source PDF name, page number, affinity score as a percentage, an ASCII affinity bar (`█░`), and a 400-character text preview.

---

## Core Stack

| Component | Technology |
|---|---|
| **LLM** | `Qwen/Qwen2.5-0.5B-Instruct` — lightweight, instruction-tuned |
| **Embeddings** | `mixedbread-ai/mxbai-embed-large-v1` — 1024-dim, normalized |
| **Vector Store** | **LanceDB** — columnar Lance format, disk-persistent (`./nectar_lancedb/`) |
| **Schema** | **PyArrow** — typed `pa.schema` with `paper_name`, `chunk`, `vector (float32[1024])` |
| **ANN Index** | **IVF_PQ** — selectable metric: `cosine`, `l2`, `dot` |
| **Document Loader** | LangChain `PyPDFLoader` |
| **Text Splitter** | `RecursiveCharacterTextSplitter` (1000 chars, 100 overlap) |
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
Upload PDFs → Indicizza PDF → (optional) Crea Indice IVF_PQ → ask a question
```

The LanceDB store persists between sessions. You only need to ingest each paper once.

---

## 🖥️ UI Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  ROW 1 — INGESTION                                                  │
│  📂 Upload PDFs (multi)       │  📚 Mostra paper nel DB             │
│  ⚡ Indicizza PDF             │  🗑️  Svuota DB                      │
│  [ingestion status + count]   │  [DB status]                        │
├─────────────────────────────────────────────────────────────────────┤
│  ROW 2 — IVF_PQ INDEX BUILDER                                       │
│  📊 Dropdown: cosine / l2 / dot          │  🔨 Crea Indice IVF_PQ  │
│  ℹ️  Metric explanation panel            │  [index status]          │
├─────────────────────────────────────────────────────────────────────┤
│  ROW 3 — QUERY                                                      │
│  🔍 Chunk Retrieval Panel     │  💬 Query input                     │
│                               │  🚀 Chiedi                          │
│  📊 Slider k  [1 ——●—— 20]   │  📝 Answer output                   │
│                               │                                     │
│  Chunk 1 — 📄 paper_A.pdf    │                                     │
│  Pagina: 4 | Affinità: 87%   │                                     │
│  ████████████████░░░░         │                                     │
│  > "...testo del chunk..."    │                                     │
│                               │                                     │
│  Chunk 2 — 📄 paper_B.pdf    │                                     │
│  Pagina: 11 | Affinità: 73%  │                                     │
│  ██████████████░░░░░░         │                                     │
│  > "...testo del chunk..."    │                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ How It Works

```
PDF Upload (1 or N files)
    │
    ▼
PyPDFLoader → RecursiveCharacterTextSplitter (1000 chars / 100 overlap)
    │
    ▼
Tag metadata per chunk: paper_id (MD5), paper_name, page
    │
    ▼
mxbai-embed-large-v1 → float32[1024] vectors
    │
    ▼
pa.table({ paper_id, paper_name, page, chunk, vector })  ← typed PyArrow batch
    │
    ▼
LanceDB tbl.add(batch) → written to ./nectar_lancedb/ (Lance columnar format)
    │
    │   [on restart: tbl = db.open_table() — instant, no re-embedding]
    │
    ▼
(optional) tbl.create_index(metric=...) → IVF_PQ ANN index
    │
    ▼
User Query + k (from slider)
    │
    ▼
embed_query → tbl.search(vector).limit(k) → top-k rows + _distance
    │
    ├──→ Chunk Panel: paper_name + page + affinity (1 − dist/2) + bar + preview
    │
    ▼
ChatML Prompt (context + question) → Qwen2.5-0.5B-Instruct
    │
    ▼
Grounded Answer → Gradio UI
```

The LLM is instructed to answer **only from the retrieved context** — no hallucination from pre-training knowledge. Every answer is traceable to a specific chunk, page, and PDF via the Chunk Explorer Panel.

---

## 🗂️ IVF_PQ Index Builder

The index builder applies an **Inverted File + Product Quantization (IVF_PQ)** index to the `vector` column. Without an index, LanceDB performs an exact exhaustive scan — correct but O(n). With the index, search becomes approximate (ANN) and scales to millions of vectors.

### Distance Metrics

| Metric | Formula | Best for |
|---|---|---|
| **cosine** | `1 − (a·b / ‖a‖‖b‖)` | Normalized text embeddings — **recommended for mxbai** |
| **l2** | `√Σ(aᵢ−bᵢ)²` | Non-normalized vectors; geometry-sensitive tasks |
| **dot** | `−(a·b)` | Max-inner-product search; favors high-magnitude vectors |

> With `normalize_embeddings=True` (Nectar's default), cosine and dot produce equivalent rankings. L2 is the safe default for unnormalized vectors.

### Row Count Guard

IVF_PQ training requires a minimum number of vectors to populate partitions meaningfully. Nectar blocks index creation if the corpus has fewer than **256 chunks**, returning:

```
⚠️ Numero di chunk insufficiente per creare un indice ANN.
   Chunk attuali: 47
   Minimo richiesto: 256
   Aggiungi altri paper e riprova.
```

### Auto-tuned Parameters

| Parameter | Formula | Example (n=8000) |
|---|---|---|
| `num_partitions` | `max(1, n // 4096)` | 1 |
| `num_sub_vectors` | `EMBED_DIM // 8` | 128 |

Parameters are recomputed at index creation time based on the current row count. Re-running the builder with `replace=True` safely overwrites the previous index.

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace LLM |
| `EMBED_MODEL_ID` | `mixedbread-ai/mxbai-embed-large-v1` | Embedding model |
| `EMBED_DIM` | `1024` | Embedding output dimension |
| `LANCE_DIR` | `./nectar_lancedb` | LanceDB root directory |
| `TABLE_NAME` | `nectar_papers` | LanceDB table name |
| `INDEX_MIN_ROWS` | `256` | Minimum chunks required to build IVF_PQ index |
| `DEVICE` | auto (CUDA / CPU) | Compute backend |
| `chunk_size` | `1000` | Characters per chunk |
| `chunk_overlap` | `100` | Overlap between adjacent chunks |
| `k` (slider) | `3` | Retrieved chunks per query (range 1–20) |
| `max_new_tokens` | `512` | Maximum LLM output length |
| `temperature` | `0.5` | LLM sampling temperature |
| `repetition_penalty` | `1.1` | Penalizes token repetition |

---

## Project Structure

```
Nectar/
├── nectar.py              # Main application — RAG pipeline + Gradio Blocks UI
├── environment.yml        # Conda environment (Python 3.10+, CUDA-aware)
├── nectar_lancedb/        # LanceDB root (auto-created on first ingest)
│   └── nectar_papers.lance/  # Lance columnar table
└── README.md
```

> `nectar_lancedb/` is created automatically on first run. Add it to `.gitignore` if you do not want to commit your embeddings to the repository.

---

## Roadmap

The following features are in active development for v4.

### Vector Space Visualization

- **Interactive 3D scatterplot** of the entire embedding space (UMAP / t-SNE dimensionality reduction to 3D)
- Real-time query visualization: the query embedding is projected into the space and rendered live alongside the corpus at search time
- Retrieved chunks highlighted with distance rings; cluster coloring by paper origin, topic, or date
- Built with Plotly or Three.js; powered by direct PyArrow columnar reads from LanceDB

### Paper Mesh & Auto-Annotation

- **Mesh layer per paper**: semantic structure automatically extracted on ingest (Abstract → Methods → Results → Discussion → Conclusions)
- Section-aware retrieval: optionally restrict queries to specific paper sections
- Auto-tagging pipeline: LLM-generated keywords, MeSH terms, and named entity extraction (genes, diseases, compounds) stored as LanceDB columns
- Visual paper graph: nodes = papers, edges = semantic similarity — auto-computed on corpus updates via DuckDB or Polars on the Lance table

---

## 🩺 Nectar in Healthcare

Nectar is purpose-built for the pace and volume of biomedical literature. With multi-paper ingestion, a persistent columnar knowledge base, and an ANN index tuned to the researcher's preferred distance metric, Nectar scales from a handful of papers to hundreds without changing the workflow.

**Typical use cases:**

- **Systematic review acceleration** — query your entire corpus for PICO elements (Population, Intervention, Comparison, Outcome) across hundreds of papers without reading each one; the chunk panel cites every source
- **Microbiome & gut health research** — cross-reference findings from cohort studies, RCTs, and meta-analyses on the microbiota-gut-brain axis; IVF_PQ cosine search surfaces the most semantically aligned passages across the full corpus
- **Clinical trial monitoring** — ingest trial PDFs and extract adverse events, endpoints, and inclusion criteria by natural language query
- **Drug-gene-disease mapping** — ask relational questions across pharmacology papers ("Which studies link *Lactobacillus rhamnosus* to anxiety reduction?")
- **Medical education** — residents and students query curated paper sets for evidence-based answers with full source traceability down to the page level

The persistent LanceDB store means your knowledge base grows with your reading list and is always ready — no warmup, no re-indexing, no re-embedding.

> Nectar does not replace clinical judgment. All answers must be verified against the primary sources cited in the Chunk Explorer Panel.

---

## Contributing

Nectar is in active development. If you want to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/vector-visualization`)
3. Open a Pull Request with a clear description of the change

Issues, ideas, and domain-specific feedback (especially from healthcare researchers) are welcome.

---

## License

MIT — use freely, credit appreciated.

---

<div align="center">

Built by [@biologypeak](https://github.com/biologypeak) · Powered by open-source models · No data leaves your machine

**v1** — single PDF · in-memory ChromaDB &nbsp;→&nbsp;
**v2** — multi-PDF · persistent ChromaDB · k-slider · chunk explorer &nbsp;→&nbsp;
**v3** — LanceDB · PyArrow schema · IVF_PQ index · metric selector

</div>