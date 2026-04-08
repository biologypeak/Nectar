<div align="center">

# 🍯 Nectar

### Local RAG Platform for Scientific Literature — Built for Researchers, Designed for Health

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain.com)
[![LanceDB](https://img.shields.io/badge/LanceDB-columnar%20VDB-4F86C6?style=flat-square)](https://lancedb.com)
[![PyArrow](https://img.shields.io/badge/PyArrow-typed%20schema-E25A1C?style=flat-square)](https://arrow.apache.org/docs/python)
[![Version](https://img.shields.io/badge/version-4.0-F59E0B?style=flat-square)](#)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

**Query scientific literature in natural language. 100% local. Zero API keys. Zero cloud.**

[What's New in v4](#-whats-new-in-v4) · [Quickstart](#-quickstart) · [Architecture](#️-architecture) · [Pages](#-pages) · [API Reference](#-backend-api-reference) · [Roadmap](#-roadmap) · [Healthcare](#-nectar-in-healthcare)

---

</div>

## What is Nectar?

Nectar is a **fully local Retrieval-Augmented Generation (RAG) platform** for scientific literature. Upload a corpus of PDF papers, embed them into a persistent columnar vector store, and interrogate your entire library in natural language — with full traceability from answer back to source chunk and page.

Built on **Qwen2.5-0.5B-Instruct** (LLM), **mixedbread-ai/mxbai-embed-large-v1** (embeddings), **LanceDB** (vector store), and **Streamlit** (UI), Nectar is designed as a professional research tool with a clean dark interface suited for daily use.

> No data leaves your machine. No subscription. No warmup between sessions.

---

## ✨ What's New in v4

Version 4 is a complete UI rewrite. The Gradio single-page interface is replaced by a **Streamlit multipage application** with a persistent dark-themed sidebar, three dedicated pages, and a fully decoupled backend module.

### Interface migration: Gradio → Streamlit

The entire frontend is now built with Streamlit Blocks. The application launches via `streamlit run app.py` and exposes three pages through Streamlit's native multipage routing (`pages/` directory convention). A persistent sidebar shows live database status (paper count, chunk count, index state) on every page.

### Decoupled backend architecture

All RAG and database logic is extracted into `core/backend.py`, a standalone Python module with no Streamlit dependency. Every function returns plain Python dicts or lists — making the backend independently testable and reusable outside the UI. The UI layer only calls backend functions and renders their output.

### Three dedicated pages

Each page has a single responsibility: **Knowledge Base** for corpus management, **Query** for RAG retrieval and answering, **Explorer** for direct vector store inspection. Pages share state exclusively through LanceDB on disk — no Streamlit session state is needed for cross-page data persistence.

---

## Core Stack

| Component | Technology | Version |
|---|---|---|
| **UI Framework** | Streamlit multipage app | ≥ 1.32 |
| **LLM** | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace Hub |
| **Embeddings** | `mixedbread-ai/mxbai-embed-large-v1` | 1024-dim, L2-normalized |
| **Vector Store** | LanceDB — Lance columnar format | ≥ 0.6 |
| **Schema** | PyArrow typed `pa.schema` | ≥ 14.0 |
| **ANN Index** | IVF_PQ — selectable metric | cosine / l2 / dot |
| **Document Loader** | LangChain `PyPDFLoader` | ≥ 0.3 |
| **Text Splitter** | `RecursiveCharacterTextSplitter` | 1000 chars · 100 overlap |
| **Prompt Format** | Qwen native ChatML | `<\|im_start\|>` / `<\|im_end\|>` |
| **Compute** | CUDA (GPU) or CPU | auto-detected via `torch.cuda.is_available()` |

---

## Quickstart

### 1. Clone the repository and switch to the Streamlit branch

```bash
git clone https://github.com/biologypeak/Nectar.git
cd Nectar
git checkout streamlit
```

### 2. Create the environment

```bash
pip install -r requirements.txt
```

### 3. Launch

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. The LanceDB store at `./nectar_lancedb/` is created automatically on first ingest and persists across all subsequent sessions.

---

## 🗂️ Project Structure

```
Nectar/  (branch: streamlit)
├── app.py                        # Entry point — home page + sidebar + global CSS
├── requirements.txt              # All Python dependencies
│
├── core/
│   ├── __init__.py
│   └── backend.py                # All RAG/DB logic — UI-agnostic
│
├── pages/
│   ├── 1_Knowledge_Base.py       # Upload · Ingest · Index Builder · DB Stats
│   ├── 2_Query.py                # Natural language query · k-slider · Chunk Panel
│   └── 3_Explorer.py             # Filter · Browse · Cards/Table view
│
└── nectar_lancedb/               # LanceDB root — auto-created on first ingest
    └── nectar_papers.lance/      # Lance columnar table
```

---

## 📄 Pages

### `app.py` — Home & Navigation

The Streamlit entry point. Renders the home page with three feature cards and a persistent sidebar present on every page. The sidebar displays live database stats (papers, chunks, index status) by calling `get_db_stats()` from the backend on each render.

Navigation uses `st.page_link()` to route between the three pages:

```
📥  Knowledge Base   →  pages/1_Knowledge_Base.py
🔍  Query            →  pages/2_Query.py
🗂️  Explorer         →  pages/3_Explorer.py
```

---

### `pages/1_Knowledge_Base.py` — Knowledge Base

The corpus management page. Three independent sections on a single scrollable view.

**Section 1 — Upload & Ingest**

A multi-file uploader (`st.file_uploader(accept_multiple_files=True)`) accepts any number of PDFs in a single batch. On clicking "Embed & Add to Database", files are written to a temporary directory and passed to `ingest_multiple_pdfs()`. Results are displayed as a per-file status table with three badge states:

| Badge | Meaning |
|---|---|
| `INDEXED` (green) | File successfully chunked, embedded, and written to LanceDB |
| `SKIPPED` (yellow) | Paper already present in the DB (MD5 dedup) — no re-embedding |
| `ERROR` (red) | Extraction or embedding failed — error message displayed |

**Section 2 — IVF_PQ Index Builder**

A `st.selectbox` lets the user choose among three distance metrics. Selecting a metric immediately renders a detail card showing the metric's mathematical formula, description, value range, recommended use case, and a "recommended" badge for cosine. Clicking "Build Index" calls `build_index(metric_key)`. If the corpus has fewer than `INDEX_MIN_ROWS` (256) chunks, an error is shown with the current count and the minimum required — index creation is blocked. On success, a four-column KPI card shows the metric used, number of IVF partitions, PQ sub-vectors, and total indexed chunks.

**Section 3 — Database Statistics**

Live KPIs rendered as metric cards: total papers, total chunks, average chunks per paper, and ANN index status (Active / None). Below the KPIs, a `st.dataframe` shows a per-paper breakdown (paper name, chunk count, unique pages) sorted by chunk count descending. A "Refresh" button re-runs `get_db_stats()` and a "Clear Database" button calls `clear_database()` followed by `st.rerun()`.

---

### `pages/2_Query.py` — Query

The primary RAG interface. Layout is two columns: left (1/3) for retrieval controls and the Chunk Panel, right (2/3) for question input and the answer.

**Retrieval Settings**

A `st.slider` (range 1–20, default 5, step 1) controls `k` — the number of chunks retrieved per query. The current corpus size is displayed below the slider as contextual information.

**Question Input**

A `st.text_area` accepts the research question. Submission is triggered either by clicking "Run Query" or pressing Enter. The page calls `answer_query(query, k)` from the backend.

**Answer Panel**

The LLM response is rendered inside a custom `.answer-box` div with preserved whitespace. If `answer_query` returns an error, `st.error()` is displayed instead.

**Chunk Panel**

After every query, the left column renders one `.chunk-card` per retrieved chunk. Each card shows:

- Paper name and page number in the `.chunk-meta` header
- A gradient CSS affinity bar (width proportional to affinity score)
- Affinity percentage and raw distance value
- Up to 480 characters of chunk text as preview

Affinity is computed as `max(0, 1 − distance / 2)`, converting LanceDB's raw distance (lower = more similar) into an intuitive 0–100% score.

---

### `pages/3_Explorer.py` — Explorer

Direct vector store inspection without running a retrieval query. Useful for auditing the corpus, verifying ingestion quality, and locating specific passages.

**Corpus Overview**

Four metric cards at the top of the page: total papers, total chunks, average chunks per paper, ANN index status. Sourced from `get_db_stats()`.

**Filters**

Four filter controls in a single row:

| Control | Type | Function |
|---|---|---|
| Paper | `st.selectbox` | Filter to a specific paper by filename, or "All papers" |
| Page from | `st.number_input` | Minimum page number (inclusive) |
| Page to | `st.number_input` | Maximum page number (inclusive) |
| Max results | `st.selectbox` | Limit: 25 / 50 / 100 / 250 / 500 |

A free-text keyword input below the row filters chunks by substring match (case-insensitive) against the `chunk` column. All filters are applied together via `explore_database()`.

**Results**

A result count line shows how many chunks matched. A `st.radio` toggle switches between two display modes:

- **Cards** — chunks grouped by paper, each in a `.chunk-card` with page number. If a keyword filter is active, matches are highlighted inline with a `<mark>` styled span (green on dark background).
- **Table** — flat `st.dataframe` with columns Paper, Page, Chunk (truncated to 300 chars for readability).

---

## 🔧 Backend API Reference

All functions are in `core/backend.py`. The module is UI-agnostic — every function accepts and returns plain Python types.

### Configuration constants

| Constant | Value | Description |
|---|---|---|
| `LLM_MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace LLM identifier |
| `EMBED_MODEL_ID` | `mixedbread-ai/mxbai-embed-large-v1` | Embedding model identifier |
| `LANCE_DIR` | `./nectar_lancedb` | LanceDB root directory |
| `TABLE_NAME` | `nectar_papers` | LanceDB table name |
| `EMBED_DIM` | `1024` | Embedding output dimension |
| `INDEX_MIN_ROWS` | `256` | Minimum chunks required to build IVF_PQ index |

### PyArrow Schema

```python
SCHEMA = pa.schema([
    pa.field("paper_id",   pa.string()),           # MD5 of absolute filepath — dedup key
    pa.field("paper_name", pa.string()),            # Original PDF filename
    pa.field("page",       pa.int32()),             # Source page number (0-indexed)
    pa.field("chunk",      pa.string()),            # Raw text of the chunk
    pa.field("vector",     pa.list_(pa.float32(), 1024)),  # Embedding vector
])
```

### Singletons

**`get_llm() → HuggingFacePipeline`**
Returns the loaded Qwen2.5-0.5B-Instruct pipeline. Loaded once on first call and cached in `_llm_instance`. Uses `torch.float16` on CUDA, `float32` on CPU. Parameters: `max_new_tokens=512`, `temperature=0.5`, `repetition_penalty=1.1`.

**`get_embedding_model() → HuggingFaceEmbeddings`**
Returns the mxbai-embed-large-v1 model. Loaded once and cached in `_embedding_instance`. Embeddings are L2-normalized (`normalize_embeddings=True`) with a retrieval prefix prompt applied at encode time.

### Database helpers

**`get_db() → lancedb.LanceDBConnection`**
Opens a connection to the LanceDB store at `LANCE_DIR`.

**`get_table() → lancedb.Table | None`**
Returns the `nectar_papers` table if it exists, `None` otherwise.

**`get_or_create_table() → lancedb.Table`**
Returns the table, creating it with `SCHEMA` if it does not exist yet.

**`row_count() → int`**
Returns the total number of rows (chunks) in the table. Returns 0 if the table does not exist.

**`paper_id(filepath: str) → str`**
Returns the MD5 hex digest of the absolute path of a file. Used as the deduplication key.

**`already_indexed(filepath: str) → bool`**
Returns `True` if a row with the file's `paper_id` already exists in the table. Uses a LanceDB SQL prefilter: `WHERE paper_id = '{pid}'`.

**`embed_texts(texts: list[str]) → np.ndarray`**
Embeds a list of strings using the embedding model singleton. Returns a `float32` NumPy array of shape `(len(texts), 1024)`.

### Ingestion

**`ingest_pdf(filepath: str) → dict`**
Loads, splits, and embeds a single PDF. Builds a typed `pa.table` batch and appends it to the LanceDB table. Returns a result dict:

```python
{
    "name":    str,    # PDF filename
    "status":  str,    # "ok" | "skipped" | "empty" | "error"
    "chunks":  int,    # Number of chunks written (0 if skipped/error)
    "skipped": bool,   # True if already indexed
    "error":   str|None
}
```

**`ingest_multiple_pdfs(filepaths: list[str]) → list[dict]`**
Calls `ingest_pdf` for each path and returns the list of result dicts.

**`get_db_stats() → dict`**
Returns aggregate database statistics:

```python
{
    "total_chunks":  int,
    "total_papers":  int,
    "papers":        list[dict],   # per-paper: paper_name, chunks, pages
    "has_index":     bool
}
```
Index detection uses `tbl.index_stats("vector")`.

**`clear_database() → bool`**
Drops the `nectar_papers` table from LanceDB. Returns `True`.

### Index builder

**`build_index(metric_key: str) → dict`**
Creates an IVF_PQ ANN index on the `vector` column. `metric_key` is one of `"cosine"`, `"l2"`, `"dot"`. Auto-tunes parameters:

- `num_partitions = max(1, n // 4096)`
- `num_sub_vectors = EMBED_DIM // 8` → 128

Returns:

```python
{
    "ok":      bool,
    "message": str,
    "params":  dict   # metric, num_partitions, num_sub_vectors, indexed_chunks
}
```

Returns `ok=False` with a descriptive message if the table is empty or has fewer than `INDEX_MIN_ROWS` chunks. Uses `replace=True` so re-running safely overwrites the previous index.

### Retrieval

**`retrieve_chunks(query: str, k: int) → list[dict]`**
Embeds the query, performs a vector search via `tbl.search().limit(k).select([...]).to_pandas()`, and returns the top-k results:

```python
[{
    "paper_name": str,
    "page":       int,
    "chunk":      str,
    "affinity":   float,   # max(0, 1 − distance/2), range [0, 1]
    "distance":   float    # raw LanceDB _distance value
}]
```

**`answer_query(query: str, k: int) → dict`**
Full RAG pipeline: retrieves chunks, builds context string, runs the Qwen ChatML prompt chain, and returns:

```python
{
    "answer": str,
    "chunks": list[dict],   # same as retrieve_chunks output
    "error":  str|None
}
```

Returns `error` string (not raises) on any exception so the UI can display it gracefully.

### Explorer

**`explore_database(paper_filter, page_min, page_max, keyword, limit) → list[dict]`**
Loads the full table (columns: `paper_name`, `page`, `chunk`) into Pandas and applies four sequential filters: paper name substring, page range, keyword substring, row limit. Returns list of dicts with the same three fields.

**`get_paper_names() → list[str]`**
Returns a sorted list of unique paper filenames currently in the database. Used to populate the Explorer filter dropdown.

---

## 🗂️ IVF_PQ Index Builder — Metric Reference

| Metric | Formula | Range | Recommended for |
|---|---|---|---|
| **cosine** | `1 − (a·b / ‖a‖‖b‖)` | [0, 2] | Normalized text embeddings — **default for mxbai** |
| **l2** | `√Σ(aᵢ−bᵢ)²` | [0, ∞) | Non-normalized vectors, geometry-sensitive tasks |
| **dot** | `−(a·b)` | (−∞, ∞) | Max inner-product search, recommendation |

With `normalize_embeddings=True` (Nectar's default), cosine and dot produce equivalent rankings. Cosine is the safe choice for all text retrieval tasks.

### Row count guard

Index training requires at minimum `INDEX_MIN_ROWS = 256` chunks. Attempting to build an index on a smaller corpus returns:

```
Insufficient data for ANN index training.
Current chunks: N
Minimum required: 256
Add more papers and try again.
```

### Auto-tuned parameters

| Parameter | Formula | Example (n = 10,000) |
|---|---|---|
| `num_partitions` | `max(1, n // 4096)` | 2 |
| `num_sub_vectors` | `EMBED_DIM // 8` | 128 |

---

## ⚙️ How It Works

```
PDF Upload (1 or N files, via st.file_uploader)
    │
    ▼
Saved to tempfile.TemporaryDirectory()
    │
    ▼
ingest_pdf() per file:
    PyPDFLoader → RecursiveCharacterTextSplitter (1000 chars / 100 overlap)
    │
    ▼
embed_texts() → mxbai-embed-large-v1 → float32[1024] vectors
    │
    ▼
pa.table({paper_id, paper_name, page, chunk, vector})  ← typed PyArrow batch
    │
    ▼
tbl.add(batch)  →  ./nectar_lancedb/nectar_papers.lance  (persistent)
    │
    │   [on restart: db.open_table() — instant, no re-embedding]
    │
    ▼
(optional) build_index(metric)  →  IVF_PQ ANN index on vector column
    │
    ╔══════════════════════════════════╗
    ║  Query page                      ║
    ╠══════════════════════════════════╣
    ║  answer_query(query, k)          ║
    ║    └─ retrieve_chunks(query, k)  ║
    ║         └─ embed_query()         ║
    ║         └─ tbl.search().limit(k) ║
    ║              → top-k (doc, dist) ║
    ║    └─ affinity = 1 − dist/2      ║
    ║    └─ ChatML prompt + Qwen LLM   ║
    ║    └─ answer + chunks returned   ║
    ╚══════════════════════════════════╝
    │
    ▼
Streamlit renders: answer-box + chunk cards with affinity bars
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace model identifier |
| `EMBED_MODEL_ID` | `mixedbread-ai/mxbai-embed-large-v1` | Embedding model |
| `EMBED_DIM` | `1024` | Embedding output dimension |
| `LANCE_DIR` | `./nectar_lancedb` | LanceDB root directory |
| `TABLE_NAME` | `nectar_papers` | LanceDB table name |
| `INDEX_MIN_ROWS` | `256` | Minimum chunks to allow IVF_PQ index training |
| `DEVICE` | auto (CUDA / CPU) | Compute backend |
| `chunk_size` | `1000` | Characters per chunk |
| `chunk_overlap` | `100` | Overlap between adjacent chunks |
| `k` (slider default) | `5` | Retrieved chunks per query (UI range: 1–20) |
| `max_new_tokens` | `512` | Maximum LLM output length |
| `temperature` | `0.5` | LLM sampling temperature |
| `repetition_penalty` | `1.1` | Penalizes repeated tokens |

---

## Roadmap

The following features are planned for v5.

### Vector Space Visualization

Interactive 3D scatterplot of the entire embedding space using UMAP or t-SNE dimensionality reduction. When a query is submitted, its embedding is projected into the same space and rendered in real time alongside the corpus. Retrieved chunks are highlighted with distance rings. Color-coded by paper of origin. Built on top of direct PyArrow column reads from LanceDB — no intermediate export required.

### Paper Mesh & Auto-Annotation

Semantic structure extraction on ingest: each paper is automatically segmented into Abstract, Methods, Results, Discussion, and Conclusions sections. Section labels are stored as a new `section` column in the LanceDB schema, enabling section-aware retrieval. A parallel auto-tagging pipeline generates LLM-derived keywords, MeSH terms, and named entities (genes, diseases, compounds) stored as additional typed columns. A visual paper graph (nodes = papers, edges = cosine similarity between centroids) is auto-computed on each corpus update.

### PyArrow / DuckDB Analytics Layer

Direct SQL queries over the Lance table via DuckDB, without loading data into memory. Enables advanced corpus analytics: chunk length distributions, per-paper coverage heatmaps, duplicate detection, and vocabulary statistics — all rendered as Streamlit charts on a dedicated Analytics page.

---

## 🩺 Nectar in Healthcare

Nectar is purpose-built for the volume and heterogeneity of biomedical literature. The persistent LanceDB store scales from a handful of papers to hundreds without changing the workflow — the knowledge base grows with the reading list and is always ready on the next session.

**Typical research workflows:**

- **Systematic review acceleration** — query the entire corpus for PICO elements (Population, Intervention, Comparison, Outcome) across hundreds of studies; the Chunk Panel cites every source with page-level precision
- **Microbiome & gut-brain axis research** — cross-reference findings from cohort studies, RCTs, and meta-analyses; IVF_PQ cosine search surfaces the most semantically aligned passages across the full corpus
- **Clinical trial monitoring** — ingest trial PDFs and extract adverse events, endpoints, and inclusion criteria by natural language query; the Explorer page allows direct keyword search for regulatory terms
- **Drug-gene-disease mapping** — ask relational questions across pharmacology papers and inspect every supporting chunk directly
- **Medical education** — residents and students query curated paper sets for evidence-based answers with full source traceability down to the page

> Nectar does not replace clinical judgment. All answers must be verified against the primary sources shown in the Chunk Panel.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/vector-visualization`)
3. Open a Pull Request against the `streamlit` branch with a description of the change

Issues, feedback from healthcare researchers, and contributions to the visualization and annotation roadmap are especially welcome.

---

## License

MIT — use freely, credit appreciated.

---

<div align="center">

Built by [@biologypeak](https://github.com/biologypeak) · Powered by open-source models · No data leaves your machine

**v1** — single PDF · in-memory ChromaDB &nbsp;→&nbsp;
**v2** — multi-PDF · persistent ChromaDB · k-slider · chunk explorer &nbsp;→&nbsp;
**v3** — LanceDB · PyArrow schema · IVF_PQ index · metric selector &nbsp;→&nbsp;
**v4** — Streamlit multipage · decoupled backend · Knowledge Base · Query · Explorer

</div>

