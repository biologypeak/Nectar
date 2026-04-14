<div align="center">

# 🍯 Nectar

### Local RAG Platform for Scientific Literature — Built for Researchers, Designed for Health

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain.com)
[![LanceDB](https://img.shields.io/badge/LanceDB-columnar%20VDB-4F86C6?style=flat-square)](https://lancedb.com)
[![PyArrow](https://img.shields.io/badge/PyArrow-typed%20schema-E25A1C?style=flat-square)](https://arrow.apache.org/docs/python)
[![Version](https://img.shields.io/badge/version-4.1-F59E0B?style=flat-square)](#)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

**Query scientific literature in natural language. 100% local. Zero API keys. Zero cloud.**

[What's New](#-whats-new) · [Quickstart](#-quickstart) · [Architecture](#️-architecture) · [Pages](#-pages) · [API Reference](#-backend-api-reference) · [Roadmap](#-roadmap) · [Healthcare](#-nectar-in-healthcare)

---

</div>

## What is Nectar?

Nectar is a **fully local Retrieval-Augmented Generation (RAG) platform** for scientific literature. Upload a corpus of PDF papers, embed them into a persistent columnar vector store, and interrogate your entire library in natural language — with full traceability from answer back to source chunk and page. A dedicated **Vector Space Explorer** lets you visualise, cluster, and navigate the entire embedding space interactively.

Built on **Qwen2.5-0.5B-Instruct** (LLM), **mixedbread-ai/mxbai-embed-xsmall-v1** (embeddings), **LanceDB** (vector store), and **Streamlit** (UI), Nectar is designed as a professional research tool with a clean dark interface suited for daily use.

> No data leaves your machine. No subscription. No warmup between sessions.

---

## ✨ What's New

### v4.1 — Vector Space Explorer

A new fourth page (**Vector Space**) adds a fully interactive 2D/3D visualisation of the entire embedding space:

- **Four dimensionality reduction algorithms** — PaCMAP, UMAP, TriMap, t-SNE — selectable at runtime with a 2D/3D toggle
- **Five clustering algorithms** — HDBSCAN, Leiden, K-Means, Spectral, Agglomerative — applied independently from projection so parameters can be tweaked without recomputing the layout
- **Persistent disk cache** — projections and fitted reducers are saved to `./nectar_cache/` and reload instantly across Streamlit restarts; cache auto-invalidates when the corpus changes
- **Real-time query projection** — type a sentence, click *Project →*, and a ★ marker appears in the space showing where the query embedding lands (supported by PaCMAP, UMAP, and t-SNE via their `transform()` API)
- **Sub-clustering** — click *Sub-cluster ★* to drill into the cluster where the query landed; the view is replaced with a fresh reduction + clustering of that subset only; *← Full corpus* restores the full view
- **Chunk preview panel** — click any point in the scatter to read its full chunk text in the right-hand panel; hover shows a 220-character preview directly in the Plotly tooltip without triggering Streamlit reruns
- **WebGL rendering** (`go.Scattergl`) for smooth interaction at 20 000+ points

### v4.0 — Streamlit multipage & decoupled backend

Complete rewrite from a Gradio single-page interface to a **Streamlit multipage application** with a persistent dark-themed sidebar. All RAG and database logic was extracted into `core/backend.py` — a standalone, UI-agnostic Python module.

---

## Core Stack

| Component | Technology | Version |
|---|---|---|
| **UI Framework** | Streamlit multipage app | ≥ 1.32 |
| **LLM** | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace Hub |
| **Embeddings** | `mixedbread-ai/mxbai-embed-xsmall-v1` | 384-dim, L2-normalized |
| **Vector Store** | LanceDB — Lance columnar format | ≥ 0.6 |
| **Schema** | PyArrow typed `pa.schema` | ≥ 14.0 |
| **ANN Index** | IVF_PQ — selectable metric | cosine / l2 / dot |
| **Document Loader** | LangChain `PyPDFLoader` | ≥ 0.3 |
| **Text Splitter** | `RecursiveCharacterTextSplitter` | 1000 chars · 100 overlap |
| **Prompt Format** | Qwen native ChatML | `<\|im_start\|>` / `<\|im_end\|>` |
| **Compute** | CUDA (GPU) or CPU | auto-detected via `torch.cuda.is_available()` |
| **Visualisation** | Plotly + streamlit-plotly-events | WebGL scatter |
| **Dim. Reduction** | PaCMAP · UMAP · TriMap · t-SNE | selectable |
| **Clustering** | HDBSCAN · Leiden · K-Means · Spectral · Agglomerative | selectable |

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/biologypeak/Nectar.git
cd Nectar
```

### 2. Create the environment

```bash
pip install -r requirements.txt
```

### 3. Launch

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. The LanceDB store at `./nectar_lancedb/` is created automatically on first ingest and persists across sessions. The visualisation cache at `./nectar_cache/` is created on first projection.

---

## 🗂️ Project Structure

```
Nectar/
├── app.py                        # Entry point — home page + sidebar + global CSS
├── requirements.txt              # All Python dependencies
│
├── core/
│   └── backend.py                # All RAG/DB logic — UI-agnostic
│
├── pages/
│   ├── 1_Knowledge_Base.py       # Upload · Ingest · Index Builder · DB Stats
│   ├── 2_Query.py                # Natural language query · k-slider · Chunk Panel
│   ├── 3_Explorer.py             # Filter · Browse · Cards/Table view
│   └── 4_Vector_Space.py         # Interactive embedding space visualiser
│
├── nectar_lancedb/               # LanceDB root — auto-created on first ingest
│   └── nectar_papers.lance/      # Lance columnar table
│
└── nectar_cache/                 # Projection cache — auto-created on first projection
    └── vs_<sha1>.pkl             # Reducer + projection per method/corpus combination
```

---

## 📄 Pages

### `app.py` — Home & Navigation

The Streamlit entry point. Renders the home page with four feature cards and a persistent sidebar present on every page. The sidebar displays live database stats (papers, chunks, index status) by calling `get_db_stats()` from the backend on each render.

Navigation uses `st.page_link()` to route between the four pages:

```
📥  Knowledge Base   →  pages/1_Knowledge_Base.py
🔍  Query            →  pages/2_Query.py
🗂️  Explorer         →  pages/3_Explorer.py
🔭  Vector Space     →  pages/4_Vector_Space.py
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

A `st.selectbox` lets the user choose among three distance metrics. Selecting a metric immediately renders a detail card showing the metric's mathematical formula, description, value range, recommended use case, and a "recommended" badge for cosine. Clicking "Build Index" calls `build_index(metric_key)`. If the corpus has fewer than `INDEX_MIN_ROWS` (256) chunks, an error is shown with the current count and the minimum required. On success, a four-column KPI card shows the metric used, number of IVF partitions, PQ sub-vectors, and total indexed chunks.

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

### `pages/4_Vector_Space.py` — Vector Space Explorer

Interactive visualisation of the entire embedding space. The layout is a Plotly scatter (left, 2/3) and a query + chunk preview panel (right, 1/3).

**Controls**

A top bar exposes four selectors and one toggle:

| Control | Options |
|---|---|
| Reduction method | PaCMAP · UMAP · TriMap · t-SNE |
| Clustering | HDBSCAN · Leiden · K-Means · Spectral · Agglomerative |
| Color by | `cluster_label` · `paper_name` · `page` |
| 3D toggle | switches between 2D and 3D projection |

An expandable **Clustering parameters** section exposes `min_cluster_size` / `min_samples` (HDBSCAN), `resolution` / `n_neighbors` (Leiden), or `n_clusters` (K-Means, Spectral, Agglomerative).

Two action buttons separate the two expensive operations:

- **Build projection** — runs the selected reducer, saves to disk cache, then immediately applies clustering
- **Apply clustering** — re-runs only the clusterer on the existing projection (fast, no cache write needed)

**Projection cache**

Results are stored in `./nectar_cache/vs_{sha1}.pkl`. The cache key encodes method, number of rows, and number of components — changing any of these causes a cache miss and forces a new computation. The fitted reducer object is also cached so query projection is available after a Streamlit restart without re-fitting.

**Real-time query projection**

Typing a sentence in the right panel and clicking *Project →* embeds the text and calls `fitted_reducer.transform()` to place a ★ marker in the space. The nearest corpus point is found via `scipy.spatial.KDTree` to identify the landing cluster. Out-of-sample projection is supported by PaCMAP, UMAP, and t-SNE (openTSNE); TriMap shows a warning and suggests switching.

**Sub-clustering**

Once a query has been projected and a landing cluster identified, *Sub-cluster ★* replaces the full-corpus view with a new reduction + clustering computed on that cluster's chunks only. The sub-cluster indices are stored as a boolean mask over the full corpus in `st.session_state`. *← Full corpus* resets all state and returns to the global view. Sub-clustering can be applied recursively.

**Chunk preview**

Hovering over any point shows a Plotly tooltip with paper name, page number, and the first 220 characters of the chunk — rendered natively by Plotly with no Streamlit rerun. Clicking a point writes the full chunk to the right-hand preview panel (`st.session_state["vs_clicked_chunk"]`).

**Performance at scale**

The 2D scatter uses `go.Scattergl` (WebGL), which handles 20 000+ points without lag. Hover events are deliberately disabled (`hover_event=False` in `streamlit_plotly_events`) to prevent per-cursor-move reruns; only click events trigger Streamlit updates.

---

## 🔧 Backend API Reference

All functions are in `core/backend.py`. The module is UI-agnostic — every function accepts and returns plain Python types.

### Configuration constants

| Constant | Value | Description |
|---|---|---|
| `LLM_MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace LLM identifier |
| `EMBED_MODEL_ID` | `mixedbread-ai/mxbai-embed-xsmall-v1` | Embedding model identifier |
| `LANCE_DIR` | `./nectar_lancedb` | LanceDB root directory |
| `TABLE_NAME` | `nectar_papers` | LanceDB table name |
| `EMBED_DIM` | `384` | Embedding output dimension |
| `INDEX_MIN_ROWS` | `256` | Minimum chunks required to build IVF_PQ index |

### PyArrow Schema

```python
SCHEMA = pa.schema([
    pa.field("paper_id",   pa.string()),                      # MD5 of absolute filepath — dedup key
    pa.field("paper_name", pa.string()),                      # Original PDF filename
    pa.field("page",       pa.int32()),                       # Source page number (0-indexed)
    pa.field("chunk",      pa.string()),                      # Raw text of the chunk
    pa.field("vector",     pa.list_(pa.float32(), 384)),      # Embedding vector
])
```

### Singletons

**`get_llm() → HuggingFacePipeline`**
Returns the loaded Qwen2.5-0.5B-Instruct pipeline. Loaded once on first call and cached in `_llm_instance`. Uses `torch.float16` on CUDA, `float32` on CPU. Parameters: `max_new_tokens=512`, `temperature=0.5`, `repetition_penalty=1.1`.

**`get_embedding_model() → HuggingFaceEmbeddings`**
Returns the mxbai-embed-xsmall-v1 model. Loaded once and cached in `_embedding_instance`. Embeddings are L2-normalized (`normalize_embeddings=True`) with a retrieval prefix prompt applied at encode time.

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
Returns `True` if a row with the file's `paper_id` already exists in the table.

**`embed_texts(texts: list[str]) → np.ndarray`**
Embeds a list of strings using the embedding model singleton. Returns a `float32` NumPy array of shape `(len(texts), 384)`.

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

**`clear_database() → bool`**
Drops the `nectar_papers` table from LanceDB. Returns `True`.

### Index builder

**`build_index(metric_key: str) → dict`**
Creates an IVF_PQ ANN index on the `vector` column. `metric_key` is one of `"cosine"`, `"l2"`, `"dot"`. Auto-tunes parameters:

- `num_partitions = max(1, n // 4096)`
- `num_sub_vectors = EMBED_DIM // 8` → 48

Returns `ok=False` with a descriptive message if the table is empty or has fewer than `INDEX_MIN_ROWS` chunks.

### Retrieval

**`retrieve_chunks(query: str, k: int) → list[dict]`**
Embeds the query, performs a vector search, and returns the top-k results:

```python
[{
    "paper_name": str,
    "page":       int,
    "chunk":      str,
    "affinity":   float,   # max(0, 1 − distance/2), range [0, 1]
    "distance":   float
}]
```

**`answer_query(query: str, k: int) → dict`**
Full RAG pipeline: retrieves chunks, builds context string, runs the Qwen ChatML prompt chain:

```python
{
    "answer": str,
    "chunks": list[dict],
    "error":  str|None
}
```

Returns `error` string (not raises) on any exception so the UI can display it gracefully.

### Explorer

**`explore_database(paper_filter, page_min, page_max, keyword, limit) → list[dict]`**
Loads the full table and applies four sequential filters: paper name substring, page range, keyword substring, row limit. Returns list of dicts with fields `paper_name`, `page`, `chunk`.

**`get_paper_names() → list[str]`**
Returns a sorted list of unique paper filenames currently in the database.

### Vector Space

**`load_all_vectors() → tuple[np.ndarray, pd.DataFrame]`**
Loads every row from LanceDB. Returns `(vectors, meta_df)` where:
- `vectors` — `float32` ndarray of shape `(n, 384)`
- `meta_df` — DataFrame with columns `paper_id`, `paper_name`, `page`, `chunk`; index aligned with `vectors` rows

Returns `(empty_array, empty_df)` when the table does not exist.

---

## 🗂️ IVF_PQ Index Builder — Metric Reference

| Metric | Formula | Range | Recommended for |
|---|---|---|---|
| **cosine** | `1 − (a·b / ‖a‖‖b‖)` | [0, 2] | Normalized text embeddings — **default for mxbai** |
| **l2** | `√Σ(aᵢ−bᵢ)²` | [0, ∞) | Non-normalized vectors, geometry-sensitive tasks |
| **dot** | `−(a·b)` | (−∞, ∞) | Max inner-product search, recommendation |

### Auto-tuned parameters

| Parameter | Formula | Example (n = 10 000) |
|---|---|---|
| `num_partitions` | `max(1, n // 4096)` | 2 |
| `num_sub_vectors` | `EMBED_DIM // 8` | 48 |

---

## 🔭 Reduction & Clustering Reference

### Dimensionality reduction

| Method | Package | Out-of-sample | Scale | Notes |
|---|---|---|---|---|
| **PaCMAP** | `pacmap` | ✅ `transform()` | any | Default — most stable, strong global structure preservation |
| **UMAP** | `umap-learn` | ✅ `transform()` | any | Fast, widely adopted in bioinformatics |
| **TriMap** | `trimap` | ❌ | > 10 k | Fastest on very large corpora, no query projection |
| **t-SNE** | `openTSNE` | ✅ `transform()` | < 10 k | Classic local-structure method; slower at scale |

### Clustering

| Method | Package | Fixed k | Noise | Notes |
|---|---|---|---|---|
| **HDBSCAN** | `hdbscan` | ❌ auto | ✅ (`-1`) | Default — density-based, best for sub-clustering |
| **Leiden** | `leidenalg` + `igraph` | ❌ resolution | ❌ | Graph-based; standard in scRNA-seq pipelines |
| **K-Means** | `sklearn` | ✅ | ❌ | Fast baseline, user-controlled cluster count |
| **Spectral** | `sklearn` | ✅ | ❌ | Good for non-convex shapes; slow > 5 k points |
| **Agglomerative** | `sklearn` | ✅ | ❌ | Hierarchical; linkage-based, interpretable |

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
embed_texts() → mxbai-embed-xsmall-v1 → float32[384] vectors
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
    ╔══════════════════════════════════╗   ╔══════════════════════════════════════════╗
    ║  Query page                      ║   ║  Vector Space page                       ║
    ╠══════════════════════════════════╣   ╠══════════════════════════════════════════╣
    ║  answer_query(query, k)          ║   ║  load_all_vectors()                      ║
    ║    └─ retrieve_chunks(query, k)  ║   ║    └─ all float32[384] vectors + meta    ║
    ║         └─ embed_query()         ║   ║  run_reducer(method, vectors, dims)      ║
    ║         └─ tbl.search().limit(k) ║   ║    └─ 2D/3D projection                  ║
    ║              → top-k (doc, dist) ║   ║  run_clusterer(method, projection)      ║
    ║    └─ affinity = 1 − dist/2      ║   ║    └─ integer labels                    ║
    ║    └─ ChatML prompt + Qwen LLM   ║   ║  save to ./nectar_cache/ (pickle)       ║
    ║    └─ answer + chunks returned   ║   ║  project_query() → ★ marker             ║
    ╚══════════════════════════════════╝   ║  sub-cluster: mask → re-project subset  ║
                                           ╚══════════════════════════════════════════╝
    │
    ▼
Streamlit renders: answer-box + chunk cards with affinity bars
                   Plotly Scattergl + chunk preview panel
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace model identifier |
| `EMBED_MODEL_ID` | `mixedbread-ai/mxbai-embed-xsmall-v1` | Embedding model |
| `EMBED_DIM` | `384` | Embedding output dimension |
| `LANCE_DIR` | `./nectar_lancedb` | LanceDB root directory |
| `TABLE_NAME` | `nectar_papers` | LanceDB table name |
| `INDEX_MIN_ROWS` | `256` | Minimum chunks to allow IVF_PQ index training |
| `DEVICE` | auto (CUDA / CPU) | Compute backend |
| `chunk_size` | `1000` | Characters per chunk |
| `chunk_overlap` | `100` | Overlap between adjacent chunks |
| `k` (slider default) | `5` | Retrieved chunks per query (UI range: 1–20) |
| `max_new_tokens` | `512` | Maximum LLM output length |
| `temperature` | `0.5` | LLM sampling temperature |
| `repetition_penalty` | `1.1` | Penalises repeated tokens |

---

## Roadmap

The following features are planned for v5.

### Paper Mesh & Auto-Annotation

Semantic structure extraction on ingest: each paper is automatically segmented into Abstract, Methods, Results, Discussion, and Conclusions sections. Section labels are stored as a new `section` column in the LanceDB schema, enabling section-aware retrieval. A parallel auto-tagging pipeline generates LLM-derived keywords, MeSH terms, and named entities (genes, diseases, compounds) stored as additional typed columns. A visual paper graph (nodes = papers, edges = cosine similarity between centroids) is auto-computed on each corpus update.

### PyArrow / DuckDB Analytics Layer

Direct SQL queries over the Lance table via DuckDB, without loading data into memory. Enables advanced corpus analytics: chunk length distributions, per-paper coverage heatmaps, duplicate detection, and vocabulary statistics — all rendered as Streamlit charts on a dedicated Analytics page.

---

## 🩺 Nectar in Healthcare

Nectar is purpose-built for the volume and heterogeneity of biomedical literature. The persistent LanceDB store scales from a handful of papers to hundreds without changing the workflow. The Vector Space Explorer is particularly suited to biomedical corpora: HDBSCAN naturally isolates disease areas, drug classes, and methodological clusters; Leiden mirrors the community-detection approach of Seurat and Scanpy used in single-cell genomics.

**Typical research workflows:**

- **Systematic review acceleration** — query the entire corpus for PICO elements across hundreds of studies; the Chunk Panel cites every source with page-level precision
- **Microbiome & gut-brain axis research** — cross-reference findings from cohort studies, RCTs, and meta-analyses; IVF_PQ cosine search surfaces the most semantically aligned passages
- **Corpus topology exploration** — use the Vector Space page to visually identify thematic clusters, detect outlier papers, and understand how your literature is distributed in semantic space
- **Sub-field drilling** — project a specific hypothesis into the space, identify which cluster it lands in, and sub-cluster that region to find the most tightly related passages
- **Clinical trial monitoring** — ingest trial PDFs and extract adverse events, endpoints, and inclusion criteria by natural language query; the Explorer page allows direct keyword search for regulatory terms
- **Medical education** — residents and students query curated paper sets for evidence-based answers with full source traceability down to the page

> Nectar does not replace clinical judgment. All answers must be verified against the primary sources shown in the Chunk Panel.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Open a Pull Request against the `main` branch with a description of the change

Issues, feedback from healthcare researchers, and contributions to the analytics and annotation roadmap are especially welcome.

---

## License

MIT — use freely, credit appreciated.

---

<div align="center">

Built by [@biologypeak](https://github.com/biologypeak) · Powered by open-source models · No data leaves your machine

**v1** — single PDF · in-memory ChromaDB &nbsp;→&nbsp;
**v2** — multi-PDF · persistent ChromaDB · k-slider · chunk explorer &nbsp;→&nbsp;
**v3** — LanceDB · PyArrow schema · IVF_PQ index · metric selector &nbsp;→&nbsp;
**v4** — Streamlit multipage · decoupled backend · Knowledge Base · Query · Explorer · Vector Space

</div>
