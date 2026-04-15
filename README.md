<div align="center">

# 🍯 Nectar

### Local RAG Platform for Any Document Format — Built for Researchers, Designed for Health

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain.com)
[![LanceDB](https://img.shields.io/badge/LanceDB-columnar%20VDB-4F86C6?style=flat-square)](https://lancedb.com)
[![PyArrow](https://img.shields.io/badge/PyArrow-typed%20schema-E25A1C?style=flat-square)](https://arrow.apache.org/docs/python)
[![Version](https://img.shields.io/badge/version-4.2-F59E0B?style=flat-square)](#)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

**Query any document collection in natural language. 100% local. Zero API keys. Zero cloud.**

[What's New](#-whats-new) · [Quickstart](#-quickstart) · [Architecture](#️-architecture) · [Pages](#-pages) · [API Reference](#-backend-api-reference) · [Roadmap](#-roadmap) · [Healthcare](#-nectar-in-healthcare)

---

</div>

## What is Nectar?

Nectar is a **fully local Retrieval-Augmented Generation (RAG) platform** for building and querying a personal knowledge base. Upload documents in any format — PDFs, Word files, spreadsheets, Markdown notes, source code, e-books and more — embed them into a persistent vector store, and interrogate your entire library in natural language, with full traceability from answer back to source chunk and page. A dedicated **Vector Space Explorer** lets you visualise, cluster, and navigate the entire embedding space interactively.

Built on **Qwen2.5-0.5B-Instruct** (LLM), **mixedbread-ai/mxbai-embed-xsmall-v1** (embeddings), **mixedbread-ai/mxbai-rerank-xsmall-v1** (reranker), **LanceDB** (vector store), and **Streamlit** (UI), Nectar is designed as a professional research tool with a clean dark interface suited for daily use.

> No data leaves your machine. No subscription. No warmup between sessions.

---

## ✨ What's New

### v4.2 — Universal document loader · Reranking pipeline · Smarter ingestion

**25+ document formats supported**
Every file is now converted to Markdown before embedding — PDFs, Word documents, OpenDocument, RTF, HTML, Markdown, LaTeX, RST, EPUB, CSV, TSV, JSON, JSONL, YAML, and all common source code formats. A new `core/loader.py` module handles conversion using format-specific libraries (python-docx, ebooklib, pylatexenc, BeautifulSoup, and others).

**Configurable ingestion pipeline**
The Knowledge Base page now exposes a full **Processing Pipeline** panel with three tabs:
- *Pre-processing* — toggle character normalisation (ligatures, non-breaking spaces), header/footer removal, and two types of line-break merging
- *Splitting* — control chunk size and overlap with sliders; see overlap as a percentage of chunk size in real time
- *Post-filtering* — set a minimum chunk length, a maximum stop-word density to discard noisy fragments, and a Jaccard deduplication threshold to drop near-identical passages within the same document

**Cross-encoder reranking**
Retrieval now runs in two stages. After the initial vector similarity search fetches a pool of candidates, a cross-encoder (`mxbai-rerank-xsmall-v1`) re-scores every candidate by reading the query and chunk together. Only the top-N chunks by rerank score are passed to the LLM. The Query page exposes three sliders: number of candidates to retrieve, number of chunks to keep after reranking, and a confidence threshold below which the model declines to answer rather than guess.

**Improved chunk panel**
Retrieved chunks are sorted by rerank score (not similarity). Chunks that fall below 10% rerank score are visually dimmed. The panel header makes the sort order explicit.

---

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

| Component | Technology | Notes |
|---|---|---|
| **UI Framework** | Streamlit multipage app | ≥ 1.32 |
| **LLM** | `Qwen/Qwen2.5-0.5B-Instruct` | Generates answers from retrieved context |
| **Embeddings** | `mixedbread-ai/mxbai-embed-xsmall-v1` | 384-dim, L2-normalised, cosine distance |
| **Reranker** | `mixedbread-ai/mxbai-rerank-xsmall-v1` | Cross-encoder, sigmoid output 0–1 |
| **Vector Store** | LanceDB — Lance columnar format | ≥ 0.6, persistent on disk |
| **Schema** | PyArrow typed `pa.schema` | ≥ 14.0 |
| **ANN Index** | IVF_PQ — selectable metric | cosine / l2 / dot |
| **Document Loader** | `core/loader.py` — 25+ formats | PDF, DOCX, ODT, RTF, HTML, MD, CSV, JSON, EPUB, LaTeX, RST, code… |
| **Text Splitter** | `RecursiveCharacterTextSplitter` | Default 1000 chars · 100 overlap (user-adjustable) |
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
│   ├── backend.py                # All RAG/DB logic — UI-agnostic
│   └── loader.py                 # Universal document loader (25+ formats → Markdown)
│
├── pages/
│   ├── 1_Knowledge_Base.py       # Upload · Ingest · Processing Pipeline · Index Builder · DB Stats
│   ├── 2_Query.py                # Natural language query · reranking · threshold · Chunk Panel
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

A multi-file uploader accepts any number of files in a single batch across all 25+ supported formats. Below the uploader, a collapsible **Processing Pipeline** panel lets users configure ingestion before clicking "Embed & Add to Database":

- *1 · Pre-processing tab* — four toggles: character normalisation (ligatures, special spaces, bullet variants), header/footer removal, hyphenated line-break merging, and soft line-break merging
- *2 · Splitting tab* — chunk size (200–4000 chars) and chunk overlap (0–500 chars) sliders; a live label shows overlap as a percentage of chunk size
- *3 · Post-filtering tab* — minimum chunk length, maximum stop-word ratio (0 = off), and Jaccard deduplication threshold (0 = off)

On ingestion, files are written to a temporary directory and passed to `ingest_documents()`. Results are displayed as a per-file status row with badge and format label:

| Badge | Meaning |
|---|---|
| `INDEXED` (green) | File successfully processed, embedded, and written to LanceDB |
| `SKIPPED` (yellow) | File already present in the DB (path hash dedup) — no re-embedding |
| `EMPTY` (red) | No usable text remained after cleaning and filtering |
| `ERROR` (red) | Extraction or embedding failed — error message displayed |

**Section 2 — IVF_PQ Index Builder**

A `st.selectbox` lets the user choose among three distance metrics. Selecting a metric immediately renders a detail card showing the metric's formula, description, value range, recommended use case, and a "recommended" badge for cosine. Clicking "Build Index" calls `build_index(metric_key)`. If the corpus has fewer than 256 chunks, an error is shown. On success, a four-column KPI card shows the metric used, number of IVF partitions, PQ sub-vectors, and total indexed chunks.

**Section 3 — Database Statistics**

Live KPIs rendered as metric cards: total documents, total chunks, average chunks per document, and ANN index status. Below the KPIs, a `st.dataframe` shows a per-document breakdown sorted by chunk count descending. A "Refresh" button and a "Clear Database" button are also provided.

---

### `pages/2_Query.py` — Query

The primary RAG interface. Layout is two columns: left (1/3) for retrieval controls and the Chunk Panel, right (2/3) for question input and the answer.

**Retrieval Settings**

Three sliders control the two-stage retrieval pipeline:

| Slider | Default | Effect |
|---|---|---|
| Candidates retrieved by similarity | 50 | How many chunks to fetch from the vector store in the first pass |
| Best chunks after reranking | 10 | How many of those candidates to keep after the cross-encoder re-scores them |
| Minimum rerank score to answer | 10% | If every retained chunk scores below this, the model replies "not enough information" instead of generating an answer. Set to 0 to disable. |

The current corpus size is shown below the sliders for context.

**Question Input**

A `st.text_area` accepts the research question. Clicking "Run Query" calls `answer_query(query, k_retrieve, k_rerank, rerank_threshold)` from the backend.

**Answer Panel**

The LLM response is rendered inside a styled `.answer-box`. If the rerank threshold was not met, the box switches to an amber warning style with a ⚠ prefix. If the backend returns an error, `st.error()` is shown instead.

**Chunk Panel**

After every query, the left column renders one card per chunk, **sorted by rerank score descending**. Each card shows:

- Document name and page number
- Rerank score (primary, left) and similarity score (secondary, right) — both as percentage pills and progress bars
- Raw cosine distance
- Up to 480 characters of chunk text as preview
- Chunks with rerank score below 10% are visually dimmed to 55% opacity

All text from document content is HTML-escaped before rendering, so PDF passages containing `<`, `>` or `&` display correctly instead of breaking the card layout.

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
| `EMBED_MODEL_ID` | `mixedbread-ai/mxbai-embed-xsmall-v1` | Embedding model (384-dim) |
| `RERANK_MODEL_ID` | `mixedbread-ai/mxbai-rerank-xsmall-v1` | Cross-encoder reranker |
| `LANCE_DIR` | `./nectar_lancedb` | LanceDB root directory |
| `TABLE_NAME` | `nectar_papers` | LanceDB table name |
| `EMBED_DIM` | `384` | Embedding output dimension |
| `INDEX_MIN_ROWS` | `256` | Minimum chunks required to build IVF_PQ index |

### PyArrow Schema

```python
SCHEMA = pa.schema([
    pa.field("paper_id",   pa.string()),                 # MD5 of absolute filepath — dedup key
    pa.field("paper_name", pa.string()),                 # Original filename
    pa.field("page",       pa.int32()),                  # Page or section number (0-indexed)
    pa.field("chunk",      pa.string()),                 # Text content of the chunk
    pa.field("vector",     pa.list_(pa.float32(), 384)), # Embedding vector
])
```

### Singletons

**`get_llm() → HuggingFacePipeline`**
Loads Qwen2.5-0.5B-Instruct on first call and caches it. Uses `float16` on CUDA, `float32` on CPU. Generation params: `max_new_tokens=512`, `temperature=0.5`, `repetition_penalty=1.1`.

**`get_embedding_model() → HuggingFaceEmbeddings`**
Loads mxbai-embed-xsmall-v1 on first call and caches it. L2-normalised output; retrieval prefix prompt applied at encode time.

**`get_reranker() → CrossEncoder`**
Loads mxbai-rerank-xsmall-v1 on first call and caches it. Sigmoid activation applied so scores are in [0, 1].

### Ingestion config

```python
@dataclass
class IngestConfig:
    normalize_chars: bool        = True   # ligatures, non-breaking spaces, bullets
    remove_headers_footers: bool = True   # strip short noise lines at page edges
    merge_hyphen_breaks: bool    = True   # "word-\nrest" → "wordrest"
    merge_soft_breaks: bool      = True   # single \n → space (keep \n\n as paragraph break)
    chunk_size: int              = 1000
    chunk_overlap: int           = 100
    min_chunk_chars: int         = 80     # discard shorter chunks
    max_stopword_ratio: float    = 0.0    # 0 = disabled
    dedup_threshold: float       = 0.0    # 0 = disabled (Jaccard)
```

### Ingestion

**`ingest_document(filepath: str, config: IngestConfig | None) → dict`**
Loads any supported format via `core/loader.py`, applies the cleaning/splitting/filtering pipeline, and appends to LanceDB. Returns:

```python
{
    "name":    str,          # filename
    "status":  str,          # "ok" | "skipped" | "empty" | "error"
    "chunks":  int,          # chunks written (0 if skipped/error)
    "skipped": bool,
    "error":   str | None,
    "format":  str           # human-readable format label, e.g. "PDF", "Word Document"
}
```

`ingest_pdf` and `ingest_multiple_pdfs` are kept as backward-compatible aliases.

**`ingest_documents(filepaths: list[str], config: IngestConfig | None) → list[dict]`**
Calls `ingest_document` for each path and returns the list of result dicts.

**`get_db_stats() → dict`**
Returns aggregate statistics:

```python
{
    "total_chunks":  int,
    "total_papers":  int,
    "papers":        list[dict],   # per-document: paper_name, chunks, pages
    "has_index":     bool
}
```

**`clear_database() → bool`**
Drops the `nectar_papers` table from LanceDB. Returns `True`.

### Index builder

**`build_index(metric_key: str) → dict`**
Creates an IVF_PQ ANN index on the `vector` column. `metric_key` is one of `"cosine"`, `"l2"`, `"dot"`. Auto-tunes `num_partitions = max(1, n // 4096)` and `num_sub_vectors = EMBED_DIM // 8`. Returns `ok=False` with a descriptive message if the corpus is too small.

### Retrieval

**`retrieve_chunks(query: str, k: int) → list[dict]`**
Embeds the query, performs a vector search, and returns the top-k chunks with `paper_name`, `page`, `chunk`, `affinity` (`max(0, 1 − distance/2)`), and `distance`.

**`rerank_chunks(query: str, chunks: list[dict], top_n: int) → list[dict]`**
Scores each (query, chunk) pair with the cross-encoder. Adds `rerank_score` (float, 0–1) to each dict and returns the top-n sorted descending.

**`answer_query(query, k_retrieve, k_rerank, rerank_threshold=0.0) → dict`**
Full two-stage RAG pipeline. If all reranked chunks score below `rerank_threshold`, returns a fixed "not enough information" answer without calling the LLM. Returns:

```python
{
    "answer":          str,
    "chunks":          list[dict],   # top chunks with affinity + rerank_score
    "error":           str | None,
    "below_threshold": bool          # True when the threshold blocked generation
}
```

### Explorer

**`explore_database(paper_filter, page_min, page_max, keyword, limit) → list[dict]`**
Applies four sequential filters (document name, page range, keyword, row limit) and returns matching chunks.

**`get_paper_names() → list[str]`**
Returns a sorted list of unique document filenames in the database.

### Vector Space

**`load_all_vectors() → tuple[np.ndarray, pd.DataFrame]`**
Loads every row from LanceDB. Returns `(vectors, meta_df)` where `vectors` is a `float32` ndarray of shape `(n, 384)` and `meta_df` is a DataFrame with `paper_id`, `paper_name`, `page`, `chunk` aligned by index.

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
Any file (PDF, DOCX, CSV, MD, EPUB, code, …)
    │
    ▼
core/loader.py  →  Markdown text  (format-specific conversion)
    │
    ▼
ingest_document() per file:
    _clean_text()   →  normalise · remove noise · fix line breaks
    RecursiveCharacterTextSplitter (default 1000 chars / 100 overlap)
    _filter_chunks() →  min length · stopword density · deduplication
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
    ╔══════════════════════════════════════════╗   ╔══════════════════════════════════════════╗
    ║  Query page                              ║   ║  Vector Space page                       ║
    ╠══════════════════════════════════════════╣   ╠══════════════════════════════════════════╣
    ║  answer_query(query, k_ret, k_rr, thr)  ║   ║  load_all_vectors()                      ║
    ║    └─ retrieve_chunks(query, k_ret)      ║   ║    └─ all float32[384] vectors + meta    ║
    ║         └─ embed query                   ║   ║  run_reducer(method, vectors, dims)      ║
    ║         └─ tbl.search().limit(k_ret)     ║   ║    └─ 2D/3D projection                  ║
    ║         └─ affinity = 1 − dist/2         ║   ║  run_clusterer(method, projection)       ║
    ║    └─ rerank_chunks(query, cands, k_rr)  ║   ║    └─ integer labels                    ║
    ║         └─ cross-encoder scores 0–1      ║   ║  save to ./nectar_cache/ (pickle)       ║
    ║         └─ sort by rerank_score desc     ║   ║  project_query() → ★ marker             ║
    ║    └─ threshold check (skip LLM if low)  ║   ║  sub-cluster: mask → re-project subset  ║
    ║    └─ ChatML prompt + Qwen LLM           ║   ╚══════════════════════════════════════════╝
    ║    └─ answer + chunks + below_threshold  ║
    ╚══════════════════════════════════════════╝
    │
    ▼
Streamlit renders: answer-box (amber if threshold not met)
                   chunk cards sorted by rerank · affinity bars
                   Plotly Scattergl + chunk preview panel
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace LLM |
| `EMBED_MODEL_ID` | `mixedbread-ai/mxbai-embed-xsmall-v1` | Embedding model |
| `RERANK_MODEL_ID` | `mixedbread-ai/mxbai-rerank-xsmall-v1` | Cross-encoder reranker |
| `EMBED_DIM` | `384` | Embedding dimension |
| `LANCE_DIR` | `./nectar_lancedb` | LanceDB root |
| `TABLE_NAME` | `nectar_papers` | LanceDB table name |
| `INDEX_MIN_ROWS` | `256` | Minimum chunks to train IVF_PQ index |
| `DEVICE` | auto (CUDA / CPU) | Compute backend |
| `chunk_size` | `1000` | Characters per chunk (user-adjustable 200–4000) |
| `chunk_overlap` | `100` | Overlap between chunks (user-adjustable 0–500) |
| `min_chunk_chars` | `80` | Post-filter: discard chunks shorter than this |
| `max_stopword_ratio` | `0.0` | Post-filter: 0 = off; max stop-word fraction allowed |
| `dedup_threshold` | `0.0` | Post-filter: 0 = off; Jaccard similarity above which a chunk is a duplicate |
| `k_retrieve` (slider) | `50` | Candidates fetched from the vector store |
| `k_rerank` (slider) | `10` | Chunks kept after reranking |
| `rerank_threshold` (slider) | `10%` | Minimum rerank score; 0 = disabled |
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
**v4.0** — Streamlit multipage · decoupled backend · Knowledge Base · Query · Explorer · Vector Space &nbsp;→&nbsp;
**v4.1** — Vector Space Explorer · PaCMAP/UMAP/TriMap/t-SNE · HDBSCAN/Leiden · sub-clustering · disk cache &nbsp;→&nbsp;
**v4.2** — 25+ document formats · cross-encoder reranking · configurable ingestion pipeline · rerank threshold

</div>
