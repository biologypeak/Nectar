# ============================================================
# Nectar v4 — Core Backend
# All RAG/DB logic, completely decoupled from any UI framework
# ============================================================

import os
import hashlib
import traceback
from pathlib import Path

import numpy as np
import pyarrow as pa
import lancedb
import torch

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────
LLM_MODEL_ID    = "Qwen/Qwen2.5-0.5B-Instruct"
EMBED_MODEL_ID  = "mixedbread-ai/mxbai-embed-xsmall-v1" # mxbai-embed-large-v1
RERANK_MODEL_ID = "mixedbread-ai/mxbai-rerank-xsmall-v1"
LANCE_DIR       = "./nectar_lancedb"
TABLE_NAME      = "nectar_papers"
EMBED_DIM       = 384   #1024
INDEX_MIN_ROWS  = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── PyArrow Schema ────────────────────────────────────────────
SCHEMA = pa.schema([
    pa.field("paper_id",   pa.string()),
    pa.field("paper_name", pa.string()),
    pa.field("page",       pa.int32()),
    pa.field("chunk",      pa.string()),
    pa.field("vector",     pa.list_(pa.float32(), EMBED_DIM)),
])

# ── Metric descriptions ───────────────────────────────────────
METRIC_INFO = {
    "cosine": {
        "api_key": "cosine",
        "label":   "Cosine Distance",
        "formula": "1 − (a·b / ‖a‖‖b‖)",
        "range":   "[0, 2] — 0 = identical",
        "description": (
            "Measures the angle between two vectors, ignoring magnitude. "
            "Ideal for normalized text embeddings (mxbai-embed-large-v1 outputs unit vectors). "
            "Two chunks discussing the same concept will have a very small angle, "
            "regardless of how many times the concept is mentioned."
        ),
        "recommended": True,
        "use_case": "Scientific text retrieval, RAG pipelines, semantic search",
    },
    "l2": {
        "api_key": "l2",
        "label":   "L2 (Euclidean) Distance",
        "formula": "√Σ(aᵢ − bᵢ)²",
        "range":   "[0, ∞) — 0 = identical",
        "description": (
            "Measures geometric distance in the embedding space. "
            "Sensitive to vector magnitude — longer documents may produce larger vectors "
            "and inflate distances. Equivalent to cosine for unit-normalized vectors."
        ),
        "recommended": False,
        "use_case": "Non-normalized embeddings, image retrieval, clustering tasks",
    },
    "dot": {
        "api_key": "dot",
        "label":   "Dot Product",
        "formula": "−(a · b)",
        "range":   "(−∞, ∞) — higher = more similar",
        "description": (
            "Computes the scalar product between two vectors. "
            "With unit-normalized vectors (Nectar's default) it is mathematically equivalent to cosine. "
            "Faster to compute. Favors high-magnitude vectors, "
            "which can bias results toward longer documents."
        ),
        "recommended": False,
        "use_case": "Max inner-product search, recommendation systems, dense retrieval",
    },
}

# ── Singletons ────────────────────────────────────────────────
_llm_instance       = None
_embedding_instance = None
_reranker_instance  = None


def get_llm():
    global _llm_instance
    if _llm_instance is None:
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto",
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.5,
            do_sample=True,
            repetition_penalty=1.1,
            return_full_text=False,
        )
        _llm_instance = HuggingFacePipeline(pipeline=pipe)
    return _llm_instance


def get_embedding_model():
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_ID,
            model_kwargs={"device": DEVICE},
            encode_kwargs={
                "normalize_embeddings": True,
                "prompt": "Represent this sentence for searching relevant passages: ",
            },
        )
    return _embedding_instance


def get_reranker():
    global _reranker_instance
    if _reranker_instance is None:
        from sentence_transformers import CrossEncoder
        import torch.nn as nn
        _reranker_instance = CrossEncoder(
            RERANK_MODEL_ID,
            device=DEVICE,
            default_activation_function=nn.Sigmoid(),
        )
    return _reranker_instance


# ── LanceDB helpers ───────────────────────────────────────────
def get_db():
    return lancedb.connect(LANCE_DIR)


def get_table():
    db = get_db()
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)
    return None


def get_or_create_table():
    db = get_db()
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)
    return db.create_table(TABLE_NAME, schema=SCHEMA)


def row_count() -> int:
    tbl = get_table()
    return 0 if tbl is None else tbl.count_rows()


def paper_id(filepath: str) -> str:
    return hashlib.md5(os.path.abspath(filepath).encode()).hexdigest()


def already_indexed(filepath: str) -> bool:
    tbl = get_table()
    if tbl is None:
        return False
    pid = paper_id(filepath)
    results = tbl.search().where(f"paper_id = '{pid}'", prefilter=True).limit(1).to_arrow()
    return len(results) > 0


def embed_texts(texts: list[str]) -> np.ndarray:
    return np.array(get_embedding_model().embed_documents(texts), dtype=np.float32)


def rerank_chunks(query: str, chunks: list[dict], top_n: int) -> list[dict]:
    """
    Cross-encoder reranking with mxbai-rerank-xsmall-v1.
    Adds 'rerank_score' (float 0–1) to each chunk dict.
    Returns the top_n chunks sorted by rerank_score descending.
    """
    if not chunks:
        return []
    reranker = get_reranker()
    pairs    = [(query, c["chunk"]) for c in chunks]
    scores   = reranker.predict(pairs)          # numpy array, sigmoid-activated → [0, 1]
    for c, score in zip(chunks, scores):
        c["rerank_score"] = round(float(score), 4)
    ranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_n]


# ── Ingestion ─────────────────────────────────────────────────
def ingest_pdf(filepath: str) -> dict:
    """
    Returns dict with keys: name, status, chunks, skipped, error
    """
    name = Path(filepath).name
    pid  = paper_id(filepath)

    if already_indexed(filepath):
        return {"name": name, "status": "skipped", "chunks": 0, "skipped": True, "error": None}

    try:
        loader  = PyPDFLoader(filepath)
        pages   = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)
        chunks  = splitter.split_documents(pages)

        if not chunks:
            return {"name": name, "status": "empty", "chunks": 0, "skipped": False, "error": "No extractable text"}

        texts      = [c.page_content for c in chunks]
        page_nums  = [int(c.metadata.get("page", 0)) for c in chunks]
        embeddings = embed_texts(texts)

        batch = pa.table({
            "paper_id":   pa.array([pid]  * len(texts), type=pa.string()),
            "paper_name": pa.array([name] * len(texts), type=pa.string()),
            "page":       pa.array(page_nums,            type=pa.int32()),
            "chunk":      pa.array(texts,                type=pa.string()),
            "vector":     pa.array(embeddings.tolist(),  type=pa.list_(pa.float32(), EMBED_DIM)),
        })
        tbl = get_or_create_table()
        tbl.add(batch)
        return {"name": name, "status": "ok", "chunks": len(texts), "skipped": False, "error": None}

    except Exception as e:
        return {"name": name, "status": "error", "chunks": 0, "skipped": False, "error": str(e)}


def ingest_multiple_pdfs(filepaths: list[str]) -> list[dict]:
    return [ingest_pdf(fp) for fp in filepaths]


def get_db_stats() -> dict:
    """Return summary statistics for the database overview."""
    tbl = get_table()
    if tbl is None:
        return {"total_chunks": 0, "total_papers": 0, "papers": [], "has_index": False}

    #df = tbl.to_pandas(columns=["paper_id", "paper_name", "page"])
    df = tbl.search().select(["paper_id", "paper_name", "page"]).to_pandas()
    papers_df = df.groupby("paper_id").agg(
        paper_name=("paper_name", "first"),
        chunks=("paper_id", "count"),
        pages=("page", "nunique"),
    ).reset_index()

    try:
        idx_stats = tbl.index_stats("vector")
        has_index = idx_stats is not None
    except Exception:
        has_index = False

    return {
        "total_chunks": len(df),
        "total_papers": len(papers_df),
        "papers": papers_df.to_dict("records"),
        "has_index": has_index,
    }


def clear_database() -> bool:
    db = get_db()
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)
    return True


# ── Index builder ─────────────────────────────────────────────
def build_index(metric_key: str) -> dict:
    """
    Returns dict with keys: ok (bool), message, params
    """
    tbl = get_table()
    n   = row_count()

    if tbl is None or n == 0:
        return {"ok": False, "message": "Database is empty. Ingest at least one PDF first.", "params": {}}

    if n < INDEX_MIN_ROWS:
        return {
            "ok": False,
            "message": (
                f"Insufficient data for ANN index training.\n"
                f"Current chunks: **{n}**\n"
                f"Minimum required: **{INDEX_MIN_ROWS}**\n\n"
                f"Add more papers and try again."
            ),
            "params": {},
        }

    metric_api      = METRIC_INFO[metric_key]["api_key"]
    num_partitions  = max(1, n // 4096)
    num_sub_vectors = EMBED_DIM // 8

    try:
        tbl.create_index(
            metric=metric_api,
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            vector_column_name="vector",
            replace=True,
        )
        params = {
            "metric":           metric_api.upper(),
            "num_partitions":   num_partitions,
            "num_sub_vectors":  num_sub_vectors,
            "indexed_chunks":   n,
        }
        return {"ok": True, "message": "IVF_PQ index created successfully.", "params": params}
    except Exception as e:
        return {"ok": False, "message": f"Index creation failed: {e}", "params": {}}


# ── Retrieval ─────────────────────────────────────────────────
def retrieve_chunks(query: str, k: int) -> list[dict]:
    """
    Returns list of dicts: paper_name, page, chunk, affinity, distance
    """
    tbl = get_table()
    if tbl is None:
        return []

    emb = get_embedding_model().embed_query(query)
    df  = (
        tbl.search(emb, vector_column_name="vector")
           .limit(k)
           .select(["paper_name", "page", "chunk"])
           .to_pandas()
    )

    results = []
    for row in df.itertuples():
        distance = getattr(row, "_distance", 0.0)
        affinity = max(0.0, 1.0 - float(distance) / 2.0)
        results.append({
            "paper_name": row.paper_name,
            "page":       int(row.page),
            "chunk":      str(row.chunk).strip(),
            "affinity":   round(affinity, 4),
            "distance":   round(float(distance), 4),
        })
    return results


def answer_query(query: str, k_retrieve: int, k_rerank: int,
                 rerank_threshold: float = 0.0) -> dict:
    """
    Two-stage retrieval + reranking pipeline.
    1. Retrieve k_retrieve candidates by vector similarity.
    2. Rerank with mxbai-rerank-xsmall-v1, keep top k_rerank.
    3. If rerank_threshold > 0, discard chunks whose rerank_score < threshold.
       If no chunks survive the threshold, return a no-information answer immediately
       without calling the LLM.
    Returns dict: answer (str), chunks (list[dict]), error (str|None), below_threshold (bool)
    Each chunk dict has: paper_name, page, chunk, affinity, distance, rerank_score
    """
    if not query or not query.strip():
        return {"answer": "", "chunks": [], "error": "Empty query.", "below_threshold": False}
    if row_count() == 0:
        return {"answer": "", "chunks": [], "error": "Database is empty.", "below_threshold": False}

    try:
        candidates = retrieve_chunks(query, k_retrieve)
        if not candidates:
            return {"answer": "No relevant context found for this query.", "chunks": [], "error": None, "below_threshold": False}

        reranked = rerank_chunks(query, candidates, k_rerank)

        # Apply rerank threshold: keep only chunks that meet the minimum score
        if rerank_threshold > 0.0:
            above = [c for c in reranked if c["rerank_score"] >= rerank_threshold]
            if not above:
                return {
                    "answer": "I don't have enough information in the knowledge base to answer this question.",
                    "chunks": reranked,   # still return all chunks so the panel shows why
                    "error": None,
                    "below_threshold": True,
                }
            reranked = above

        context = "\n\n".join(c["chunk"] for c in reranked)
        prompt  = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "<|im_start|>system\n"
                "You are a helpful assistant. Answer the question using only the provided context. "
                "If the answer is not in the context, say \"I don't have enough information to answer this question.\""
                "<|im_end|>\n<|im_start|>user\nContext:\n{context}\n\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n"
            ),
        )
        chain  = prompt | get_llm() | StrOutputParser()
        result = chain.invoke({"context": context, "question": query})
        if "<|im_end|>" in result:
            result = result.split("<|im_end|>")[0]
        return {"answer": result.strip(), "chunks": reranked, "error": None, "below_threshold": False}

    except Exception as e:
        return {"answer": "", "chunks": [], "error": traceback.format_exc(), "below_threshold": False}


# ── Explorer ─────────────────────────────────────────────────
def explore_database(paper_filter: str = "", page_min: int = 0, page_max: int = 9999,
                     keyword: str = "", limit: int = 100) -> list[dict]:
    """
    Simple filter-based exploration of the vector table.
    Returns list of dicts: paper_name, page, chunk (no vectors).
    """
    tbl = get_table()
    if tbl is None:
        return []

    #df = tbl.to_pandas(columns=["paper_name", "page", "chunk"])
    df = tbl.search().select(["paper_name", "page", "chunk"]).to_pandas()

    if paper_filter:
        df = df[df["paper_name"].str.contains(paper_filter, case=False, na=False)]
    df = df[(df["page"] >= page_min) & (df["page"] <= page_max)]
    if keyword:
        df = df[df["chunk"].str.contains(keyword, case=False, na=False)]

    return df.head(limit).to_dict("records")


def get_paper_names() -> list[str]:
    tbl = get_table()
    if tbl is None:
        return []
    #df = tbl.to_pandas(columns=["paper_name"]).drop_duplicates()
    df = tbl.search().select(["paper_name"]).to_pandas().drop_duplicates()
    return sorted(df["paper_name"].tolist())


def load_all_vectors() -> tuple[np.ndarray, "pd.DataFrame"]:
    """
    Load every row from LanceDB.
    Returns (vectors, meta_df) where:
      - vectors  : float32 ndarray shape (n, EMBED_DIM)
      - meta_df  : DataFrame with columns paper_id, paper_name, page, chunk
                   index is reset and aligned with vectors rows.
    Returns (empty_array, empty_df) when the table does not exist.
    """
    import pandas as pd

    tbl = get_table()
    if tbl is None:
        return np.empty((0, EMBED_DIM), dtype=np.float32), pd.DataFrame()

    df = (
        tbl.search()
           .select(["paper_id", "paper_name", "page", "chunk", "vector"])
           .to_pandas()
    )
    if df.empty:
        return np.empty((0, EMBED_DIM), dtype=np.float32), pd.DataFrame()

    vectors = np.stack(df["vector"].values).astype(np.float32)
    meta    = df.drop(columns=["vector"]).reset_index(drop=True)
    return vectors, meta

    