# ============================================================
# Nectar v3 — RAG Chatbot
# mixedbread-ai Embeddings + Qwen2.5 0.5B + LanceDB
#
# NEW in v3 (vs v2):
#   [A] LanceDB replaces ChromaDB — columnar Lance format on disk
#   [B] Explicit PyArrow schema (paper_name, chunk, vector)
#   [C] IVF_PQ index builder with metric selector (L2/Cosine/Dot)
#       + dropdown with per-metric explanation
#       + guard: index blocked when row count is insufficient
#
# PRESERVED from v2:
#   [1] Multi-PDF ingestion with duplicate detection
#   [2] Persistent DB (disk-backed, survives restarts)
#   [3] Slider to control k (retrieved chunks)
#   [4] Chunk Explorer Panel (source PDF + page + affinity bar)
#
# 100% locale — nessuna API key necessaria
# pip install lancedb pyarrow
# ============================================================

import os
import hashlib
import traceback
from pathlib import Path

import numpy as np
import pyarrow as pa
import lancedb
import torch
import gradio as gr

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURAZIONE
# ============================================================

LLM_MODEL_ID   = "Qwen/Qwen2.5-0.5B-Instruct"
EMBED_MODEL_ID = "mixedbread-ai/mxbai-embed-xsmall-v1"
LANCE_DIR      = "./nectar_lancedb"       # [A] LanceDB root directory
TABLE_NAME     = "nectar_papers"
EMBED_DIM      = 384                     # mxbai-embed-large-v1 output dim

# IVF_PQ index: minimum rows needed for meaningful training
# LanceDB requires at least num_partitions vectors; we guard at 256
INDEX_MIN_ROWS = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Dispositivo: {DEVICE}")
print(f"[INFO] LanceDB in: {os.path.abspath(LANCE_DIR)}")

# ============================================================
# [B] PyArrow SCHEMA
# Three columns: paper_name (str), chunk (str), vector (float32 list)
# ============================================================

SCHEMA = pa.schema([
    pa.field("paper_id",   pa.string()),          # MD5 dedup key
    pa.field("paper_name", pa.string()),           # filename shown in UI
    pa.field("page",       pa.int32()),            # source page number
    pa.field("chunk",      pa.string()),           # raw text of the chunk
    pa.field("vector",     pa.list_(pa.float32(), EMBED_DIM)),  # embedding
])

# ============================================================
# METRIC DESCRIPTIONS — shown in the UI dropdown
# ============================================================

METRIC_INFO = {
    "cosine": (
        "cosine",
        "📐 Cosine — misura l'angolo tra due vettori ignorando la magnitudine. "
        "Range: [0, 2] dove 0 = identici. "
        "Ideale per embedding testuali normalizzati (come mxbai-embed-large-v1). "
        "✅ Consigliata per RAG su testi scientifici."
    ),
    "l2": (
        "l2",
        "📏 L2 (Euclidean) — misura la distanza geometrica nello spazio vettoriale. "
        "Range: [0, ∞). Sensibile alla magnitudine del vettore. "
        "Buona quando i vettori NON sono normalizzati o si vuole tenere conto della lunghezza. "
        "⚠️ Con embedding normalizzati produce risultati simili a cosine."
    ),
    "dot": (
        "dot",
        "⚡ Dot Product — prodotto scalare tra due vettori. "
        "Range: (-∞, ∞). Con vettori normalizzati equivale alla cosine similarity. "
        "Più veloce da calcolare ma richiede vettori ben scalati. "
        "✅ Ottima se si vuole privilegiare vettori di alta magnitudine (es. documenti lunghi)."
    ),
}

# ============================================================
# SINGLETON — LLM e Embedding (caricati una sola volta)
# ============================================================

_llm_instance       = None
_embedding_instance = None

def get_llm():
    global _llm_instance
    if _llm_instance is None:
        print("[INFO] Caricamento LLM...")
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
        print("[INFO] LLM pronto.")
    return _llm_instance


def get_embedding_model():
    global _embedding_instance
    if _embedding_instance is None:
        print("[INFO] Caricamento embedding model...")
        _embedding_instance = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_ID,
            model_kwargs={"device": DEVICE},
            encode_kwargs={
                "normalize_embeddings": True,
                "prompt": "Represent this sentence for searching relevant passages: ",
            },
        )
        print("[INFO] Embedding model pronto.")
    return _embedding_instance

# ============================================================
# [A] LANCEDB — connection & table helpers
# ============================================================

def get_db():
    return lancedb.connect(LANCE_DIR)


def get_table():
    """Open the table if it exists, else return None."""
    db = get_db()
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)
    return None


def get_or_create_table():
    """Open existing table or create an empty one with the PyArrow schema."""
    db = get_db()
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)
    # [B] Create with explicit schema — no data yet
    return db.create_table(TABLE_NAME, schema=SCHEMA)


def row_count() -> int:
    tbl = get_table()
    if tbl is None:
        return 0
    return tbl.count_rows()

# ============================================================
# [1] MULTI-PDF INGESTION
# ============================================================

def paper_id(filepath: str) -> str:
    return hashlib.md5(os.path.abspath(filepath).encode()).hexdigest()


def already_indexed(filepath: str) -> bool:
    tbl = get_table()
    if tbl is None:
        return False
    pid = paper_id(filepath)
    # LanceDB SQL filter
    results = tbl.search().where(f"paper_id = '{pid}'", prefilter=True).limit(1).to_arrow()
    return len(results) > 0


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    return np.array(model.embed_documents(texts), dtype=np.float32)


def ingest_pdf(filepath: str) -> str:
    pid  = paper_id(filepath)
    name = Path(filepath).name

    if already_indexed(filepath):
        return f"⚡ '{name}' già nel DB — skip."

    print(f"[INFO] Indicizzazione: {name}")
    loader = PyPDFLoader(filepath)
    pages  = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = splitter.split_documents(pages)

    if not chunks:
        return f"⚠️ '{name}' — nessun testo estraibile."

    texts      = [c.page_content for c in chunks]
    page_nums  = [int(c.metadata.get("page", 0)) for c in chunks]
    embeddings = embed_texts(texts)

    # [B] Build PyArrow RecordBatch from typed arrays
    batch = pa.table({
        "paper_id":   pa.array([pid]  * len(texts), type=pa.string()),
        "paper_name": pa.array([name] * len(texts), type=pa.string()),
        "page":       pa.array(page_nums,            type=pa.int32()),
        "chunk":      pa.array(texts,                type=pa.string()),
        "vector":     pa.array(embeddings.tolist(),  type=pa.list_(pa.float32(), EMBED_DIM)),
    })

    tbl = get_or_create_table()
    tbl.add(batch)
    print(f"[INFO] '{name}' → {len(texts)} chunk aggiunti.")
    return f"✅ '{name}' — {len(texts)} chunk indicizzati."


def ingest_multiple_pdfs(files) -> str:
    if not files:
        return "⚠️ Nessun file selezionato."
    lines = []
    for f in files:
        path = f.name if hasattr(f, "name") else str(f)
        try:
            lines.append(ingest_pdf(path))
        except Exception as e:
            lines.append(f"❌ Errore su '{Path(path).name}': {e}")
    n = row_count()
    lines.append(f"\n📊 Totale chunk nel DB: {n}")
    if n < INDEX_MIN_ROWS:
        lines.append(f"💡 Servono almeno {INDEX_MIN_ROWS} chunk per creare un indice ANN.")
    return "\n".join(lines)


def list_indexed_papers() -> str:
    tbl = get_table()
    if tbl is None or row_count() == 0:
        return "📭 Nessun paper nel DB."
    df   = tbl.to_pandas()[["paper_id", "paper_name"]].drop_duplicates("paper_id")
    lines = [f"📚 Paper nel DB ({len(df)} totali):"]
    for _, row in df.iterrows():
        lines.append(f"  • {row['paper_name']}")
    lines.append(f"\n📊 Chunk totali: {row_count()}")
    return "\n".join(lines)


def clear_database() -> str:
    db = get_db()
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)
    return "🗑️ Database svuotato."

# ============================================================
# [C] IVF_PQ INDEX BUILDER
# ============================================================

def build_index(metric_label: str) -> str:
    """
    Create an IVF_PQ ANN index on the vector column.
    Guards against insufficient row count.
    Metric is one of: cosine, l2, dot
    """
    tbl = get_table()
    if tbl is None or row_count() == 0:
        return "❌ Il DB è vuoto. Indicizza almeno un PDF prima di creare un indice."

    n = row_count()
    if n < INDEX_MIN_ROWS:
        return (
            f"⚠️ Numero di chunk insufficiente per creare un indice ANN.\n"
            f"   Chunk attuali: {n}\n"
            f"   Minimo richiesto: {INDEX_MIN_ROWS}\n"
            f"   Aggiungi altri paper e riprova."
        )

    # Auto-tune IVF_PQ parameters following LanceDB docs:
    # num_partitions = max(1, n // 4096)
    # num_sub_vectors = EMBED_DIM // 8
    num_partitions  = max(1, n // 4096)
    num_sub_vectors = EMBED_DIM // 8   # 1024 // 8 = 128

    metric_key = metric_label.split(" ")[0].lower()   # "cosine", "l2", "dot"
    metric_api = METRIC_INFO.get(metric_key, ("l2",))[0]

    try:
        print(f"[INFO] Creazione indice IVF_PQ | metric={metric_api} "
              f"| partitions={num_partitions} | sub_vectors={num_sub_vectors}")
        tbl.create_index(
            metric=metric_api,
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            vector_column_name="vector",
            replace=True,
        )
        return (
            f"✅ Indice IVF_PQ creato con successo.\n"
            f"   Metrica: {metric_api.upper()}\n"
            f"   Partizioni (IVF): {num_partitions}\n"
            f"   Sub-vettori (PQ): {num_sub_vectors}\n"
            f"   Chunk indicizzati: {n}\n\n"
            f"Le prossime query useranno la ricerca approssimata (ANN) anziché la scansione esaustiva."
        )
    except Exception as e:
        return f"❌ Errore durante la creazione dell'indice:\n{traceback.format_exc()}"

# ============================================================
# PROMPT — formato ChatML nativo di Qwen
# ============================================================

def build_prompt():
    template = (
        "<|im_start|>system\n"
        "You are a helpful assistant. Answer the question using only the provided context. "
        "If the answer is not in the context, say \"I don't have enough information to answer this question.\""
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Context:\n{context}\n\n"
        "Question: {question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return PromptTemplate(input_variables=["context", "question"], template=template)

# ============================================================
# [3] + [4] RETRIEVAL WITH k-CONTROL AND CHUNK PANEL
# ============================================================

def retrieve_with_scores(query: str, k: int):
    """
    Embed the query, search LanceDB, return:
      - context_text : plain text for the LLM
      - panel_md     : Markdown for the Chunk Explorer Panel
    """
    tbl = get_table()
    if tbl is None:
        return "", "⚠️ DB vuoto."

    emb = get_embedding_model().embed_query(query)

    # LanceDB returns _distance column (lower = more similar for l2/cosine)
    results_df = (
        tbl.search(emb, vector_column_name="vector")
           .limit(k)
           .select(["paper_name", "page", "chunk"])
           .to_pandas()
    )

    if results_df.empty:
        return "", "⚠️ Nessun risultato trovato."

    context_parts = []
    panel_parts   = []

    for rank, row in enumerate(results_df.itertuples(), start=1):
        distance   = getattr(row, "_distance", 0.0)
        # Convert distance to affinity score ∈ [0,1]
        # For cosine/l2: distance ∈ [0,2]; affinity = 1 - distance/2
        affinity   = max(0.0, 1.0 - float(distance) / 2.0)
        pct        = int(affinity * 100)
        bar_full   = int(affinity * 20)
        bar        = "█" * bar_full + "░" * (20 - bar_full)

        text = str(row.chunk).strip()
        context_parts.append(text)

        panel_parts.append(
            f"### Chunk {rank} — 📄 {row.paper_name}\n"
            f"**Pagina:** {row.page} &nbsp;|&nbsp; "
            f"**Affinità:** `{pct}%` &nbsp; `{bar}`\n\n"
            f"> {text[:400]}{'…' if len(text) > 400 else ''}\n"
        )

    context_text = "\n\n".join(context_parts)
    panel_md     = "\n---\n".join(panel_parts)
    return context_text, panel_md


def retriever_qa(query: str, k: int):
    if not query or query.strip() == "":
        return "❌ Inserisci una domanda.", ""
    if row_count() == 0:
        return "❌ Il DB è vuoto. Carica almeno un PDF prima.", ""
    try:
        context_text, panel_md = retrieve_with_scores(query, k=int(k))
        if not context_text:
            return "⚠️ Nessun contesto trovato.", panel_md

        llm    = get_llm()
        prompt = build_prompt()
        chain  = prompt | llm | StrOutputParser()
        result = chain.invoke({"context": context_text, "question": query})

        if "<|im_end|>" in result:
            result = result.split("<|im_end|>")[0]

        return result.strip(), panel_md
    except Exception as e:
        print(f"[ERROR]\n{traceback.format_exc()}")
        return f"❌ Errore: {str(e)}", ""

# ============================================================
# GRADIO UI — Blocks layout
# ============================================================

METRIC_CHOICES = [
    "cosine — 📐 angolo tra vettori, ideale per testi normalizzati",
    "l2     — 📏 distanza euclidea, sensibile alla magnitudine",
    "dot    — ⚡ prodotto scalare, veloce con vettori scalati",
]

with gr.Blocks(
    title="🍯 Nectar v3 — LanceDB RAG",
    theme=gr.themes.Soft(),
    css=".chunk-panel { font-size:0.85rem; } .status-box { font-size:0.80rem; color:#555; }",
) as rag_application:

    gr.Markdown(
        "# 🍯 Nectar v3\n"
        "### Local RAG — Multi-PDF · LanceDB · PyArrow Schema · IVF_PQ Index · Chunk Explorer\n"
        "`100% locale` &nbsp;|&nbsp; "
        "`Embeddings: mxbai-embed-large-v1` &nbsp;|&nbsp; "
        "`LLM: Qwen2.5-0.5B-Instruct` &nbsp;|&nbsp; "
        "`VectorDB: LanceDB + Lance columnar format`"
    )

    # ── ROW 1: Ingestion + DB management ────────────────────
    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(
                label="📂 Carica uno o più PDF",
                file_count="multiple",
                file_types=[".pdf"],
                type="filepath",
            )
            ingest_btn    = gr.Button("⚡ Indicizza PDF", variant="primary")
            ingest_status = gr.Textbox(
                label="Stato indicizzazione",
                lines=5,
                interactive=False,
                elem_classes=["status-box"],
            )

        with gr.Column(scale=1):
            list_btn  = gr.Button("📚 Mostra paper nel DB")
            clear_btn = gr.Button("🗑️ Svuota DB", variant="stop")
            db_status = gr.Textbox(
                label="Database",
                lines=7,
                interactive=False,
                elem_classes=["status-box"],
            )

    gr.Markdown("---")

    # ── ROW 2: Index builder ─────────────────────────────────
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 🗂️ IVF_PQ Index Builder")
            metric_dropdown = gr.Dropdown(
                choices=METRIC_CHOICES,
                value=METRIC_CHOICES[0],
                label="📊 Metrica di distanza",
                info="Seleziona la metrica con cui costruire l'indice ANN",
            )
            metric_info_box = gr.Textbox(
                label="ℹ️ Come funziona questa metrica",
                value=METRIC_INFO["cosine"][1],
                lines=3,
                interactive=False,
                elem_classes=["status-box"],
            )
        with gr.Column(scale=1):
            index_btn    = gr.Button("🔨 Crea Indice IVF_PQ", variant="primary")
            index_status = gr.Textbox(
                label="Stato indice",
                lines=7,
                interactive=False,
                elem_classes=["status-box"],
            )

    gr.Markdown("---")

    # ── ROW 3: Query + Chunk Panel ───────────────────────────
    with gr.Row():

        # Left — Chunk Explorer [4]
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### 🔍 Chunk Retrieval Panel")
            k_slider = gr.Slider(
                minimum=1, maximum=20, value=3, step=1,
                label="📊 Chunk da recuperare (k)",
                info="Numero di frammenti rilevanti usati come contesto",
            )
            chunk_panel = gr.Markdown(
                value="_I chunk appariranno qui dopo la prima domanda._",
                elem_classes=["chunk-panel"],
            )

        # Right — Q&A [3]
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="💬 Domanda",
                lines=3,
                placeholder="Es: Qual è il ruolo del microbiota nell'asse gut-brain?",
            )
            ask_btn = gr.Button("🚀 Chiedi", variant="primary")
            answer_output = gr.Textbox(
                label="📝 Risposta",
                lines=12,
                interactive=False,
            )

    # ── EVENT BINDINGS ────────────────────────────────────────

    def on_metric_change(choice: str):
        key = choice.split(" ")[0].strip().lower()
        return METRIC_INFO.get(key, METRIC_INFO["cosine"])[1]

    metric_dropdown.change(
        fn=on_metric_change,
        inputs=[metric_dropdown],
        outputs=[metric_info_box],
    )

    ingest_btn.click(
        fn=ingest_multiple_pdfs,
        inputs=[file_input],
        outputs=[ingest_status],
    )
    list_btn.click(
        fn=list_indexed_papers,
        inputs=[],
        outputs=[db_status],
    )
    clear_btn.click(
        fn=clear_database,
        inputs=[],
        outputs=[db_status],
    )
    index_btn.click(
        fn=build_index,
        inputs=[metric_dropdown],
        outputs=[index_status],
    )
    ask_btn.click(
        fn=retriever_qa,
        inputs=[query_input, k_slider],
        outputs=[answer_output, chunk_panel],
    )
    query_input.submit(
        fn=retriever_qa,
        inputs=[query_input, k_slider],
        outputs=[answer_output, chunk_panel],
    )

# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    rag_application.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
    )