# ============================================================
# Nectar v2 — RAG Chatbot
# mixedbread-ai Embeddings + Qwen2.5 0.5B
#
# NEW in v2:
#   [1] Multi-PDF ingestion (upload N papers simultaneously)
#   [2] Persistent ChromaDB (disk-backed, survives restarts)
#   [3] Slider to control k (number of retrieved chunks)
#   [4] Chunk panel: source PDF name + affinity score per chunk
#
# 100% locale — nessuna API key necessaria
# ============================================================

import os
import re
import hashlib
import traceback
from pathlib import Path

import torch
import gradio as gr

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURAZIONE
# ============================================================

LLM_MODEL_ID   = "Qwen/Qwen2.5-0.5B-Instruct"
EMBED_MODEL_ID = "mixedbread-ai/mxbai-embed-large-v1"
CHROMA_DIR     = "./nectar_chroma_db"   # [2] Persistent storage path
COLLECTION     = "nectar_papers"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Dispositivo: {DEVICE}")
print(f"[INFO] ChromaDB persistente in: {os.path.abspath(CHROMA_DIR)}")

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
# [2] PERSISTENT CHROMADB
# ============================================================

def get_vectordb():
    """Return (or create) the persistent Chroma collection."""
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embedding_model(),
        persist_directory=CHROMA_DIR,
    )


def paper_id(filepath: str) -> str:
    """Stable ID for a PDF based on its absolute path."""
    return hashlib.md5(os.path.abspath(filepath).encode()).hexdigest()


def already_indexed(filepath: str) -> bool:
    """Check whether this PDF has already been embedded in the persistent DB."""
    pid = paper_id(filepath)
    db = get_vectordb()
    results = db.get(where={"paper_id": pid}, limit=1)
    return len(results["ids"]) > 0

# ============================================================
# [1] MULTI-PDF INGESTION
# ============================================================

def ingest_pdf(filepath: str) -> str:
    """
    Load, split and embed a single PDF.
    Skips ingestion if the paper is already in the persistent DB.
    Returns a status string.
    """
    pid  = paper_id(filepath)
    name = Path(filepath).name

    if already_indexed(filepath):
        return f"⚡ '{name}' già presente nel DB — skip."

    print(f"[INFO] Indicizzazione: {name}")
    loader = PyPDFLoader(filepath)
    pages  = loader.load()

    # Tag every chunk with paper metadata
    for doc in pages:
        doc.metadata["paper_id"]   = pid
        doc.metadata["paper_name"] = name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = splitter.split_documents(pages)

    # Propagate metadata to chunks (splitter may reset it)
    for chunk in chunks:
        chunk.metadata["paper_id"]   = pid
        chunk.metadata["paper_name"] = name

    db = get_vectordb()
    db.add_documents(chunks)
    print(f"[INFO] '{name}' → {len(chunks)} chunk indicizzati.")
    return f"✅ '{name}' — {len(chunks)} chunk indicizzati."


def ingest_multiple_pdfs(files) -> str:
    """
    Accept a list of file paths (Gradio multi-file upload).
    Returns a summary of what was ingested.
    """
    if not files:
        return "⚠️ Nessun file selezionato."

    lines = []
    for f in files:
        path = f.name if hasattr(f, "name") else str(f)
        try:
            lines.append(ingest_pdf(path))
        except Exception as e:
            lines.append(f"❌ Errore su '{Path(path).name}': {e}")
    return "\n".join(lines)


def list_indexed_papers() -> str:
    """Return a formatted list of all papers currently in the DB."""
    db  = get_vectordb()
    res = db.get(include=["metadatas"])
    if not res["ids"]:
        return "📭 Nessun paper nel DB."
    seen  = {}
    for meta in res["metadatas"]:
        pid  = meta.get("paper_id", "unknown")
        name = meta.get("paper_name", "unknown")
        seen[pid] = name
    lines = ["📚 Paper nel DB:"] + [f"  • {n}" for n in seen.values()]
    return "\n".join(lines)


def clear_database() -> str:
    """Delete all documents from the persistent collection."""
    db = get_vectordb()
    db.delete_collection()
    return "🗑️ Database svuotato."

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
    Query the persistent DB and return:
      - docs_text : plain context string for the LLM
      - panel_md  : Markdown for the chunk panel (source + affinity)
    """
    db = get_vectordb()

    # similarity_search_with_relevance_scores returns (doc, score) pairs
    # score ∈ [0, 1] where 1 = perfect match (cosine similarity after normalization)
    results = db.similarity_search_with_relevance_scores(query, k=k)

    if not results:
        return "", "⚠️ Nessun chunk trovato nel DB. Carica almeno un PDF."

    context_parts = []
    panel_parts   = []

    for rank, (doc, score) in enumerate(results, start=1):
        paper_name = doc.metadata.get("paper_name", "Unknown")
        page_num   = doc.metadata.get("page", "?")
        text       = doc.page_content.strip()

        # Context for the LLM (clean text only)
        context_parts.append(text)

        # Affinity bar (visual)
        pct      = int(score * 100)
        bar_full = int(score * 20)
        bar      = "█" * bar_full + "░" * (20 - bar_full)

        panel_parts.append(
            f"### Chunk {rank} — 📄 {paper_name}\n"
            f"**Pagina:** {page_num} &nbsp;|&nbsp; "
            f"**Affinità:** `{pct}%` &nbsp; `{bar}`\n\n"
            f"> {text[:400]}{'…' if len(text) > 400 else ''}\n"
        )

    docs_text = "\n\n".join(context_parts)
    panel_md  = "\n---\n".join(panel_parts)
    return docs_text, panel_md


def retriever_qa(query: str, k: int):
    """
    Main QA function wired to the Gradio UI.
    Returns (answer, chunk_panel_markdown).
    """
    # Validation
    if not query or query.strip() == "":
        return "❌ Inserisci una domanda.", ""

    db = get_vectordb()
    if not db.get(limit=1)["ids"]:
        return "❌ Il database è vuoto. Carica almeno un PDF prima di fare domande.", ""

    try:
        print(f"[DEBUG] Query: {query} | k={k}")

        context_text, panel_md = retrieve_with_scores(query, k=int(k))

        if not context_text:
            return "⚠️ Nessun contesto trovato per questa domanda.", panel_md

        llm    = get_llm()
        prompt = build_prompt()

        # Build chain with pre-fetched context (bypasses LangChain retriever
        # so we can capture scores independently)
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"context": context_text, "question": query})

        # Clean residual Qwen tokens
        if "<|im_end|>" in result:
            result = result.split("<|im_end|>")[0]

        print("[DEBUG] Risposta generata.")
        return result.strip(), panel_md

    except Exception as e:
        print(f"[ERROR]\n{traceback.format_exc()}")
        return f"❌ Errore: {str(e)}", ""

# ============================================================
# GRADIO UI — Blocks layout
# ============================================================

with gr.Blocks(
    title="🍯 Nectar v2 — Local RAG",
    theme=gr.themes.Soft(),
    css="""
        .chunk-panel { font-size: 0.85rem; }
        .status-box  { font-size: 0.80rem; color: #555; }
    """,
) as rag_application:

    gr.Markdown(
        "# 🍯 Nectar v2\n"
        "### Local RAG — Multi-PDF · Persistent DB · Chunk Explorer\n"
        "`100% locale` &nbsp;|&nbsp; "
        "`Embeddings: mxbai-embed-large-v1` &nbsp;|&nbsp; "
        "`LLM: Qwen2.5-0.5B-Instruct`"
    )

    # ── TOP ROW: ingestion + DB management ──────────────────
    with gr.Row():
        with gr.Column(scale=2):
            # [1] Multi-file upload
            file_input = gr.File(
                label="📂 Carica uno o più PDF",
                file_count="multiple",       # ← multi-paper
                file_types=[".pdf"],
                type="filepath",
            )
            ingest_btn   = gr.Button("⚡ Indicizza PDF", variant="primary")
            ingest_status = gr.Textbox(
                label="Stato indicizzazione",
                lines=4,
                interactive=False,
                elem_classes=["status-box"],
            )

        with gr.Column(scale=1):
            list_btn    = gr.Button("📚 Mostra paper nel DB")
            clear_btn   = gr.Button("🗑️ Svuota DB", variant="stop")
            db_status   = gr.Textbox(
                label="Database",
                lines=6,
                interactive=False,
                elem_classes=["status-box"],
            )

    gr.Markdown("---")

    # ── BOTTOM ROW: query + answer + chunk panel ─────────────
    with gr.Row():

        # Left column — chunk explorer [4]
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### 🔍 Chunk Retrieval Panel")

            # [3] Slider for k
            k_slider = gr.Slider(
                minimum=1,
                maximum=20,
                value=3,
                step=1,
                label="📊 Chunk da recuperare (k)",
                info="Quanti frammenti rilevanti usare come contesto",
            )

            chunk_panel = gr.Markdown(
                value="_I chunk recuperati appariranno qui dopo la prima domanda._",
                elem_classes=["chunk-panel"],
            )

        # Right column — Q&A
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="💬 Domanda",
                lines=3,
                placeholder="Es: Qual è il ruolo del microbiota intestinale nell'asse gut-brain?",
            )
            ask_btn = gr.Button("🚀 Chiedi", variant="primary")
            answer_output = gr.Textbox(
                label="📝 Risposta",
                lines=12,
                interactive=False,
            )

    # ── EVENT BINDINGS ────────────────────────────────────────

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

    ask_btn.click(
        fn=retriever_qa,
        inputs=[query_input, k_slider],
        outputs=[answer_output, chunk_panel],
    )

    # Also trigger on Enter key in the textbox
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