"""
Page 2 — Query
Natural language querying · retrieval + reranking · Answer panel · Chunk Explorer with dual metrics
"""

import sys, os, html as _html
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from core.backend import answer_query, row_count, get_paper_names

st.set_page_config(page_title="Query — Nectar", page_icon="🔍", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"]   { background:#0f1117; border-right:1px solid #1e2130; }
[data-testid="stSidebar"] * { color:#e0e4f0 !important; }
header[data-testid="stHeader"] { background:#0f1117; }
.main .block-container { padding-top:2rem; padding-bottom:3rem; }
.nectar-card   { background:#1a1d2e; border:1px solid #2a2d3e; border-radius:10px; padding:1.4rem 1.6rem; margin-bottom:1rem; }
.chunk-card    { background:#12151f; border:1px solid #2a2d3e; border-left:3px solid #7c9ef5; border-radius:6px; padding:1rem 1.2rem; margin-bottom:0.8rem; font-size:0.88rem; line-height:1.6; color:#c8cfe0; }
.chunk-meta    { font-size:0.72rem; color:#8890a8; margin-bottom:0.5rem; letter-spacing:0.03em; }
.answer-box    { background:#12151f; border:1px solid #2a2d3e; border-radius:8px; padding:1.4rem 1.6rem; font-size:0.95rem; line-height:1.75; color:#dce3f5; white-space:pre-wrap; }
.section-label { font-size:0.7rem; letter-spacing:0.12em; text-transform:uppercase; color:#7c9ef5; font-weight:600; margin-bottom:0.5rem; }
.metric-pill   { display:inline-block; border-radius:4px; padding:1px 7px; font-size:0.70rem; font-weight:600; letter-spacing:0.04em; }
.pill-sim      { background:#1e2a4a; color:#7c9ef5; }
.pill-rr       { background:#1e3530; color:#4ecca3; }
hr { border-color:#1e2130; }
.stButton > button { border-radius:6px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────
st.markdown("## 🔍 Query")
st.markdown(
    "<div style='color:#8890a8; font-size:0.9rem; margin-bottom:1.5rem;'>"
    "Ask questions across your corpus in natural language. "
    "Retrieval uses vector similarity; a cross-encoder reranker then selects the most relevant chunks."
    "</div>",
    unsafe_allow_html=True,
)

n_chunks = row_count()
if n_chunks == 0:
    st.warning("The database is empty. Go to **Knowledge Base** and ingest at least one PDF.")
    st.stop()

# ── Layout: left = controls + chunks, right = answer ──────────
left, right = st.columns([1, 2], gap="large")

# ────────────────────────────────────────────────────────────────
# LEFT — Query controls + Chunk Explorer
# ────────────────────────────────────────────────────────────────
with left:
    st.markdown('<div class="section-label">Retrieval Settings</div>', unsafe_allow_html=True)

    k_retrieve = st.slider(
        "Candidates retrieved by similarity",
        min_value=20,
        max_value=200,
        value=50,
        step=5,
        help=(
            "Number of chunks fetched from the vector store by cosine similarity. "
            "These are then re-scored by the cross-encoder reranker."
        ),
    )

    k_rerank = st.slider(
        "Best chunks after reranking",
        min_value=5,
        max_value=30,
        value=10,
        step=1,
        help=(
            "Number of chunks kept after cross-encoder reranking "
            "(mxbai-rerank-xsmall-v1). Only these are passed to the LLM as context "
            "and shown in the chunk panel."
        ),
    )

    # Clamp silently in case user edits state directly
    k_rerank = min(k_rerank, k_retrieve)

    rerank_threshold_pct = st.slider(
        "Minimum rerank score to answer",
        min_value=0,
        max_value=50,
        value=10,
        step=1,
        format="%d%%",
        help=(
            "If every retrieved chunk scores below this threshold after reranking, "
            "the model will reply that it lacks sufficient information rather than "
            "generating an answer from low-confidence context. Set to 0 to disable."
        ),
    )
    rerank_threshold = rerank_threshold_pct / 100.0

    st.markdown(
        f"<div style='font-size:0.8rem; color:#8890a8; margin-top:-0.3rem; margin-bottom:1rem;'>"
        f"Searching across <b style='color:#e0e4f0;'>{n_chunks:,}</b> indexed chunks."
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        '<div class="section-label">Retrieved Chunks</div>'
        '<div style="font-size:0.68rem; color:#555e78; margin-top:-0.2rem; margin-bottom:0.6rem;">'
        '✦ sorted by rerank score · ◈ similarity for reference'
        '</div>',
        unsafe_allow_html=True,
    )
    chunk_placeholder = st.empty()
    chunk_placeholder.markdown(
        "<div style='color:#555e78; font-size:0.85rem; font-style:italic;'>"
        "Chunks will appear here after a query."
        "</div>",
        unsafe_allow_html=True,
    )

# ────────────────────────────────────────────────────────────────
# RIGHT — Query input + Answer
# ────────────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="section-label">Question</div>', unsafe_allow_html=True)

    query = st.text_area(
        "Enter your research question",
        placeholder="e.g. What is the role of Lactobacillus in modulating the gut-brain axis?",
        height=110,
        label_visibility="collapsed",
    )

    submit = st.button("  Run Query  ", type="primary", use_container_width=False)

    st.markdown("---")
    st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)
    answer_placeholder = st.empty()
    answer_placeholder.markdown(
        "<div style='color:#555e78; font-size:0.85rem; font-style:italic;'>"
        "The model's answer will appear here."
        "</div>",
        unsafe_allow_html=True,
    )

# ── Run query ─────────────────────────────────────────────────
if submit:
    if not query.strip():
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner(f"Retrieving {k_retrieve} candidates, reranking to top {k_rerank}…"):
            result = answer_query(query.strip(), k_retrieve, k_rerank, rerank_threshold)

        if result["error"]:
            answer_placeholder.error(f"Error: {result['error']}")
        else:
            # Render answer — use a warning style when below threshold
            if result.get("below_threshold"):
                answer_placeholder.markdown(
                    f"<div class='answer-box' style='border-color:#4a3020; color:#e8a87c;'>"
                    f"⚠ {result['answer']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                answer_placeholder.markdown(
                    f"<div class='answer-box'>{result['answer']}</div>",
                    unsafe_allow_html=True,
                )

            # Render chunk panel
            chunks = result["chunks"]
            # Defensive sort: ensure display order is always by rerank score descending
            chunks = sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)
            if chunks:
                chunk_html = ""
                for i, c in enumerate(chunks, 1):
                    sim_pct     = int(c["affinity"] * 100)
                    rr_pct      = int(c.get("rerank_score", 0) * 100)
                    sim_bar     = sim_pct
                    rr_bar      = rr_pct
                    preview     = _html.escape(c["chunk"][:480]) + ("…" if len(c["chunk"]) > 480 else "")
                    paper_name  = _html.escape(c["paper_name"])
                    # Dim chunks with very low rerank score (< 10%)
                    card_opacity = "1.0" if rr_pct >= 10 else "0.55"

                    chunk_html += f"""
                    <div class="chunk-card" style="opacity:{card_opacity};">
                        <div class="chunk-meta">
                            #{i} &nbsp;·&nbsp;
                            <b style="color:#c8cfe0;">{paper_name}</b>
                            &nbsp;·&nbsp; p.{c['page']}
                        </div>
                        <div style="display:flex; gap:0.6rem; align-items:center; margin-bottom:0.55rem; flex-wrap:wrap;">
                            <span class="metric-pill pill-rr">
                                ✦ Rerank&nbsp;&nbsp;<b>{rr_pct}%</b>
                            </span>
                            <span class="metric-pill pill-sim">
                                ◈ Similarity&nbsp;&nbsp;<b>{sim_pct}%</b>
                            </span>
                            <span style="font-size:0.68rem; color:#555e78;">dist {c['distance']}</span>
                        </div>
                        <div style="display:flex; gap:6px; margin-bottom:0.7rem;">
                            <div style="flex:1;">
                                <div style="font-size:0.62rem; color:#2a4040; margin-bottom:2px;">rerank</div>
                                <div style="background:#1e2130; border-radius:3px; height:3px; width:100%;">
                                    <div style="background:linear-gradient(90deg,#4ecca3,#38a89d); border-radius:3px; height:3px; width:{rr_bar}%;"></div>
                                </div>
                            </div>
                            <div style="flex:1;">
                                <div style="font-size:0.62rem; color:#4a5270; margin-bottom:2px;">similarity</div>
                                <div style="background:#1e2130; border-radius:3px; height:3px; width:100%;">
                                    <div style="background:linear-gradient(90deg,#7c9ef5,#a78bfa); border-radius:3px; height:3px; width:{sim_bar}%;"></div>
                                </div>
                            </div>
                        </div>
                        {preview}
                    </div>
                    """
                chunk_placeholder.markdown(chunk_html, unsafe_allow_html=True)
            else:
                chunk_placeholder.markdown(
                    "<div style='color:#555e78; font-size:0.85rem;'>No chunks retrieved.</div>",
                    unsafe_allow_html=True,
                )
