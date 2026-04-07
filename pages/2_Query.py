"""
Page 2 — Query
Natural language querying · k-slider · Answer panel · Chunk Explorer with affinity scores
"""

import sys, os
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
hr { border-color:#1e2130; }
.stButton > button { border-radius:6px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────
st.markdown("## 🔍 Query")
st.markdown(
    "<div style='color:#8890a8; font-size:0.9rem; margin-bottom:1.5rem;'>"
    "Ask questions across your corpus in natural language. "
    "The answer is grounded exclusively in the retrieved chunks — no hallucination from pre-training knowledge."
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

    k = st.slider(
        "Chunks to retrieve (k)",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help=(
            "Number of vector-store chunks passed as context to the LLM. "
            "Higher k = more context, potentially slower and noisier. "
            "Lower k = more precise but may miss relevant passages."
        ),
    )

    st.markdown(
        f"<div style='font-size:0.8rem; color:#8890a8; margin-top:-0.3rem; margin-bottom:1rem;'>"
        f"Searching across <b style='color:#e0e4f0;'>{n_chunks:,}</b> indexed chunks."
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown('<div class="section-label">Retrieved Chunks</div>', unsafe_allow_html=True)
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
        with st.spinner("Retrieving and generating…"):
            result = answer_query(query.strip(), k)

        if result["error"]:
            answer_placeholder.error(f"Error: {result['error']}")
        else:
            # Render answer
            answer_placeholder.markdown(
                f"<div class='answer-box'>{result['answer']}</div>",
                unsafe_allow_html=True,
            )

            # Render chunk panel
            chunks = result["chunks"]
            if chunks:
                chunk_html = ""
                for i, c in enumerate(chunks, 1):
                    pct      = int(c["affinity"] * 100)
                    bar_w    = int(c["affinity"] * 100)
                    preview  = c["chunk"][:480] + ("…" if len(c["chunk"]) > 480 else "")

                    chunk_html += f"""
                    <div class="chunk-card">
                        <div class="chunk-meta">
                            #{i} &nbsp;·&nbsp;
                            <b style="color:#c8cfe0;">{c['paper_name']}</b>
                            &nbsp;·&nbsp; p.{c['page']}
                            &nbsp;·&nbsp;
                            <b style="color:#7c9ef5;">{pct}% affinity</b>
                            &nbsp;·&nbsp; dist {c['distance']}
                        </div>
                        <div style="background:#1e2130; border-radius:3px; height:4px; width:100%; margin-bottom:0.6rem;">
                            <div style="background:linear-gradient(90deg,#7c9ef5,#a78bfa); border-radius:3px; height:4px; width:{bar_w}%;"></div>
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