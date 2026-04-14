"""
Nectar — Local RAG Platform for Scientific Literature
Entry point: run with `streamlit run app.py`
"""

import streamlit as st

st.set_page_config(
    page_title="Nectar — Scientific RAG",
    page_icon="🍯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f1117;
        border-right: 1px solid #1e2130;
    }
    [data-testid="stSidebar"] * { color: #e0e4f0 !important; }

    /* Hide default Streamlit nav labels */
    [data-testid="stSidebarNav"] { display: none; }

    /* Top bar */
    header[data-testid="stHeader"] { background: #0f1117; }

    /* Main background */
    .main .block-container { padding-top: 2rem; padding-bottom: 3rem; }

    /* Cards */
    .nectar-card {
        background: #1a1d2e;
        border: 1px solid #2a2d3e;
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }

    /* Metric cards */
    .metric-card {
        background: #12151f;
        border: 1px solid #2a2d3e;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #7c9ef5; }
    .metric-label { font-size: 0.75rem; color: #8890a8; letter-spacing: 0.08em; text-transform: uppercase; }

    /* Affinity bar */
    .affinity-bar-bg {
        background: #1e2130;
        border-radius: 4px;
        height: 6px;
        width: 100%;
    }
    .affinity-bar-fill {
        background: linear-gradient(90deg, #7c9ef5, #a78bfa);
        border-radius: 4px;
        height: 6px;
    }

    /* Chunk card */
    .chunk-card {
        background: #12151f;
        border: 1px solid #2a2d3e;
        border-left: 3px solid #7c9ef5;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        font-size: 0.88rem;
        line-height: 1.6;
        color: #c8cfe0;
    }
    .chunk-meta {
        font-size: 0.75rem;
        color: #8890a8;
        margin-bottom: 0.5rem;
        letter-spacing: 0.03em;
    }

    /* Status badge */
    .badge-ok      { background:#1a3a2a; color:#4ade80; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
    .badge-skip    { background:#2a2a1a; color:#facc15; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
    .badge-error   { background:#3a1a1a; color:#f87171; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
    .badge-indexed { background:#1a2a3a; color:#60a5fa; border-radius:4px; padding:2px 8px; font-size:0.75rem; }

    /* Divider */
    hr { border-color: #1e2130; }

    /* Answer box */
    .answer-box {
        background: #12151f;
        border: 1px solid #2a2d3e;
        border-radius: 8px;
        padding: 1.4rem 1.6rem;
        font-size: 0.95rem;
        line-height: 1.75;
        color: #dce3f5;
        white-space: pre-wrap;
    }

    /* Section label */
    .section-label {
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #7c9ef5;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    /* Table override */
    [data-testid="stDataFrame"] { border: 1px solid #2a2d3e; border-radius: 8px; }

    /* Buttons */
    .stButton > button {
        border-radius: 6px;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 1.5rem 0;">
        <div style="font-size:1.6rem; font-weight:800; color:#7c9ef5; letter-spacing:-0.02em;">🍯 Nectar</div>
        <div style="font-size:0.72rem; color:#8890a8; letter-spacing:0.1em; text-transform:uppercase; margin-top:2px;">
            Scientific RAG Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Navigation</div>', unsafe_allow_html=True)

    pages = {
        "📥  Knowledge Base":   "pages/1_Knowledge_Base.py",
        "🔍  Query":            "pages/2_Query.py",
        "🗂️  Explorer":         "pages/3_Explorer.py",
        "🔭  Vector Space":     "pages/4_Vector_Space.py",
    }
    for label in pages:
        st.page_link(pages[label], label=label)

    st.markdown("---")
    st.markdown('<div class="section-label">System</div>', unsafe_allow_html=True)

    # Quick DB status in sidebar
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from core.backend import row_count, get_db_stats
        stats = get_db_stats()
        st.markdown(
            f"<div style='font-size:0.8rem; color:#8890a8;'>"
            f"<b style='color:#e0e4f0;'>{stats['total_papers']}</b> papers &nbsp;·&nbsp; "
            f"<b style='color:#e0e4f0;'>{stats['total_chunks']:,}</b> chunks"
            f"</div>",
            unsafe_allow_html=True,
        )
        index_label = "ANN index active" if stats["has_index"] else "No index (exact scan)"
        index_color = "#4ade80" if stats["has_index"] else "#facc15"
        st.markdown(
            f"<div style='font-size:0.75rem; margin-top:4px; color:{index_color};'>⬤ {index_label}</div>",
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.7rem; color:#555e78; line-height:1.6;'>"
        "LLM: Qwen2.5-0.5B<br>"
        "Embeddings: mxbai-embed-large-v1<br>"
        "VectorDB: LanceDB · Lance format<br>"
        "Schema: PyArrow typed"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Home content ─────────────────────────────────────────────
st.markdown("## Welcome to Nectar")
st.markdown(
    "Nectar is a fully local Retrieval-Augmented Generation platform for scientific literature. "
    "All processing runs on your machine — no API keys, no cloud, no data leaving your environment."
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="nectar-card">
        <div style="font-size:1.4rem; margin-bottom:0.5rem;">📥</div>
        <div style="font-weight:700; color:#e0e4f0; margin-bottom:0.4rem;">Knowledge Base</div>
        <div style="font-size:0.85rem; color:#8890a8; line-height:1.5;">
            Upload PDFs, embed them into a persistent LanceDB vector store, 
            build an IVF_PQ index, and monitor corpus statistics.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="nectar-card">
        <div style="font-size:1.4rem; margin-bottom:0.5rem;">🔍</div>
        <div style="font-weight:700; color:#e0e4f0; margin-bottom:0.4rem;">Query</div>
        <div style="font-size:0.85rem; color:#8890a8; line-height:1.5;">
            Ask questions across your entire corpus in natural language. 
            Inspect retrieved chunks, affinity scores, and source attribution.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="nectar-card">
        <div style="font-size:1.4rem; margin-bottom:0.5rem;">🗂️</div>
        <div style="font-weight:700; color:#e0e4f0; margin-bottom:0.4rem;">Explorer</div>
        <div style="font-size:0.85rem; color:#8890a8; line-height:1.5;">
            Browse the vector store directly. Filter by paper, page range, 
            or keyword to inspect raw chunks without running a query.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<div style='font-size:0.8rem; color:#555e78;'>"
    "Use the navigation panel on the left to switch between pages."
    "</div>",
    unsafe_allow_html=True,
)