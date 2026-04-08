"""
Page 3 — Explorer
Browse the vector store directly · Filter by paper, page range, keyword
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
from core.backend import explore_database, get_paper_names, get_db_stats, row_count

st.set_page_config(page_title="Explorer — Nectar", page_icon="🗂️", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"]   { background:#0f1117; border-right:1px solid #1e2130; }
[data-testid="stSidebar"] * { color:#e0e4f0 !important; }
header[data-testid="stHeader"] { background:#0f1117; }
.main .block-container { padding-top:2rem; padding-bottom:3rem; }
.nectar-card   { background:#1a1d2e; border:1px solid #2a2d3e; border-radius:10px; padding:1.4rem 1.6rem; margin-bottom:1rem; }
.chunk-card    { background:#12151f; border:1px solid #2a2d3e; border-left:3px solid #a78bfa; border-radius:6px; padding:0.9rem 1.1rem; margin-bottom:0.6rem; font-size:0.86rem; line-height:1.6; color:#c8cfe0; }
.chunk-meta    { font-size:0.72rem; color:#8890a8; margin-bottom:0.4rem; }
.section-label { font-size:0.7rem; letter-spacing:0.12em; text-transform:uppercase; color:#7c9ef5; font-weight:600; margin-bottom:0.5rem; }
.metric-card   { background:#12151f; border:1px solid #2a2d3e; border-radius:8px; padding:0.9rem 1rem; text-align:center; }
.metric-value  { font-size:1.8rem; font-weight:700; color:#7c9ef5; }
.metric-label  { font-size:0.7rem; color:#8890a8; letter-spacing:0.08em; text-transform:uppercase; }
hr { border-color:#1e2130; }
.stButton > button { border-radius:6px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────
st.markdown("## 🗂️ Explorer")
st.markdown(
    "<div style='color:#8890a8; font-size:0.9rem; margin-bottom:1.5rem;'>"
    "Browse the vector store directly. Apply filters to inspect raw chunks "
    "without running a retrieval query."
    "</div>",
    unsafe_allow_html=True,
)

if row_count() == 0:
    st.warning("The database is empty. Go to **Knowledge Base** and ingest at least one PDF.")
    st.stop()

# ════════════════════════════════════════════════════════════════
# SECTION 1 — Corpus overview (compact KPIs)
# ════════════════════════════════════════════════════════════════
stats = get_db_stats()

st.markdown('<div class="section-label">Corpus Overview</div>', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{stats['total_papers']}</div><div class='metric-label'>Papers</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{stats['total_chunks']:,}</div><div class='metric-label'>Chunks</div></div>", unsafe_allow_html=True)
with k3:
    avg = round(stats['total_chunks'] / stats['total_papers'], 1) if stats['total_papers'] else 0
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{avg}</div><div class='metric-label'>Avg / Paper</div></div>", unsafe_allow_html=True)
with k4:
    idx_label = "Active" if stats["has_index"] else "None"
    idx_color = "#4ade80" if stats["has_index"] else "#facc15"
    st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:{idx_color}; font-size:1.3rem;'>{idx_label}</div><div class='metric-label'>ANN Index</div></div>", unsafe_allow_html=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# SECTION 2 — Filters
# ════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Filters</div>', unsafe_allow_html=True)

all_papers = get_paper_names()

col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

with col1:
    paper_options = ["— All papers —"] + all_papers
    selected_paper = st.selectbox(
        "Paper",
        options=paper_options,
        index=0,
        help="Filter results to a specific paper.",
    )
    paper_filter = "" if selected_paper == "— All papers —" else selected_paper

with col2:
    page_min = st.number_input(
        "Page from",
        min_value=0,
        max_value=9999,
        value=0,
        step=1,
        help="Minimum page number (inclusive).",
    )

with col3:
    page_max = st.number_input(
        "Page to",
        min_value=0,
        max_value=9999,
        value=9999,
        step=1,
        help="Maximum page number (inclusive).",
    )

with col4:
    result_limit = st.selectbox(
        "Max results",
        options=[25, 50, 100, 250, 500],
        index=1,
        help="Maximum number of chunks to display.",
    )

keyword = st.text_input(
    "Keyword filter",
    placeholder="Search within chunk text (case-insensitive substring match)…",
    help="Filters chunks whose text contains this string. Not a semantic search — use the Query page for that.",
)

apply_filters = st.button("  Apply Filters  ", type="primary")

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# SECTION 3 — Results
# ════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)

# Run on page load with defaults, or when button clicked
results = explore_database(
    paper_filter=paper_filter,
    page_min=int(page_min),
    page_max=int(page_max),
    keyword=keyword.strip(),
    limit=int(result_limit),
)

if not results:
    st.info("No chunks match the current filters.")
else:
    result_count = len(results)
    st.markdown(
        f"<div style='font-size:0.82rem; color:#8890a8; margin-bottom:1rem;'>"
        f"Showing <b style='color:#e0e4f0;'>{result_count}</b> chunk(s)"
        + (f" — limit reached, narrow your filters to see more." if result_count == result_limit else ".")
        + "</div>",
        unsafe_allow_html=True,
    )

    # View toggle: cards vs table
    view_mode = st.radio(
        "Display as",
        options=["Cards", "Table"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if view_mode == "Table":
        df = pd.DataFrame(results)[["paper_name", "page", "chunk"]]
        df.columns = ["Paper", "Page", "Chunk"]
        df.index += 1
        # Truncate long chunks for table display
        df["Chunk"] = df["Chunk"].str[:300] + df["Chunk"].apply(lambda x: "…" if len(x) > 300 else "")
        st.dataframe(df, use_container_width=True, height=min(40 * len(df) + 60, 600))

    else:
        # Card view — grouped by paper for readability
        current_paper = None
        for row in results:
            if row["paper_name"] != current_paper:
                current_paper = row["paper_name"]
                st.markdown(
                    f"<div style='font-size:0.75rem; color:#7c9ef5; letter-spacing:0.08em; "
                    f"text-transform:uppercase; margin-top:1.2rem; margin-bottom:0.4rem; font-weight:600;'>"
                    f"📄 {current_paper}</div>",
                    unsafe_allow_html=True,
                )

            preview = row["chunk"][:500] + ("…" if len(row["chunk"]) > 500 else "")
            # Highlight keyword if present
            if keyword.strip():
                import re
                preview = re.sub(
                    f"({re.escape(keyword.strip())})",
                    r"<mark style='background:#2a3a1a; color:#86efac; border-radius:2px; padding:0 2px;'>\1</mark>",
                    preview,
                    flags=re.IGNORECASE,
                )

            st.markdown(
                f"<div class='chunk-card'>"
                f"<div class='chunk-meta'>Page {row['page']}</div>"
                f"{preview}"
                f"</div>",
                unsafe_allow_html=True,
            )