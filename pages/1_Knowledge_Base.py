"""
Page 1 — Knowledge Base
Upload PDFs · Ingest into LanceDB · Build IVF_PQ Index · Database Statistics
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from core.backend import (
    ingest_documents, get_db_stats, clear_database,
    build_index, METRIC_INFO, INDEX_MIN_ROWS, row_count, IngestConfig,
)
from core.loader import SUPPORTED_EXTENSIONS, EXTENSION_LABELS

st.set_page_config(page_title="Knowledge Base — Nectar", page_icon="📥", layout="wide")

# ── Shared CSS (re-injected per page) ────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"]   { background:#0f1117; border-right:1px solid #1e2130; }
[data-testid="stSidebar"] * { color:#e0e4f0 !important; }
header[data-testid="stHeader"] { background:#0f1117; }
.main .block-container { padding-top:2rem; padding-bottom:3rem; }
.nectar-card  { background:#1a1d2e; border:1px solid #2a2d3e; border-radius:10px; padding:1.4rem 1.6rem; margin-bottom:1rem; }
.metric-card  { background:#12151f; border:1px solid #2a2d3e; border-radius:8px; padding:1rem 1.2rem; text-align:center; }
.metric-value { font-size:2rem; font-weight:700; color:#7c9ef5; }
.metric-label { font-size:0.7rem; color:#8890a8; letter-spacing:0.08em; text-transform:uppercase; }
.badge-ok     { background:#1a3a2a; color:#4ade80; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
.badge-skip   { background:#2a2a1a; color:#facc15; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
.badge-error  { background:#3a1a1a; color:#f87171; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
.section-label { font-size:0.7rem; letter-spacing:0.12em; text-transform:uppercase; color:#7c9ef5; font-weight:600; margin-bottom:0.5rem; }
hr { border-color:#1e2130; }
.stButton > button { border-radius:6px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────
st.markdown("## 📥 Knowledge Base")
st.markdown(
    "<div style='color:#8890a8; font-size:0.9rem; margin-bottom:0.8rem;'>"
    "Upload documents in any supported format. Every file is converted to Markdown, "
    "cleaned, chunked and embedded into the persistent vector store."
    "</div>",
    unsafe_allow_html=True,
)

# Supported formats pill list
_fmt_groups = {
    "Text / Logs":    [".txt", ".log", ".md"],
    "Office":         [".pdf", ".docx", ".doc", ".odt", ".rtf"],
    "Web / Markup":   [".html", ".htm", ".xml", ".rst", ".tex"],
    "Data":           [".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml"],
    "E-book":         [".epub"],
    "Source Code":    [".py", ".js", ".c", ".cpp", ".sh"],
}
pills_html = "<div style='display:flex; flex-wrap:wrap; gap:0.4rem; margin-bottom:1.5rem;'>"
for group, exts in _fmt_groups.items():
    pills_html += (
        f"<span style='font-size:0.65rem; color:#8890a8; margin-right:0.2rem; "
        f"align-self:center;'>{group}:</span>"
    )
    for e in exts:
        pills_html += (
            f"<span style='background:#1a1d2e; border:1px solid #2a2d3e; color:#7c9ef5; "
            f"border-radius:4px; padding:1px 7px; font-size:0.68rem; font-family:monospace;'>"
            f"{e}</span>"
        )
pills_html += "</div>"
st.markdown(pills_html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SECTION 1 — Upload & Ingest
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Upload & Ingest</div>', unsafe_allow_html=True)

with st.container():
    # Build accepted type list from SUPPORTED_EXTENSIONS (strip leading dot)
    _accepted = sorted(e.lstrip(".") for e in SUPPORTED_EXTENSIONS)
    uploaded_files = st.file_uploader(
        "Select one or more files",
        type=_accepted,
        accept_multiple_files=True,
        help=(
            "Any supported format is accepted. "
            "Each file is deduplicated by path hash — re-uploading the same file is safe."
        ),
        label_visibility="collapsed",
    )

    # ── Processing Pipeline ───────────────────────────────────
    with st.expander("⚙  Processing Pipeline", expanded=False):
        st.markdown(
            "<div style='font-size:0.8rem; color:#8890a8; margin-bottom:1rem;'>"
            "Configure how PDFs are cleaned, split and filtered before embedding. "
            "Defaults work well for most scientific papers."
            "</div>",
            unsafe_allow_html=True,
        )

        tab_pre, tab_split, tab_post = st.tabs(["1 · Pre-processing", "2 · Splitting", "3 · Post-filtering"])

        with tab_pre:
            st.markdown(
                "<div style='font-size:0.78rem; color:#8890a8; margin-bottom:0.8rem;'>"
                "Applied to each page <b style='color:#c8cfe0;'>before</b> splitting."
                "</div>", unsafe_allow_html=True,
            )
            col_a, col_b = st.columns(2)
            with col_a:
                cfg_normalize = st.toggle(
                    "Normalize characters",
                    value=True,
                    help="Converts typographic ligatures (ﬁ→fi, ﬂ→fl), soft hyphens, "
                         "non-breaking spaces and bullet variants to standard ASCII.",
                )
                cfg_rm_hf = st.toggle(
                    "Remove headers / footers",
                    value=True,
                    help="Strips short repeated lines at the top and bottom of each page "
                         "(page numbers, chapter titles, running headers).",
                )
            with col_b:
                cfg_hyphen = st.toggle(
                    "Merge hyphenated line breaks",
                    value=True,
                    help='Re-joins words broken by column width: "collo-\\ncazione" → "collocazione".',
                )
                cfg_softbreak = st.toggle(
                    "Merge soft line breaks",
                    value=True,
                    help="Converts single newlines (column wrap) to spaces while preserving "
                         "double newlines (paragraph boundaries).",
                )

        with tab_split:
            st.markdown(
                "<div style='font-size:0.78rem; color:#8890a8; margin-bottom:0.8rem;'>"
                "RecursiveCharacterTextSplitter — splits on paragraphs → sentences → words."
                "</div>", unsafe_allow_html=True,
            )
            col_c, col_d = st.columns(2)
            with col_c:
                cfg_chunk_size = st.slider(
                    "Chunk size (characters)",
                    min_value=200, max_value=4000, value=1000, step=100,
                    help="Maximum characters per chunk. Larger chunks give more context "
                         "but reduce retrieval precision.",
                )
            with col_d:
                cfg_overlap = st.slider(
                    "Chunk overlap (characters)",
                    min_value=0, max_value=500, value=100, step=10,
                    help="Characters shared between consecutive chunks. Prevents information "
                         "from being cut at a boundary.",
                )
            overlap_pct = round(cfg_overlap / cfg_chunk_size * 100) if cfg_chunk_size else 0
            st.markdown(
                f"<div style='font-size:0.75rem; color:#8890a8;'>"
                f"Overlap is <b style='color:#e0e4f0;'>{overlap_pct}%</b> of chunk size."
                f"</div>",
                unsafe_allow_html=True,
            )

        with tab_post:
            st.markdown(
                "<div style='font-size:0.78rem; color:#8890a8; margin-bottom:0.8rem;'>"
                "Applied <b style='color:#c8cfe0;'>after</b> splitting, before embedding."
                "</div>", unsafe_allow_html=True,
            )
            col_e, col_f, col_g = st.columns(3)
            with col_e:
                cfg_min_chars = st.slider(
                    "Min chunk length (chars)",
                    min_value=0, max_value=300, value=80, step=10,
                    help="Chunks shorter than this are discarded — typically page numbers, "
                         "isolated titles or index remnants.",
                )
            with col_f:
                cfg_sw_ratio = st.slider(
                    "Max stop-word ratio",
                    min_value=0, max_value=100, value=0, step=5, format="%d%%",
                    help="Discard chunks where stop-words exceed this share of total words. "
                         "0 = disabled. ~80% catches table-of-contents and formatting noise.",
                )
            with col_g:
                cfg_dedup = st.slider(
                    "Dedup threshold (Jaccard)",
                    min_value=0, max_value=100, value=0, step=5, format="%d%%",
                    help="Remove chunks whose word-set overlap with a previous chunk exceeds "
                         "this threshold. 0 = disabled. ~90% removes near-identical paragraphs "
                         "(disclaimers, repeated footnotes).",
                )

    # Build the IngestConfig from UI values
    ingest_config = IngestConfig(
        normalize_chars=cfg_normalize,
        remove_headers_footers=cfg_rm_hf,
        merge_hyphen_breaks=cfg_hyphen,
        merge_soft_breaks=cfg_softbreak,
        chunk_size=cfg_chunk_size,
        chunk_overlap=cfg_overlap,
        min_chunk_chars=cfg_min_chars,
        max_stopword_ratio=cfg_sw_ratio / 100.0,
        dedup_threshold=cfg_dedup / 100.0,
    )

    col_btn, col_warn = st.columns([2, 5])
    with col_btn:
        ingest_clicked = st.button("⚡  Embed & Add to Database", type="primary", use_container_width=True)

if ingest_clicked:
    if not uploaded_files:
        st.warning("No files selected. Please upload at least one PDF.")
    else:
        # Save to temp dir, then ingest
        import tempfile, shutil
        tmp_paths = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for uf in uploaded_files:
                dest = os.path.join(tmpdir, uf.name)
                with open(dest, "wb") as f:
                    f.write(uf.read())
                tmp_paths.append(dest)

            with st.spinner(f"Embedding {len(tmp_paths)} file(s)…"):
                results = ingest_documents(tmp_paths, ingest_config)

        # Results table
        st.markdown("---")
        st.markdown('<div class="section-label">Ingestion Results</div>', unsafe_allow_html=True)

        import html as _html
        for r in results:
            if r["skipped"]:
                badge  = '<span class="badge-skip">SKIPPED</span>'
                detail = "Already indexed — no re-embedding needed."
            elif r["status"] == "ok":
                badge  = '<span class="badge-ok">INDEXED</span>'
                detail = f"{r['chunks']} chunks embedded."
            elif r["status"] == "empty":
                badge  = '<span class="badge-error">EMPTY</span>'
                detail = "No usable text after cleaning and filtering."
            else:
                badge  = '<span class="badge-error">ERROR</span>'
                detail = _html.escape(r.get("error", "Unknown error."))

            fmt_pill = (
                f"<span style='background:#1a1d2e; border:1px solid #2a2d3e; color:#a78bfa; "
                f"border-radius:4px; padding:1px 6px; font-size:0.66rem; font-family:monospace;'>"
                f"{_html.escape(r.get('format','?'))}</span>"
            )
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:10px; padding:0.5rem 0; "
                f"border-bottom:1px solid #1e2130; font-size:0.88rem;'>"
                f"{badge} {fmt_pill} "
                f"<span style='color:#c8cfe0; font-weight:500;'>{_html.escape(r['name'])}</span>"
                f"<span style='color:#8890a8; margin-left:auto;'>{detail}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════
# SECTION 2 — IVF_PQ Index Builder
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">IVF_PQ Index Builder</div>', unsafe_allow_html=True)
st.markdown(
    "<div style='color:#8890a8; font-size:0.85rem; margin-bottom:1rem;'>"
    "An IVF_PQ (Inverted File + Product Quantization) index enables Approximate Nearest Neighbor (ANN) search — "
    "significantly faster than exact scan for large corpora. "
    f"Requires at least <b style='color:#e0e4f0;'>{INDEX_MIN_ROWS} chunks</b> to train."
    "</div>",
    unsafe_allow_html=True,
)

col_metric, col_info = st.columns([1, 2])

with col_metric:
    metric_labels = {
        "cosine": "📐  Cosine Distance  ✦ recommended",
        "l2":     "📏  L2 Euclidean Distance",
        "dot":    "⚡  Dot Product",
    }
    selected_metric = st.selectbox(
        "Distance metric",
        options=list(metric_labels.keys()),
        format_func=lambda k: metric_labels[k],
        index=0,
        help="Select the metric used to build the ANN index. Must match your retrieval expectations.",
    )
    build_clicked = st.button("🔨  Build Index", type="primary", use_container_width=True)

with col_info:
    m = METRIC_INFO[selected_metric]
    st.markdown(
        f"<div class='nectar-card'>"
        f"<div style='font-size:0.75rem; color:#7c9ef5; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.4rem;'>"
        f"{m['label']}</div>"
        f"<div style='font-family:monospace; font-size:0.8rem; color:#a78bfa; margin-bottom:0.6rem;'>"
        f"f(a,b) = {m['formula']}</div>"
        f"<div style='font-size:0.82rem; color:#c8cfe0; line-height:1.6; margin-bottom:0.6rem;'>"
        f"{m['description']}</div>"
        f"<div style='font-size:0.75rem; color:#8890a8;'>"
        f"<b style='color:#e0e4f0;'>Range:</b> {m['range']}<br>"
        f"<b style='color:#e0e4f0;'>Best for:</b> {m['use_case']}"
        f"</div>"
        f"{'<div style=\"margin-top:0.6rem;\"><span class=\"badge-ok\">✦ recommended for this setup</span></div>' if m['recommended'] else ''}"
        f"</div>",
        unsafe_allow_html=True,
    )

if build_clicked:
    n = row_count()
    if n < INDEX_MIN_ROWS:
        st.error(
            f"**Insufficient data.** The index requires at least **{INDEX_MIN_ROWS} chunks** to train. "
            f"Current count: **{n}**. Add more papers and try again."
        )
    else:
        with st.spinner("Building IVF_PQ index…"):
            result = build_index(selected_metric)
        if result["ok"]:
            st.success(result["message"])
            p = result["params"]
            st.markdown(
                f"<div class='nectar-card' style='margin-top:0.5rem;'>"
                f"<div style='display:grid; grid-template-columns:repeat(4,1fr); gap:1rem;'>"
                f"<div class='metric-card'><div class='metric-value'>{p['metric']}</div><div class='metric-label'>Metric</div></div>"
                f"<div class='metric-card'><div class='metric-value'>{p['num_partitions']}</div><div class='metric-label'>IVF Partitions</div></div>"
                f"<div class='metric-card'><div class='metric-value'>{p['num_sub_vectors']}</div><div class='metric-label'>PQ Sub-vectors</div></div>"
                f"<div class='metric-card'><div class='metric-value'>{p['indexed_chunks']:,}</div><div class='metric-label'>Indexed Chunks</div></div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.error(result["message"])

st.markdown("---")

# ═══════════════════════════════════════════════════════════════
# SECTION 3 — Database Statistics
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Database Statistics</div>', unsafe_allow_html=True)

refresh_col, clear_col, _ = st.columns([1, 1, 5])
with refresh_col:
    refresh = st.button("↻  Refresh", use_container_width=True)
with clear_col:
    clear_clicked = st.button("🗑  Clear Database", type="secondary", use_container_width=True)

if clear_clicked:
    clear_database()
    st.success("Database cleared.")
    st.rerun()

stats = get_db_stats()

# KPI row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-value'>{stats['total_papers']}</div>"
        f"<div class='metric-label'>Papers</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
with kpi2:
    st.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-value'>{stats['total_chunks']:,}</div>"
        f"<div class='metric-label'>Total Chunks</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
with kpi3:
    avg = round(stats["total_chunks"] / stats["total_papers"], 1) if stats["total_papers"] else 0
    st.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-value'>{avg}</div>"
        f"<div class='metric-label'>Avg Chunks / Paper</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
with kpi4:
    idx_label = "Active" if stats["has_index"] else "None"
    idx_color = "#4ade80" if stats["has_index"] else "#facc15"
    st.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-value' style='color:{idx_color}; font-size:1.4rem;'>{idx_label}</div>"
        f"<div class='metric-label'>ANN Index</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# Per-paper table
if stats["papers"]:
    st.markdown("<div style='margin-top:1.2rem;'>", unsafe_allow_html=True)
    import pandas as pd
    df_display = pd.DataFrame(stats["papers"])[["paper_name", "chunks", "pages"]]
    df_display.columns = ["Paper", "Chunks", "Unique Pages"]
    df_display = df_display.sort_values("Chunks", ascending=False).reset_index(drop=True)
    df_display.index += 1
    st.dataframe(df_display, use_container_width=True, height=min(40 * len(df_display) + 60, 420))
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("No papers in the database yet. Upload PDFs above to get started.")