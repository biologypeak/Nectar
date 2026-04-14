"""
Nectar v4 — Vector Space Explorer
Interactive 2D/3D visualisation of the embedding space with:
  - Dimensionality reduction: PaCMAP · UMAP · TriMap · t-SNE
  - Clustering: HDBSCAN · Leiden · K-Means · Spectral · Agglomerative
  - Disk cache: projections & labels persisted across sessions
  - Real-time query projection (PaCMAP / UMAP / t-SNE)
  - Sub-clustering: replace view with the cluster where the query ★ landed
"""

import hashlib
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from core.backend import get_embedding_model, load_all_vectors, row_count

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CACHE_DIR = Path("./nectar_cache")
CACHE_DIR.mkdir(exist_ok=True)

PALETTE = [
    "#7c9ef5", "#a78bfa", "#34d399", "#f87171", "#fbbf24",
    "#38bdf8", "#e879f9", "#a3e635", "#fb923c", "#94a3b8",
    "#f472b6", "#2dd4bf", "#c084fc", "#fb7185", "#86efac",
]


# ─────────────────────────────────────────────────────────────────────────────
# DISK CACHE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _proj_cache_key(method: str, params: dict, n_rows: int, n_components: int) -> str:
    raw = f"proj|{method}|{sorted(params.items())}|{n_rows}|{n_components}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"vs_{key}.pkl"


def _load_cache(key: str):
    p = _cache_path(key)
    if p.exists():
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            p.unlink(missing_ok=True)
    return None


def _save_cache(key: str, data: dict) -> None:
    with open(_cache_path(key), "wb") as f:
        pickle.dump(data, f)


# ─────────────────────────────────────────────────────────────────────────────
# DIMENSIONALITY REDUCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_reducer(
    method: str,
    vectors: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, object]:
    """
    Returns (projection, fitted_reducer_or_None).
    fitted_reducer supports .transform() for PaCMAP, UMAP and openTSNE.
    """
    if method == "PaCMAP":
        import pacmap
        reducer = pacmap.PaCMAP(n_components=n_components, random_state=42)
        proj = reducer.fit_transform(vectors, init="pca")
        return proj.astype(np.float32), reducer

    if method == "UMAP":
        import umap
        reducer = umap.UMAP(n_components=n_components, random_state=42, verbose=False)
        proj = reducer.fit_transform(vectors)
        return proj.astype(np.float32), reducer

    if method == "TriMap":
        import trimap
        proj = trimap.TRIMAP(n_dims=n_components).fit_transform(vectors)
        return proj.astype(np.float32), None   # no out-of-sample support

    if method == "t-SNE":
        from openTSNE import TSNE
        tsne = TSNE(n_components=n_components, random_state=42, verbose=False)
        embedding = tsne.fit(vectors)
        return np.array(embedding, dtype=np.float32), embedding  # openTSNE supports transform

    raise ValueError(f"Unknown reducer: {method}")


def project_query(fitted_reducer, method: str, query_vec: np.ndarray) -> np.ndarray | None:
    """Project a single (EMBED_DIM,) query vector into the already-fitted space."""
    if fitted_reducer is None:
        return None
    v = query_vec.reshape(1, -1)
    try:
        if method in ("PaCMAP", "UMAP"):
            return np.array(fitted_reducer.transform(v), dtype=np.float32)
        if method == "t-SNE":
            return np.array(fitted_reducer.transform(v), dtype=np.float32)
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def run_clusterer(method: str, projection: np.ndarray, params: dict) -> np.ndarray:
    """Returns integer label array aligned with projection rows. Noise = -1."""

    if method == "HDBSCAN":
        from hdbscan import HDBSCAN as _HDBSCAN
        return _HDBSCAN(**params).fit_predict(projection).astype(int)

    if method == "Leiden":
        import igraph as ig
        import leidenalg
        from pynndescent import NNDescent
        k          = min(params.get("n_neighbors", 15), len(projection) - 1)
        resolution = params.get("resolution", 1.0)
        index      = NNDescent(projection, n_neighbors=k, random_state=42)
        indices, _ = index.neighbor_graph
        n     = len(projection)
        edges = [
            (int(i), int(j))
            for i, nbrs in enumerate(indices)
            for j in nbrs
            if int(j) != i
        ]
        g         = ig.Graph(n=n, edges=edges, directed=False)
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            seed=42,
        )
        return np.array(partition.membership, dtype=int)

    if method == "K-Means":
        from sklearn.cluster import KMeans
        return KMeans(**params, n_init="auto", random_state=42).fit_predict(projection).astype(int)

    if method == "Spectral":
        from sklearn.cluster import SpectralClustering
        return SpectralClustering(**params, random_state=42).fit_predict(projection).astype(int)

    if method == "Agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        return AgglomerativeClustering(**params).fit_predict(projection).astype(int)

    return np.zeros(len(projection), dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def _colors_for(color_by: str, labels: np.ndarray, meta_df) -> list:
    if color_by == "cluster_label":
        unique = sorted(set(labels))
        cmap   = {l: PALETTE[i % len(PALETTE)] for i, l in enumerate(unique)}
        return [cmap[l] for l in labels]

    if color_by == "paper_name":
        unique = sorted(meta_df["paper_name"].unique())
        cmap   = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(unique)}
        return [cmap[p] for p in meta_df["paper_name"]]

    if color_by == "page":
        pages = meta_df["page"].values.astype(float)
        rng   = pages.max() - pages.min()
        norm  = (pages - pages.min()) / (rng if rng > 0 else 1)
        return [
            f"rgb({int(55 + n * 180)},{int(130 - n * 80)},{int(255 - n * 200)})"
            for n in norm
        ]

    return [PALETTE[0]] * len(labels)


def _hover_texts(meta_df) -> list[str]:
    return [
        (
            f"<b>{row.paper_name}</b> &nbsp;·&nbsp; p.{row.page}<br>"
            f"<span style='font-size:11px;color:#c8cfe0'>"
            f"{str(row.chunk)[:220].replace('<','&lt;').replace('>','&gt;')}"
            f"{'…' if len(str(row.chunk)) > 220 else ''}</span>"
        )
        for row in meta_df.itertuples()
    ]


def build_figure(
    projection: np.ndarray,
    labels: np.ndarray,
    meta_df,
    color_by: str,
    use_3d: bool,
    query_pt: np.ndarray | None,
    highlighted_cluster: int | None,
) -> go.Figure:

    is_3d   = use_3d and projection.shape[1] == 3
    colors  = _colors_for(color_by, labels, meta_df)
    hovers  = _hover_texts(meta_df)

    # Dim noise points
    opacities = [
        0.25 if l == -1 else (0.9 if highlighted_cluster is None or l == highlighted_cluster else 0.25)
        for l in labels
    ]

    fig = go.Figure()

    marker_common = dict(
        color=colors,
        opacity=opacities,
        line=dict(width=0),
    )

    if is_3d:
        fig.add_trace(go.Scatter3d(
            x=projection[:, 0],
            y=projection[:, 1],
            z=projection[:, 2],
            mode="markers",
            marker=dict(size=3, **marker_common),
            text=hovers,
            hovertemplate="%{text}<extra></extra>",
            customdata=list(range(len(projection))),
            name="corpus",
        ))
        if query_pt is not None and query_pt.shape[1] == 3:
            fig.add_trace(go.Scatter3d(
                x=[query_pt[0, 0]], y=[query_pt[0, 1]], z=[query_pt[0, 2]],
                mode="markers+text",
                marker=dict(size=10, color="#facc15", symbol="diamond"),
                text=["★"],
                textposition="top center",
                textfont=dict(size=16, color="#facc15"),
                name="query",
                hovertemplate="Query point<extra></extra>",
            ))
    else:
        # Scattergl = WebGL — handles 20k+ points smoothly
        fig.add_trace(go.Scattergl(
            x=projection[:, 0],
            y=projection[:, 1],
            mode="markers",
            marker=dict(size=4, **marker_common),
            text=hovers,
            hovertemplate="%{text}<extra></extra>",
            customdata=list(range(len(projection))),
            name="corpus",
        ))
        if query_pt is not None:
            fig.add_trace(go.Scatter(
                x=[query_pt[0, 0]], y=[query_pt[0, 1]],
                mode="markers+text",
                marker=dict(size=16, color="#facc15", symbol="star"),
                text=["★ query"],
                textposition="top center",
                textfont=dict(size=13, color="#facc15"),
                name="query",
                hovertemplate="Query point<extra></extra>",
            ))

    fig.update_layout(
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="#e0e4f0"),
        margin=dict(l=0, r=0, t=24, b=0),
        height=640,
        showlegend=False,
        hoverlabel=dict(
            bgcolor="#1a1d2e",
            bordercolor="#2a2d3e",
            font=dict(color="#e0e4f0", size=12),
            align="left",
        ),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

_SS_DEFAULTS = {
    "vs_mode":             "full",   # "full" | "sub"
    "vs_sub_mask":         None,     # bool ndarray over full corpus
    "vs_projection":       None,     # current projected coords
    "vs_labels":           None,     # current cluster labels
    "vs_fitted_reducer":   None,     # fitted reducer for transform()
    "vs_reducer_method":   None,     # which method was fitted
    "vs_query_pt":         None,     # projected query coords (1, dims)
    "vs_query_cluster":    None,     # cluster label where query landed
    "vs_clicked_chunk":    None,     # dict: paper_name, page, chunk
}

for k, v in _SS_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _cached_load(n_rows_hint: int):
    """Cache keyed by row count — auto-invalidates on new ingests."""
    return load_all_vectors()


total_rows = row_count()
if total_rows == 0:
    st.warning("Database is empty. Ingest PDFs on the Knowledge Base page first.")
    st.stop()

with st.spinner("Loading vectors…"):
    _all_vectors, _all_meta = _cached_load(total_rows)

# ─────────────────────────────────────────────────────────────────────────────
# ACTIVE SLICE (full corpus or sub-cluster)
# ─────────────────────────────────────────────────────────────────────────────

mode = st.session_state["vs_mode"]

if mode == "sub" and st.session_state["vs_sub_mask"] is not None:
    _mask        = st.session_state["vs_sub_mask"]
    active_vecs  = _all_vectors[_mask]
    active_meta  = _all_meta[_mask].reset_index(drop=True)
    _slice_label = f"Sub-cluster · {_mask.sum():,} chunks"
else:
    active_vecs  = _all_vectors
    active_meta  = _all_meta
    _slice_label = f"Full corpus · {total_rows:,} chunks"


# ─────────────────────────────────────────────────────────────────────────────
# CONTROLS — TOP BAR
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    f"<div style='font-size:0.72rem;color:#7c9ef5;letter-spacing:.1em;"
    f"text-transform:uppercase;font-weight:600;margin-bottom:.6rem'>"
    f"Vector Space Explorer &nbsp;·&nbsp; {_slice_label}</div>",
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5 = st.columns([2, 1.4, 2, 1, 1])

with c1:
    reducer_choice = st.selectbox(
        "Reduction method",
        ["PaCMAP", "UMAP", "TriMap", "t-SNE"],
        key="vs_sel_reducer",
    )
with c2:
    use_3d = st.toggle("3D", value=False, key="vs_3d")
    n_components = 3 if use_3d else 2

with c3:
    cluster_choice = st.selectbox(
        "Clustering",
        ["HDBSCAN", "Leiden", "K-Means", "Spectral", "Agglomerative"],
        key="vs_sel_cluster",
    )
with c4:
    color_by = st.selectbox(
        "Color by",
        ["cluster_label", "paper_name", "page"],
        key="vs_sel_color",
    )
with c5:
    if mode == "sub":
        if st.button("← Full corpus", use_container_width=True):
            for k in ("vs_mode", "vs_sub_mask", "vs_projection", "vs_labels",
                      "vs_fitted_reducer", "vs_reducer_method",
                      "vs_query_pt", "vs_query_cluster", "vs_clicked_chunk"):
                st.session_state[k] = _SS_DEFAULTS[k]
            st.rerun()

# ── Clustering params (one row, collapsed into expander to save space)
with st.expander("Clustering parameters", expanded=False):
    _cp1, _cp2, _cp3 = st.columns(3)
    n_pts = len(active_vecs)

    if cluster_choice == "HDBSCAN":
        with _cp1:
            mcs = st.slider(
                "min_cluster_size", 5, max(6, n_pts // 50),
                max(10, n_pts // 200), key="vs_hdbscan_mcs",
            )
        with _cp2:
            mss = st.slider("min_samples", 1, 30, 5, key="vs_hdbscan_mss")
        cluster_params = {"min_cluster_size": mcs, "min_samples": mss}

    elif cluster_choice == "Leiden":
        with _cp1:
            res = st.slider("resolution", 0.1, 5.0, 1.0, 0.1, key="vs_leiden_res")
        with _cp2:
            nb  = st.slider("n_neighbors", 5, 50, 15, key="vs_leiden_nb")
        cluster_params = {"resolution": res, "n_neighbors": nb}

    else:
        with _cp1:
            n_cl = st.slider("n_clusters", 2, 40, 8, key="vs_n_clusters")
        cluster_params = {"n_clusters": n_cl}

# ── Action buttons
btn_col1, btn_col2, _ = st.columns([1.4, 1.4, 5])
with btn_col1:
    build_proj_btn = st.button(
        "Build projection",
        type="primary",
        use_container_width=True,
        key="vs_build_proj",
    )
with btn_col2:
    apply_cluster_btn = st.button(
        "Apply clustering",
        use_container_width=True,
        key="vs_apply_cluster",
        disabled=st.session_state["vs_projection"] is None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PROJECTION — build or load from cache
# ─────────────────────────────────────────────────────────────────────────────

proj_key = _proj_cache_key(reducer_choice, {}, len(active_vecs), n_components)

if build_proj_btn:
    with st.spinner(f"Computing {reducer_choice} ({len(active_vecs):,} points, {n_components}D)…"):
        proj, fitted = run_reducer(reducer_choice, active_vecs, n_components)
    _save_cache(proj_key, {"projection": proj, "fitted": fitted})
    st.session_state["vs_projection"]     = proj
    st.session_state["vs_fitted_reducer"] = fitted
    st.session_state["vs_reducer_method"] = reducer_choice
    # Auto-apply clustering after projection
    with st.spinner(f"Clustering with {cluster_choice}…"):
        st.session_state["vs_labels"] = run_clusterer(cluster_choice, proj, cluster_params)
    st.session_state["vs_query_pt"]      = None
    st.session_state["vs_query_cluster"] = None

elif st.session_state["vs_projection"] is None:
    # Try to restore from disk (e.g. after Streamlit restart)
    cached = _load_cache(proj_key)
    if cached:
        st.session_state["vs_projection"]     = cached["projection"]
        st.session_state["vs_fitted_reducer"] = cached["fitted"]
        st.session_state["vs_reducer_method"] = reducer_choice
        # Also apply clustering if we don't have labels yet
        if st.session_state["vs_labels"] is None:
            with st.spinner("Clustering…"):
                st.session_state["vs_labels"] = run_clusterer(
                    cluster_choice, cached["projection"], cluster_params
                )

if apply_cluster_btn and st.session_state["vs_projection"] is not None:
    with st.spinner(f"Clustering with {cluster_choice}…"):
        st.session_state["vs_labels"] = run_clusterer(
            cluster_choice, st.session_state["vs_projection"], cluster_params
        )

# Bail if nothing to show yet
if st.session_state["vs_projection"] is None:
    st.info(
        "Click **Build projection** to compute the vector layout. "
        "Results are cached to disk and reload instantly next session."
    )
    st.stop()

projection = st.session_state["vs_projection"]
labels     = st.session_state["vs_labels"]

if labels is None:
    with st.spinner("Clustering…"):
        labels = run_clusterer(cluster_choice, projection, cluster_params)
        st.session_state["vs_labels"] = labels

n_clusters  = len(set(labels) - {-1})
noise_count = int((labels == -1).sum())


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT — scatter (left 2/3) + side panel (right 1/3)
# ─────────────────────────────────────────────────────────────────────────────

left_col, right_col = st.columns([2, 1], gap="medium")

# ── RIGHT PANEL ──────────────────────────────────────────────────────────────
with right_col:

    # ── Query input
    st.markdown(
        "<div style='font-size:.7rem;color:#7c9ef5;letter-spacing:.1em;"
        "text-transform:uppercase;font-weight:600;margin-bottom:.4rem'>Query</div>",
        unsafe_allow_html=True,
    )
    query_text = st.text_area(
        "Enter a sentence to locate it in the space",
        height=90,
        key="vs_query_text",
        label_visibility="collapsed",
        placeholder="Enter a sentence to locate it in the space…",
    )

    q_btn_col, sub_btn_col = st.columns(2)
    with q_btn_col:
        run_query_btn = st.button(
            "Project →",
            use_container_width=True,
            key="vs_run_query",
            disabled=not query_text.strip(),
        )
    with sub_btn_col:
        sub_cluster_btn = st.button(
            "Sub-cluster ★",
            use_container_width=True,
            key="vs_sub_btn",
            disabled=st.session_state["vs_query_cluster"] is None,
            type="primary",
        )

    # ── Run query projection
    if run_query_btn and query_text.strip():
        fitted  = st.session_state["vs_fitted_reducer"]
        method  = st.session_state["vs_reducer_method"]
        with st.spinner("Embedding…"):
            qvec = np.array(
                get_embedding_model().embed_query(query_text),
                dtype=np.float32,
            )
        qpt = project_query(fitted, method, qvec)
        if qpt is not None:
            st.session_state["vs_query_pt"] = qpt
            # Find which cluster the query point landed in
            from scipy.spatial import KDTree
            tree    = KDTree(projection)
            _, nidx = tree.query(qpt[0])
            qlabel  = int(labels[nidx])
            st.session_state["vs_query_cluster"] = qlabel
            n_in    = int((labels == qlabel).sum())
            st.success(
                f"Cluster **{qlabel}** · {n_in:,} chunks",
                icon="★",
            )
        else:
            st.warning(
                f"**{method}** doesn't support out-of-sample projection. "
                "Switch to PaCMAP, UMAP, or t-SNE.",
                icon="⚠",
            )

    # ── Sub-cluster action
    if sub_cluster_btn and st.session_state["vs_query_cluster"] is not None:
        target = st.session_state["vs_query_cluster"]

        if mode == "sub":
            # Already in sub-mode: map local labels back to global indices
            parent_mask  = st.session_state["vs_sub_mask"]
            global_idx   = np.where(parent_mask)[0]
            local_mask   = labels == target
            new_mask     = np.zeros(len(_all_vectors), dtype=bool)
            new_mask[global_idx[local_mask]] = True
        else:
            new_mask = (labels == target)

        if new_mask.sum() < 4:
            st.error("Cluster too small to sub-cluster (< 4 points).")
        else:
            st.session_state["vs_sub_mask"]       = new_mask
            st.session_state["vs_mode"]           = "sub"
            st.session_state["vs_projection"]     = None
            st.session_state["vs_labels"]         = None
            st.session_state["vs_fitted_reducer"] = None
            st.session_state["vs_reducer_method"] = None
            st.session_state["vs_query_pt"]       = None
            st.session_state["vs_query_cluster"]  = None
            st.rerun()

    # ── Cluster stats
    st.markdown(
        f"<div style='font-size:.75rem;color:#8890a8;margin-top:.6rem'>"
        f"{n_clusters} clusters &nbsp;·&nbsp; {noise_count} noise pts"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Chunk preview (updated on point click in scatter)
    st.markdown("---")
    st.markdown(
        "<div style='font-size:.7rem;color:#7c9ef5;letter-spacing:.1em;"
        "text-transform:uppercase;font-weight:600;margin-bottom:.4rem'>Chunk Preview</div>",
        unsafe_allow_html=True,
    )

    chunk_info = st.session_state["vs_clicked_chunk"]
    if chunk_info:
        st.markdown(
            f"<div class='chunk-card'>"
            f"<div class='chunk-meta'>{chunk_info['paper_name']} &nbsp;·&nbsp; p.{chunk_info['page']}</div>"
            f"{str(chunk_info['chunk']).replace('<','&lt;').replace('>','&gt;')}"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='color:#555e78;font-size:.8rem;line-height:1.6'>"
            "Click a point in the scatter to read its chunk here."
            "</div>",
            unsafe_allow_html=True,
        )


# ── LEFT PANEL — scatter ──────────────────────────────────────────────────────
with left_col:
    fig = build_figure(
        projection=projection,
        labels=labels,
        meta_df=active_meta,
        color_by=color_by,
        use_3d=use_3d,
        query_pt=st.session_state["vs_query_pt"],
        highlighted_cluster=st.session_state["vs_query_cluster"],
    )

    # Native Streamlit ≥1.33 selection API — no third-party package needed.
    # on_select="rerun" triggers a rerun when the user clicks a point.
    # Hover preview (220 chars) is shown via Plotly's hovertemplate — zero reruns.
    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="vs_plotly_chart",
    )

    sel_points = (event.selection.points if event and hasattr(event, "selection") else [])
    if sel_points:
        pt = sel_points[0]
        # curve_number 0 = corpus trace; skip clicks on the query ★ (curve 1)
        if pt.get("curve_number", 0) == 0:
            idx = pt.get("point_index")
            if idx is not None and idx < len(active_meta):
                row = active_meta.iloc[int(idx)]
                new_chunk = {
                    "paper_name": str(row["paper_name"]),
                    "page":       int(row["page"]),
                    "chunk":      str(row["chunk"]),
                }
                if new_chunk != st.session_state["vs_clicked_chunk"]:
                    st.session_state["vs_clicked_chunk"] = new_chunk
                    st.rerun()
