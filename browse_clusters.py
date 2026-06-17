"""
Browse precomputed HDBSCAN clusters in a Streamlit web UI.

Run clustering first:
    python cluster_embeddings.py --index rag_index_E5.npz --out clusters.pkl

Then launch the UI:
    streamlit run browse_clusters.py
    streamlit run browse_clusters.py -- --clusters clusters.pkl
"""

import argparse
import json
import logging
import pickle
import sys

import numpy as np
import plotly.graph_objects as go
import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ── CLI args ──────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--clusters', default='clusters.pkl')
    try:
        idx  = sys.argv.index('--')
        argv = sys.argv[idx + 1:]
    except ValueError:
        argv = []
    args, _ = parser.parse_known_args(argv)
    return args


_ARGS = _parse_args()


# ── data ──────────────────────────────────────────────────────────────────────

def _parse_payload(payload):
    clusters      = payload['clusters']
    sorted_ids    = payload['sorted_ids']
    params        = payload.get('params', {})
    coords3d      = payload.get('coords3d')
    labels        = payload.get('labels')
    metadata      = payload.get('metadata')
    cluster_names = payload.get('cluster_names', {})
    n_clusters = len([k for k in sorted_ids if k != -1])
    n_noise    = len(clusters.get(-1, []))
    total      = sum(len(v) for v in clusters.values())
    log.info('Loaded %d clusters | %d noise | %d total chunks | 3D=%s',
             n_clusters, n_noise, total, coords3d is not None)
    return clusters, sorted_ids, params, coords3d, labels, metadata, cluster_names


@st.cache_resource(show_spinner='Loading clusters…')
def load_clusters(path):
    log.info('Loading clusters from %s', path)
    with open(path, 'rb') as f:
        return _parse_payload(pickle.load(f))


@st.cache_resource(show_spinner='Loading uploaded clusters…')
def load_clusters_upload(file_bytes):
    return _parse_payload(pickle.loads(file_bytes))


# ── colors ────────────────────────────────────────────────────────────────────

CLUSTER_COLORS = [
    '#89b4fa', '#a6e3a1', '#f9e2af', '#cba6f7', '#f38ba8',
    '#89dceb', '#fab387', '#94e2d5', '#eba0ac', '#b4befe',
    '#cba6f7', '#f5c2e7', '#45475a', '#585b70', '#7f849c',
]

def cluster_color(cid):
    return '#3d3d5c' if cid == -1 else CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]


# ── 3D scatter ────────────────────────────────────────────────────────────────

def build_3d_figure(coords3d, labels, metadata, show_noise, cluster_names):
    unique_labels = sorted(set(labels))
    fig = go.Figure()

    for cid in unique_labels:
        if cid == -1 and not show_noise:
            continue
        mask  = labels == cid
        pts   = coords3d[mask]
        metas = [metadata[i] for i, m in enumerate(mask) if m]
        color = cluster_color(cid)
        name  = cluster_names.get(cid, f'Cluster {cid}') if cid != -1 else 'Noise'
        hover = [
            f"{m.get('title') or m.get('doc_id','?')}<br>{m.get('date','')}"
            for m in metas
        ]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers',
            name=name,
            marker=dict(size=3 if cid != -1 else 2, color=color, opacity=0.7 if cid != -1 else 0.3),
            text=hover,
            hovertemplate='%{text}<extra>' + name + '</extra>',
        ))

    fig.update_layout(
        paper_bgcolor='#1e1e2e',
        plot_bgcolor='#1e1e2e',
        scene=dict(
            bgcolor='#1e1e2e',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
        ),
        legend=dict(bgcolor='#2a2a3e', font=dict(color='#cdd6f4'), itemsizing='constant'),
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
    )
    return fig


# ── chunk rendering ───────────────────────────────────────────────────────────

def render_chunk(meta, idx, color):
    title  = meta.get('title') or meta.get('doc_id', '(ללא שם)')
    date   = meta.get('date', '')
    from_u = meta.get('from', '')
    to_u   = meta.get('to', '')
    url    = meta.get('url', '')
    utts   = meta.get('utterances', [])

    header = (
        f'<div style="border-left:3px solid {color};padding:6px 12px;margin-bottom:6px;'
        f'background:#2a2a3e;border-radius:4px">'
        f'<span style="color:{color};font-size:0.8em">#{idx}</span>'
        f'<span style="color:#6c7086;font-size:0.8em"> · {date}  utts {from_u}–{to_u}</span>'
        f'<p style="text-align:right;font-weight:bold;color:#cdd6f4;margin:4px 0">{title}</p>'
    )
    st.markdown(header, unsafe_allow_html=True)

    if url:
        st.markdown(f'[{url}]({url})')

    for u in utts:
        speaker = u.get('speaker', '').strip()
        text    = u.get('text', '').replace('\n', ' ').strip()
        parts = ''
        if speaker:
            parts += (f'<p style="text-align:right;color:#89b4fa;font-size:0.85em;margin:2px 0">'
                      f'<b>{speaker}</b></p>')
        parts += f'<p style="text-align:right;color:#cdd6f4;font-size:0.9em;margin:2px 0">{text}</p>'
        st.markdown(parts, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── app ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title='Knesset Cluster Browser', layout='wide')

st.markdown(
    '<style>body,.stApp{background:#1e1e2e;color:#cdd6f4}'
    'section[data-testid="stSidebar"]{background:#181825}</style>',
    unsafe_allow_html=True,
)

st.title('Knesset Cluster Browser')

with st.sidebar:
    clusters_path = st.text_input('Clusters file (local)', value=_ARGS.clusters)
    uploaded      = st.file_uploader('Or upload clusters.pkl', type=['pkl'])
    filter_text   = st.text_input('Filter (title / text)', '')
    show_noise    = st.checkbox('Show noise in 3D', value=False)

import os
if uploaded is not None:
    clusters, sorted_ids, params, coords3d, labels, metadata, cluster_names = \
        load_clusters_upload(uploaded.read())
elif os.path.exists(clusters_path):
    clusters, sorted_ids, params, coords3d, labels, metadata, cluster_names = \
        load_clusters(clusters_path)
else:
    st.info('Upload a `clusters.pkl` file generated by `cluster_embeddings.py`.')
    st.stop()

n_clusters = len([k for k in sorted_ids if k != -1])
n_noise    = len(clusters.get(-1, []))
total      = sum(len(v) for v in clusters.values())

if params:
    st.caption(
        f"{n_clusters} clusters · {n_noise} noise · {total:,} chunks  |  "
        f"index: {params.get('index','')}  "
        f"umap_dims={params.get('umap_dims')}  "
        f"min_cluster={params.get('min_cluster_size')}  "
        f"min_samples={params.get('min_samples')}  "
        f"viz={params.get('viz','?')}"
    )
else:
    st.caption(f'{n_clusters} clusters · {n_noise} noise · {total:,} chunks')

# ── 3D plot ───────────────────────────────────────────────────────────────────

if coords3d is not None and labels is not None and metadata is not None:
    fig = build_3d_figure(coords3d, np.array(labels), metadata, show_noise, cluster_names)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info('No 3D coords in this pkl. Re-run `cluster_embeddings.py` to generate them.')

st.divider()

# ── cluster list ──────────────────────────────────────────────────────────────

filt = filter_text.strip().lower()

for cid in sorted_ids:
    items = clusters[cid]
    if filt:
        items = [
            (i, m) for i, m in items
            if filt in (m.get('title', '') or '').lower()
            or any(filt in u.get('text', '').lower() for u in m.get('utterances', []))
        ]
    if not items:
        continue

    color = cluster_color(cid)
    if cid == -1:
        label = f'Noise / Unassigned  ({len(items)} chunks)'
    else:
        name  = cluster_names.get(cid, f'Cluster {cid}')
        label = f'[{cid}] {name}  ({len(items)} chunks)'

    with st.expander(label, expanded=False):
        if st.button(f'Prepare download ({len(items)} chunks)', key=f'prep_{cid}'):
            st.session_state[f'dl_{cid}'] = json.dumps(
                [{'chunk_idx': idx, **meta} for idx, meta in items],
                ensure_ascii=False, indent=2,
            ).encode('utf-8')

        if f'dl_{cid}' in st.session_state:
            st.download_button(
                label=f'⬇ Download cluster_{cid}.json',
                data=st.session_state[f'dl_{cid}'],
                file_name=f'cluster_{cid}.json',
                mime='application/json',
                key=f'dl_btn_{cid}',
            )

        for idx, meta in items:
            render_chunk(meta, idx, color)
