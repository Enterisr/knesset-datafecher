"""
Cluster Knesset embeddings with UMAP + HDBSCAN and save results to disk.

Usage:
    python cluster_embeddings.py --index rag_index_E5.npz
    python cluster_embeddings.py --index rag_index_E5.npz --umap-dims 10 --min-cluster-size 5 --out clusters.pkl
"""

import argparse
import logging
import pickle
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore', message='.*previously compiled argument types.*', module='numba')

import numpy as np
import umap
import hdbscan

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ── index ─────────────────────────────────────────────────────────────────────

def load_index(path):
    log.info('Loading index from %s', path)
    data = np.load(path, allow_pickle=True)
    vecs = data['vecs'].astype(np.float32)
    metadata = data['metadata'].tolist()
    log.info('Loaded %d chunks  (vec dim=%d)', len(metadata), vecs.shape[1])
    return vecs, metadata


# ── pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(vecs, umap_dims, umap_neighbors, min_cluster_size, min_samples):
    log.info('UMAP: %d → %d dims  (neighbors=%d)', vecs.shape[1], umap_dims, umap_neighbors)
    reducer = umap.UMAP(
        n_components=umap_dims,
        n_neighbors=umap_neighbors,
        metric='cosine',
        random_state=42,
        low_memory=False,
    )
    reduced = reducer.fit_transform(vecs)
    log.info('UMAP done')

    log.info('HDBSCAN: min_cluster_size=%d  min_samples=%d', min_cluster_size, min_samples)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='leaf',
    )
    labels = clusterer.fit_predict(reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    log.info('Clusters: %d  |  Noise: %d', n_clusters, n_noise)
    return labels, reduced


def reduce_3d(reduced, method='umap'):
    """Further reduce to 3D for visualization."""
    if method == 'pca':
        from sklearn.decomposition import PCA
        log.info('PCA: %d → 3 dims', reduced.shape[1])
        coords = PCA(n_components=3, random_state=42).fit_transform(reduced)
    else:
        log.info('UMAP: %d → 3 dims  (viz)', reduced.shape[1])
        coords = umap.UMAP(
            n_components=3,
            n_neighbors=30,
            metric='euclidean',
            random_state=42,
        ).fit_transform(reduced)
    log.info('3D reduction done')
    return coords.astype(np.float32)


def build_cluster_map(labels, metadata):
    clusters = defaultdict(list)
    for idx, (label, meta) in enumerate(zip(labels, metadata)):
        clusters[int(label)].append((idx, meta))
    sorted_ids = sorted([k for k in clusters if k != -1], key=lambda k: -len(clusters[k]))
    if -1 in clusters:
        sorted_ids.append(-1)
    return dict(clusters), sorted_ids


def name_clusters(clusters, sorted_ids, top_n=5):
    """Assign each cluster a name from its top TF-IDF terms (works for Hebrew)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    cids = [cid for cid in sorted_ids if cid != -1]

    # one "document" per cluster: all utterance text concatenated
    docs = []
    for cid in cids:
        parts = []
        for _, meta in clusters[cid]:
            for u in meta.get('utterances', []):
                parts.append(u.get('text', ''))
        docs.append(' '.join(parts))

    if not docs:
        return {}

    vec = TfidfVectorizer(
        analyzer='word',
        token_pattern=r'\S+',   # any whitespace-split token — handles Hebrew
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
    )
    tfidf = vec.fit_transform(docs)
    terms = vec.get_feature_names_out()

    names = {}
    for i, cid in enumerate(cids):
        row   = tfidf[i].toarray().flatten()
        top   = row.argsort()[::-1][:top_n]
        label = ' · '.join(terms[j] for j in top if row[j] > 0)
        names[cid] = label or f'Cluster {cid}'
        log.info('Cluster %3d  →  %s', cid, names[cid])

    return names


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index',            default='rag_index_E5.npz')
    parser.add_argument('--out',              default='clusters.pkl')
    parser.add_argument('--umap-dims',        type=int,  default=5)
    parser.add_argument('--umap-neighbors',   type=int,  default=50)
    parser.add_argument('--min-cluster-size', type=int,  default=40)
    parser.add_argument('--min-samples',      type=int,  default=1)
    parser.add_argument('--viz',              choices=['umap', 'pca'], default='umap',
                        help='Method for 3D visualization reduction')
    args = parser.parse_args()

    vecs, metadata = load_index(args.index)
    labels, reduced = run_pipeline(
        vecs,
        umap_dims=args.umap_dims,
        umap_neighbors=args.umap_neighbors,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )
    coords3d = reduce_3d(reduced, method=args.viz)
    clusters, sorted_ids = build_cluster_map(labels, metadata)
    log.info('Naming clusters via TF-IDF…')
    cluster_names = name_clusters(clusters, sorted_ids)

    payload = {
        'clusters':      clusters,
        'sorted_ids':    sorted_ids,
        'coords3d':      coords3d,
        'labels':        labels,
        'metadata':      metadata,
        'cluster_names': cluster_names,
        'params': {
            'index':            args.index,
            'umap_dims':        args.umap_dims,
            'umap_neighbors':   args.umap_neighbors,
            'min_cluster_size': args.min_cluster_size,
            'min_samples':      args.min_samples,
            'viz':              args.viz,
        },
    }

    with open(args.out, 'wb') as f:
        pickle.dump(payload, f)
    log.info('Saved → %s', args.out)


if __name__ == '__main__':
    main()
