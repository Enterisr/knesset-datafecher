"""
Build a semantic search index from committee JSON files.

Uses intfloat/multilingual-e5-small (a proper bi-encoder) instead of raw
DictaBERT mean-pool, which suffers from BERT anisotropy and produces
uniformly high similarity scores (~0.93) for unrelated documents.

Usage:
    python build_index.py
    python build_index.py --data committee_data --out rag_index.npz --chunk-size 30 --stride 25
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8")

MARKER_RE = re.compile(r"<<[^>]*>>")

EMBED_MODEL = "intfloat/multilingual-e5-small"
CHUNK_SIZE   = 20
CHUNK_STRIDE = 10
BATCH_SIZE  = 256


def clean_text(text: str) -> str:
    """Strip raw << >> protocol markers left over from doc parsing."""
    return MARKER_RE.sub("", text).strip()


def make_chunks(utterances, size, stride):
    return [
        utterances[i : i + size]
        for i in range(0, len(utterances), stride)
        if utterances[i : i + size]
    ]


def build_index(data_glob: str, out_path: str, chunk_size: int, stride: int):
    print(f"Loading model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    paths = sorted(glob.glob(data_glob, recursive=True))
    print(f"Found {len(paths)} session files")

    all_vecs = []
    all_meta = []
    skipped  = 0

    for p_idx, path in enumerate(paths):
        if p_idx % 100 == 0:
            print(f"  {p_idx}/{len(paths)} …")
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)

            utts = doc.get("utterances", [])
            if not utts:
                skipped += 1
                continue

            doc_id = os.path.splitext(os.path.basename(path))[0]
            title  = doc.get("title", "")
            date   = (doc.get("date") or doc.get("session_date") or "")[:10]
            url    = doc.get("url", doc.get("source_file", ""))

            # Embed each utterance individually as "passage: <text>"
            texts = [
                f"passage: {clean_text(u.get('text', ''))}"
                for u in utts
            ]
            utt_vecs = model.encode(
                texts,
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            for i, chunk in enumerate(make_chunks(utts, chunk_size, stride)):
                start = i * stride
                end   = start + len(chunk)
                chunk_vec = utt_vecs[start:end].mean(axis=0)
                norm = np.linalg.norm(chunk_vec)
                if norm > 1e-9:
                    chunk_vec /= norm

                all_vecs.append(chunk_vec.astype(np.float32))
                all_meta.append({
                    "doc_id": doc_id,
                    "title":  title,
                    "date":   date,
                    "url":    url,
                    "from":   int(chunk[0]["id"]),
                    "to":     int(chunk[-1]["id"]),
                    "utterances": chunk,
                })

        except Exception as e:
            print(f"  [WARN] {path}: {e}")

    print(f"Done: {len(all_meta)} chunks from {len(paths) - skipped} sessions ({skipped} skipped)")

    vecs = np.stack(all_vecs).astype(np.float32)
    np.savez_compressed(
        out_path,
        vecs=vecs,
        metadata=np.array(all_meta, dtype=object),
    )
    print(f"Saved → {out_path}  (shape: {vecs.shape})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default=os.path.join("committee_data", "**", "*.json"))
    parser.add_argument("--out",        default="rag_index.npz")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--stride",     type=int, default=CHUNK_STRIDE)
    args = parser.parse_args()

    build_index(args.data, args.out, args.chunk_size, args.stride)


if __name__ == "__main__":
    main()
