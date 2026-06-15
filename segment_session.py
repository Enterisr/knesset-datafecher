"""
RAG-augmented few-shot session segmenter.

For a given session JSON, chunks the utterances into windows,
retrieves the most similar gold-annotated session as a few-shot example,
and calls an LLM to predict topic boundaries.

Usage:
    # local parliamentary model (default)
    python segment_session.py --session_id 25_ptv_XXXXXXX

    # 4-bit quantized (less VRAM, ~4 GB)
    python segment_session.py --session_id 25_ptv_XXXXXXX --quantize

    # Anthropic API fallback
    python segment_session.py --session_id 25_ptv_XXXXXXX --backend anthropic

    # dry-run: print prompt only, no model call
    python segment_session.py --session_id 25_ptv_XXXXXXX --dry_run

Outputs a CSV: doc_id, utterance_from, utterance_to, subject_label_predicted
"""

import argparse
import csv
import json
import os
import glob
import sys
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── Config ──────────────────────────────────────────────────────────────────
GOLD_CSV = "committee_subject_labeling_signal.csv"
DATA_GLOB = os.path.join("committee_data", "**", "*.json")
CHUNK_SIZE = 80          # utterances per LLM call
CHUNK_OVERLAP = 10       # overlap between chunks to avoid boundary misses
TOP_K_EXAMPLES = 2       # gold examples to include in prompt
EMBED_MODEL = "intfloat/multilingual-e5-small"
LLM_MODEL_ANTHROPIC = "claude-sonnet-4-6"
LLM_MODEL_LOCAL = "abanwild/nation-parliamentary-gemma4-e4b-grpo-lora"
# ────────────────────────────────────────────────────────────────────────────


def load_gold(gold_csv: str):
    """Return {doc_id: [{"from": int, "to": int, "label": str, "tags": str}]}"""
    gold = {}
    with open(gold_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            doc_id = row["doc_id"]
            gold.setdefault(doc_id, []).append({
                "from": int(row["utterance_from"]),
                "to": int(row["utterance_to"]),
                "label": row["subject_label"],
                "tags": row["tags"],
                "session_title": row["session_title"],
                "session_date": row["session_date"],
            })
    return gold


def build_path_map():
    return {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob.glob(DATA_GLOB, recursive=True)
    }


def load_session(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def session_repr(doc: dict, gold_segments: list) -> str:
    """One-line text repr of a gold session for embedding / display."""
    title = doc.get("title", "")
    labels = " | ".join(s["label"] for s in gold_segments)
    return f"{title} {labels}"


# ── Embedding ────────────────────────────────────────────────────────────────

_embed_model = None

def get_embedder():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def embed(texts: list[str]) -> np.ndarray:
    model = get_embedder()
    # multilingual-e5 expects "query: " / "passage: " prefixes
    prefixed = [f"passage: {t}" for t in texts]
    return model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)


def embed_query(text: str) -> np.ndarray:
    model = get_embedder()
    return model.encode([f"query: {text}"], normalize_embeddings=True, show_progress_bar=False)[0]


# ── Gold index ───────────────────────────────────────────────────────────────

def build_gold_index(gold: dict, path_map: dict):
    """
    Returns:
        gold_ids: list of doc_ids
        gold_vecs: np.ndarray shape (N, D)
        gold_docs: {doc_id: loaded JSON}
    """
    gold_ids = []
    texts = []
    gold_docs = {}

    for doc_id, segments in gold.items():
        path = path_map.get(doc_id)
        if not path:
            continue
        doc = load_session(path)
        gold_docs[doc_id] = doc
        text = session_repr(doc, segments)
        gold_ids.append(doc_id)
        texts.append(text)

    print(f"Building index over {len(texts)} gold sessions …")
    vecs = embed(texts)
    return gold_ids, vecs, gold_docs


def retrieve_examples(query_text: str, gold_ids, gold_vecs, k=TOP_K_EXAMPLES):
    """Return indices of top-K most similar gold sessions."""
    q = embed_query(query_text)
    sims = gold_vecs @ q
    top = np.argsort(sims)[::-1][:k]
    return top.tolist(), sims[top].tolist()


# ── Prompt building ──────────────────────────────────────────────────────────

def format_utterances(utterances: list[dict], start_id: int, end_id: int) -> str:
    lines = []
    for u in utterances:
        uid = int(u["id"])
        if start_id <= uid <= end_id:
            speaker = u.get("speaker", "")
            text = u.get("text", "").replace("\n", " ").strip()
            lines.append(f"[{uid}] {speaker}: {text}")
    return "\n".join(lines)


def format_gold_example(doc: dict, segments: list) -> str:
    """Format a gold session as a labeled example for the prompt."""
    utts = doc["utterances"]
    lines = [f"=== דוגמה: {doc.get('title', doc['doc_id'])} ({doc.get('date','')[:10]}) ==="]
    for seg in segments:
        lines.append(f"\n--- קטע: אמירות {seg['from']}–{seg['to']} ---")
        lines.append(f"נושא: {seg['label']}")
        lines.append(f"תגיות: {seg['tags']}")
        snippet = format_utterances(utts, seg["from"], min(seg["from"] + 4, seg["to"]))
        lines.append(snippet)
        lines.append("  …")
    return "\n".join(lines)


SYSTEM_PROMPT = """אתה מסייע בניתוח פרוטוקולי ועדת הכנסת.
תפקידך: לחלק קטע מפרוטוקול לנושאים נפרדים ולזהות את גבולות כל נושא.

כללים:
- כל קטע מתחיל במספר האמירה הראשונה שעוסקת בנושא חדש.
- כל קטע מסתיים במספר האמירה האחרונה לפני המעבר לנושא הבא.
- ציין שם נושא קצר בעברית (עד 15 מילה).
- החזר JSON בלבד — מערך של אובייקטים עם המפתחות: from, to, subject_label.
"""

def build_chunk_prompt(chunk_utts: list[dict], examples: list[tuple]) -> str:
    """
    examples: list of (doc, segments) for few-shot.
    """
    parts = []

    if examples:
        parts.append("להלן דוגמאות לאופן החלוקה לנושאים:\n")
        for doc, segs in examples:
            parts.append(format_gold_example(doc, segs))

    start_id = int(chunk_utts[0]["id"])
    end_id = int(chunk_utts[-1]["id"])
    parts.append(f"\n=== פרוטוקול לחלוקה (אמירות {start_id}–{end_id}) ===\n")
    for u in chunk_utts:
        speaker = u.get("speaker", "")
        text = u.get("text", "").replace("\n", " ").strip()
        parts.append(f"[{u['id']}] {speaker}: {text}")

    parts.append(
        f"\n\nחלק את האמירות {start_id}–{end_id} לנושאים."
        " החזר JSON בלבד — מערך: [{\"from\": N, \"to\": M, \"subject_label\": \"...\"}]"
        " ללא שום טקסט נוסף."
    )
    return "\n".join(parts)


# ── LLM backends ─────────────────────────────────────────────────────────────

_local_pipe = None

def load_local_model(quantize: bool = False):
    global _local_pipe
    if _local_pipe is not None:
        return _local_pipe
    import torch
    from transformers import pipeline, BitsAndBytesConfig

    print(f"Loading local model: {LLM_MODEL_LOCAL}")
    kwargs = dict(
        model=LLM_MODEL_LOCAL,
        task="text-generation",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        del kwargs["torch_dtype"]

    _local_pipe = pipeline(**kwargs)
    print("Local model loaded.")
    return _local_pipe


def call_local_llm(prompt: str, quantize: bool = False) -> str:
    pipe = load_local_model(quantize)
    messages = [
        {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + prompt},
    ]
    out = pipe(messages, max_new_tokens=1024, do_sample=False)
    # pipeline returns list of dicts; extract generated text
    generated = out[0]["generated_text"]
    # the pipeline appends the assistant reply as the last message
    if isinstance(generated, list):
        return generated[-1]["content"].strip()
    return str(generated).strip()


def call_anthropic_llm(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=LLM_MODEL_ANTHROPIC,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def call_llm(prompt: str, backend: str = "local", quantize: bool = False) -> str:
    if backend == "anthropic":
        return call_anthropic_llm(prompt)
    return call_local_llm(prompt, quantize)


def parse_llm_response(text: str) -> list[dict]:
    """Extract the JSON array from the LLM response."""
    import re
    # strip markdown code fences if present
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    # fallback: find first [...] block
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    print(f"  [WARN] Could not parse LLM response:\n{text[:300]}")
    return []


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_utterances(utterances: list[dict], size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    i = 0
    while i < len(utterances):
        chunk = utterances[i: i + size]
        chunks.append(chunk)
        i += size - overlap
    return chunks


def merge_chunks(chunk_results: list[list[dict]], utterances: list[dict]) -> list[dict]:
    """
    Merge per-chunk segment predictions into a deduplicated session-level list.
    Strategy: take the union of all predicted boundaries, then merge overlapping spans.
    """
    all_boundaries = set()
    for segs in chunk_results:
        for s in segs:
            try:
                all_boundaries.add(int(s["from"]))
            except (KeyError, ValueError):
                pass

    if not all_boundaries:
        return []

    # Build spans from boundaries
    sorted_b = sorted(all_boundaries)
    max_id = int(utterances[-1]["id"])
    spans = []
    for idx, b in enumerate(sorted_b):
        end = sorted_b[idx + 1] - 1 if idx + 1 < len(sorted_b) else max_id
        spans.append({"from": b, "to": end})

    # Collect labels from chunk results
    label_map = {}
    for segs in chunk_results:
        for s in segs:
            try:
                f = int(s["from"])
                if f not in label_map:
                    label_map[f] = s.get("subject_label", "")
            except (KeyError, ValueError):
                pass

    for span in spans:
        span["subject_label"] = label_map.get(span["from"], "")

    return spans


# ── Main ─────────────────────────────────────────────────────────────────────

def segment_session(session_id: str, out_dir: str | None = None, dry_run: bool = False,
                    backend: str = "local", quantize: bool = False):
    gold = load_gold(GOLD_CSV)
    path_map = build_path_map()

    path = path_map.get(session_id)
    if not path:
        print(f"Session {session_id} not found in committee_data/")
        return

    doc = load_session(path)
    utterances = doc["utterances"]
    title = doc.get("title", session_id)
    print(f"Session: {session_id} | {len(utterances)} utterances | {title[:60]}")

    # Build retrieval index (excluding the target session itself)
    retrieval_gold = {k: v for k, v in gold.items() if k != session_id}
    gold_ids, gold_vecs, gold_docs = build_gold_index(retrieval_gold, path_map)

    # Query text for retrieval
    query_text = title + " " + " ".join(
        u["text"][:50] for u in utterances[:5]
    )

    example_idxs, sims = retrieve_examples(query_text, gold_ids, gold_vecs)
    examples = [
        (gold_docs[gold_ids[i]], gold[gold_ids[i]])
        for i in example_idxs
    ]
    print(f"Retrieved examples: {[gold_ids[i] for i in example_idxs]} (sims: {[f'{s:.3f}' for s in sims]})")

    chunks = chunk_utterances(utterances)
    print(f"Chunks: {len(chunks)} × ~{CHUNK_SIZE} utterances")

    if dry_run:
        print("\n[DRY RUN] First chunk prompt preview (first 1000 chars):")
        prompt = build_chunk_prompt(chunks[0], examples)
        print(prompt[:1000])
        print("…")
        return

    chunk_results = []
    for i, chunk in enumerate(chunks):
        start = chunk[0]["id"]
        end = chunk[-1]["id"]
        print(f"  Chunk {i+1}/{len(chunks)}: utterances {start}–{end} … ", end="", flush=True)
        prompt = build_chunk_prompt(chunk, examples)
        raw = call_llm(prompt, backend=backend, quantize=quantize)
        segs = parse_llm_response(raw)
        chunk_results.append(segs)
        print(f"{len(segs)} segments")

    merged = merge_chunks(chunk_results, utterances)
    print(f"\nFinal: {len(merged)} segments")

    rows = []
    for seg in merged:
        rows.append({
            "doc_id": session_id,
            "utterance_from": seg["from"],
            "utterance_to": seg["to"],
            "subject_label_predicted": seg.get("subject_label", ""),
        })

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{session_id}_segments.csv")
    else:
        out_path = f"{session_id}_segments.csv"

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "utterance_from", "utterance_to", "subject_label_predicted"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved → {out_path}")
    for r in rows:
        print(f"  [{r['utterance_from']}–{r['utterance_to']}] {r['subject_label_predicted']}")

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", required=True, help="e.g. 25_ptv_2559210")
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--dry_run", action="store_true", help="print prompt preview, no LLM call")
    parser.add_argument("--backend", default="local", choices=["local", "anthropic"],
                        help="local = parliamentary Gemma (default), anthropic = Claude API")
    parser.add_argument("--quantize", action="store_true",
                        help="4-bit quantization for local model (~4 GB VRAM instead of ~8 GB)")
    args = parser.parse_args()
    segment_session(args.session_id, args.out, args.dry_run, args.backend, args.quantize)
