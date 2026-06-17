"""
Embedding quality test — run after building a new index.

Checks:
  1. Random-pair similarity distribution (should be low mean, high spread)
  2. Hebrew semantic sanity pairs (similar >> different)
  3. Saves plots to output_folder/

Usage:
    python test_embedding_quality.py
    python test_embedding_quality.py --index rag_index.npz --out quality_report
"""

import argparse
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

HEBREW_PAIRS = [
    # (text_a, text_b, label)   label: 1 = similar, 0 = different
    ("תקציב המדינה לשנת 2024",          "הוצאות ממשלתיות ותקציב",              1),
    ("מיסוי מקרקעין ורכישת דירות",       'מס רכישה על נדל"ן',                   1),
    ("שר האוצר דיבר על הגירעון",         "הגירעון הממשלתי עלה השנה",            1),
    ("ריבית בנק ישראל עלתה",             "העלאת הריבית על ידי הבנק המרכזי",     1),
    ("חוק הביטוח הלאומי",               "קצבאות נכות ואבטלה",                   1),
    ("ועדת הכספים דנה בתקציב",           "חוק חינוך חינם לגיל הרך",             0),
    ("מיסוי מקרקעין",                    "ביטחון לאומי וצבא",                    0),
    ("שיעור הריבית של בנק ישראל",        "חוק הלאום",                            0),
    ("הצעת חוק הפנסיה",                  "פרויקט תשתיות כבישים",                 0),
    ("גירעון תקציבי",                    "מדיניות חוץ ויחסים דיפלומטיים",        0),
]

N_RANDOM_PAIRS = 1000
RANDOM_SEED    = 42


# ── helpers ───────────────────────────────────────────────────────────────────

def load_index(path):
    data     = np.load(path, allow_pickle=True)
    vecs     = data["vecs"].astype(np.float32)
    mean_vec = data["mean_vec"] if "mean_vec" in data else None
    return vecs, mean_vec


def center(vecs, mean_vec):
    if mean_vec is None:
        return vecs
    c = vecs - mean_vec
    norms = np.linalg.norm(c, axis=1, keepdims=True)
    return c / np.where(norms > 1e-9, norms, 1.0)


def random_pair_sims(vecs, n=N_RANDOM_PAIRS, seed=RANDOM_SEED):
    rng = random.Random(seed)
    idx = [(rng.randint(0, len(vecs) - 1), rng.randint(0, len(vecs) - 1))
           for _ in range(n)]
    return [float(vecs[a] @ vecs[b]) for a, b in idx if a != b]


def hebrew_sims(model):
    from sentence_transformers import SentenceTransformer   # noqa: F401 (already imported by caller)
    texts   = [f"query: {p[0]}" for p in HEBREW_PAIRS] + \
              [f"query: {p[1]}" for p in HEBREW_PAIRS]
    n       = len(HEBREW_PAIRS)
    vecs    = model.encode(texts, normalize_embeddings=True, show_progress_bar=False,
                           convert_to_numpy=True)
    results = []
    for i, (a, b, label) in enumerate(HEBREW_PAIRS):
        sim = float(vecs[i] @ vecs[n + i])
        results.append((a, b, label, sim))
    return results


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_random_distribution(sims, out_dir, index_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Random-pair similarity — {index_name}", fontsize=13)

    axes[0].hist(sims, bins=60, color="#89b4fa", edgecolor="none")
    axes[0].axvline(np.mean(sims), color="#f38ba8", linewidth=2, label=f"mean={np.mean(sims):.3f}")
    axes[0].set_xlabel("Cosine similarity")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution")
    axes[0].legend()

    axes[1].plot(sorted(sims), color="#a6e3a1", linewidth=1)
    axes[1].set_xlabel("Rank")
    axes[1].set_ylabel("Cosine similarity")
    axes[1].set_title("Sorted scores")
    axes[1].axhline(np.mean(sims), color="#f38ba8", linewidth=1.5, linestyle="--",
                    label=f"mean={np.mean(sims):.3f}")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "random_pair_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved → {path}")


def plot_hebrew_pairs(results, out_dir):
    labels  = [f"{a[:20]}…" for a, _, _, _ in results]
    scores  = [sim for _, _, _, sim in results]
    colors  = ["#a6e3a1" if lbl == 1 else "#f38ba8" for _, _, lbl, _ in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, scores, color=colors)
    ax.set_xlim(min(scores) - 0.05, 1.02)
    ax.axvline(0, color="white", linewidth=0.5)
    ax.set_xlabel("Cosine similarity")
    ax.set_title("Hebrew semantic pairs  (green=similar, red=different)")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#a6e3a1", label="Similar"),
                        Patch(color="#f38ba8", label="Different")])

    plt.tight_layout()
    path = os.path.join(out_dir, "hebrew_pairs.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved → {path}")


def plot_gap(sim_results, out_dir):
    sim_scores  = [s for _, _, lbl, s in sim_results if lbl == 1]
    diff_scores = [s for _, _, lbl, s in sim_results if lbl == 0]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([sim_scores, diff_scores], tick_labels=["Similar", "Different"],
               patch_artist=True,
               boxprops=dict(facecolor="#89b4fa", color="white"),
               medianprops=dict(color="#f9e2af", linewidth=2))
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Similar vs Different — gap check")
    plt.tight_layout()
    path = os.path.join(out_dir, "similar_vs_different.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="rag_index_dictabert_centered.npz")
    parser.add_argument("--embed-model", default="intfloat/multilingual-e5-small",
                        help="Sentence-transformer model for Hebrew pair test")
    parser.add_argument("--out", default="quality_report")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    index_name = os.path.basename(args.index)

    # ── 1. random pair test ───────────────────────────────────────────────────
    print(f"\nLoading index: {args.index}")
    vecs, mean_vec = load_index(args.index)
    vecs = center(vecs, mean_vec)
    print(f"  {len(vecs):,} chunks, dim={vecs.shape[1]}")

    print(f"\nSampling {N_RANDOM_PAIRS} random pairs …")
    sims = random_pair_sims(vecs)
    mean_s, std_s = np.mean(sims), np.std(sims)
    pct_08 = 100 * np.mean(np.array(sims) > 0.8)
    pct_09 = 100 * np.mean(np.array(sims) > 0.9)

    print(f"  mean={mean_s:.3f}  std={std_s:.3f}  min={min(sims):.3f}  max={max(sims):.3f}")
    print(f"  % above 0.8: {pct_08:.1f}%   % above 0.9: {pct_09:.1f}%")

    verdict = "GOOD" if mean_s < 0.5 and std_s > 0.1 else \
              "OK"   if mean_s < 0.8 else "BAD (anisotropy)"
    print(f"  → verdict: {verdict}")

    plot_random_distribution(sims, args.out, index_name)

    # ── 2. Hebrew semantic pairs ───────────────────────────────────────────────
    print(f"\nLoading embedder for Hebrew pairs: {args.embed_model}")
    from sentence_transformers import SentenceTransformer
    model   = SentenceTransformer(args.embed_model)
    results = hebrew_sims(model)

    sim_scores  = [s for _, _, lbl, s in results if lbl == 1]
    diff_scores = [s for _, _, lbl, s in results if lbl == 0]
    gap = np.mean(sim_scores) - np.mean(diff_scores)

    print("\n  Hebrew pairs:")
    for a, b, lbl, sim in results:
        tag = "similar " if lbl == 1 else "different"
        print(f"    [{tag}]  sim={sim:.3f}  {a[:35]}")

    print(f"\n  mean(similar)={np.mean(sim_scores):.3f}  "
          f"mean(different)={np.mean(diff_scores):.3f}  gap={gap:.3f}")
    heb_verdict = "GOOD" if gap > 0.08 else "OK" if gap > 0.04 else "BAD"
    print(f"  → verdict: {heb_verdict}")

    plot_hebrew_pairs(results, args.out)
    plot_gap(results, args.out)

    print(f"\nAll plots saved to ./{args.out}/")


if __name__ == "__main__":
    main()
