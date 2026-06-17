"""
Local RAG Q&A — DictaBERT retrieval + Qwen3-8B via Ollama.

Requirements:
    pip install transformers torch numpy ollama
    # and install Ollama from https://ollama.com, then:
    ollama pull qwen3:8b

Usage:
    python rag_local.py --index rag_index_dictabert.npz
"""

import argparse
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

EMBED_MODEL = 'intfloat/multilingual-e5-small'
LLM_MODEL   = 'qwen3:8b'
TOP_K       = 5

SYSTEM_PROMPT = """/no_think
אתה עוזר מחקר המתמחה בפרוטוקולי ועדות הכנסת.
ענה על שאלות המשתמש בעברית בלבד, בהתבסס בעיקר על הקטעים שסופקו.
בסוף התשובה ציין מאיזה ישיבות (שם המסמך ותאריך) שאבת את המידע.
אם הקטעים אינם מכילים מידע רלוונטי כלל — ציין זאת, אך אל תסרב לענות אם המידע קיים בקטעים."""


class Embedder:
    def __init__(self):
        print(f'Loading embedder ({EMBED_MODEL}) …')
        self.model = SentenceTransformer(EMBED_MODEL)

    def encode(self, texts):
        prefixed = [f'query: {t}' for t in texts]
        return self.model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )


def load_index(path):
    print(f'Loading index from {path} …')
    data     = np.load(path, allow_pickle=True)
    vecs     = data['vecs']
    metadata = data['metadata'].tolist()
    print(f'Loaded {len(metadata)} chunks.')
    return vecs, metadata


def retrieve(query, vecs, metadata, embedder, k=TOP_K):
    q    = embedder.encode([query])[0].astype(np.float32)
    sims = vecs @ q
    top  = np.argsort(sims)[::-1][:k]
    return [(metadata[i], float(sims[i])) for i in top]


def ask(question, vecs, metadata, embedder, k=TOP_K):
    hits = retrieve(question, vecs, metadata, embedder, k)

    context_parts = []
    for meta, sim in hits:
        display_name = meta.get('title') or meta.get('doc_id', '(ללא שם)')
        lines = [
            f"📄 {display_name[:80]}  [{meta['date']}]  "
            f"אמירות {meta['from']}–{meta['to']}  (דמיון: {sim:.2f})"
        ]
        for u in meta['utterances']:
            speaker = u.get('speaker', '')
            text    = u.get('text', '').replace('\n', ' ').strip()
            lines.append(f"  [{u['id']}] {speaker}: {text}")
        context_parts.append('\n'.join(lines))

    context  = '\n\n---\n\n'.join(context_parts)
    user_msg = (
        f"להלן קטעים רלוונטיים מפרוטוקולי ועדות הכנסת:\n\n"
        f"{context}\n\n"
        f"שאלה: {question}"
    )

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',   'content': user_msg},
        ],
        options={'temperature': 0.1},
    )
    answer = response['message']['content'].strip()
    return answer, hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default='rag_index.npz',
                        help='Path to the .npz index file')
    parser.add_argument('--top-k', type=int, default=TOP_K)
    args = parser.parse_args()

    vecs, metadata = load_index(args.index)
    embedder       = Embedder()

    print(f'\nUsing Ollama model: {LLM_MODEL}')
    print('Make sure Ollama is running: ollama serve')
    print('Ready. Type a question in Hebrew (or "exit" to quit).\n')
    while True:
        try:
            question = input('שאלה> ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() == 'exit':
            break

        answer, hits = ask(question, vecs, metadata, embedder, k=args.top_k)
        print('\n' + '─' * 60)
        print(answer)
        print('\n═ מקורות ═')
        for meta, sim in hits:
            display_name = meta.get('title') or meta.get('doc_id', '(ללא שם)')
            print(f"  • {display_name[:70]}  [{meta['date']}]  "
                  f"אמירות {meta['from']}–{meta['to']}  sim={sim:.2f}")
            url = meta.get('url', '')
            if url:
                print(f"    {url}")
        print()


if __name__ == '__main__':
    main()
