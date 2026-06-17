"""
Embedding search — DictaBERT retrieval only, no LLM.
Opens a tkinter window for input and results.

Usage:
    python search_embeddings.py --index rag_index_dictabert.npz
    python search_embeddings.py --index rag_index_dictabert.npz --top-k 10
"""

import argparse
import threading
import tkinter as tk
from tkinter import font as tkfont
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = 'intfloat/multilingual-e5-large'
TOP_K = 7


# ── embedder ──────────────────────────────────────────────────────────────────

class Embedder:
    def __init__(self):
        print(f'Loading {EMBED_MODEL}…')
        self.model = SentenceTransformer(EMBED_MODEL)

    def encode(self, texts):
        # multilingual-e5 requires "query: " prefix for search queries
        prefixed = [f'query: {t}' for t in texts]
        return self.model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )


# ── index ─────────────────────────────────────────────────────────────────────

def load_index(path):
    print(f'Loading index from {path}…')
    data = np.load(path, allow_pickle=True)
    vecs = data['vecs']
    metadata = data['metadata'].tolist()
    print(f'Loaded {len(metadata)} chunks.')
    return vecs, metadata


# ── search ────────────────────────────────────────────────────────────────────

def search(query, vecs, metadata, embedder, k=TOP_K):
    q = embedder.encode([query])[0].astype(np.float32)
    sims = vecs @ q
    top = np.argsort(sims)[::-1][:k]
    return [(metadata[i], float(sims[i])) for i in top]


# ── GUI ───────────────────────────────────────────────────────────────────────

class SearchApp:
    BG       = '#1e1e2e'
    BG_CARD  = '#2a2a3e'
    FG       = '#cdd6f4'
    FG_DIM   = '#6c7086'
    ACCENT   = '#89b4fa'
    GREEN    = '#a6e3a1'
    YELLOW   = '#f9e2af'
    RED      = '#f38ba8'
    FONT     = 'Arial'

    def __init__(self, root, vecs, metadata, embedder, top_k):
        self.root     = root
        self.vecs     = vecs
        self.metadata = metadata
        self.embedder = embedder
        self.top_k    = top_k

        root.title('Knesset Embedding Search')
        root.configure(bg=self.BG)
        root.geometry('900x700')
        root.minsize(700, 400)

        self._build_ui()

    def _build_ui(self):
        # ── top bar ───────────────────────────────────────────────────────
        top = tk.Frame(self.root, bg=self.BG, pady=10, padx=12)
        top.pack(fill='x')

        self.query_var = tk.StringVar()
        entry = tk.Entry(
            top, textvariable=self.query_var,
            font=(self.FONT, 14), bg=self.BG_CARD, fg=self.FG,
            insertbackground=self.FG, relief='flat',
            highlightthickness=1, highlightcolor=self.ACCENT,
            highlightbackground=self.FG_DIM,
        )
        entry.pack(side='left', fill='x', expand=True, ipady=6, padx=(0, 8))
        entry.bind('<Return>', lambda _: self._do_search())
        entry.focus()

        paste_btn = tk.Button(
            top, text='Paste', command=lambda: (
                self.query_var.set(self.root.clipboard_get()),
            ),
            font=(self.FONT, 11), bg=self.BG_CARD, fg=self.FG,
            relief='flat', padx=10, cursor='hand2',
            activebackground=self.FG_DIM, activeforeground=self.FG,
        )
        paste_btn.pack(side='left', padx=(0, 6))

        btn = tk.Button(
            top, text='Search', command=self._do_search,
            font=(self.FONT, 13, 'bold'), bg=self.ACCENT, fg=self.BG,
            relief='flat', padx=14, cursor='hand2',
            activebackground='#74a0e0', activeforeground=self.BG,
        )
        btn.pack(side='left')

        self.status_var = tk.StringVar(value=f'{len(self.metadata):,} chunks loaded')
        tk.Label(top, textvariable=self.status_var,
                 bg=self.BG, fg=self.FG_DIM,
                 font=(self.FONT, 10)).pack(side='right')

        # ── scrollable results area ───────────────────────────────────────
        container = tk.Frame(self.root, bg=self.BG)
        container.pack(fill='both', expand=True, padx=12, pady=(0, 12))

        self.canvas = tk.Canvas(container, bg=self.BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient='vertical',
                                 command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)

        self.results_frame = tk.Frame(self.canvas, bg=self.BG)
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.results_frame, anchor='nw')

        self.results_frame.bind('<Configure>', self._on_frame_configure)
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        self.canvas.bind_all('<MouseWheel>',
                             lambda e: self.canvas.yview_scroll(-1 * (e.delta // 120), 'units'))

    def _on_frame_configure(self, _):
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def _on_canvas_configure(self, e):
        self.canvas.itemconfig(self.canvas_window, width=e.width)

    def _do_search(self):
        query = self.query_var.get().strip()
        if not query:
            return
        self.status_var.set('Searching…')
        self.root.update_idletasks()

        def run():
            hits = search(query, self.vecs, self.metadata, self.embedder, self.top_k)
            self.root.after(0, lambda: self._show_results(query, hits))

        threading.Thread(target=run, daemon=True).start()

    def _show_results(self, query, hits):
        for w in self.results_frame.winfo_children():
            w.destroy()

        self.status_var.set(f'Top {len(hits)} results for: {query}')
        self.canvas.yview_moveto(0)

        for rank, (meta, sim) in enumerate(hits, 1):
            self._add_card(rank, meta, sim)

    def _add_card(self, rank, meta, sim):
        title    = meta.get('title') or meta.get('doc_id', '(ללא שם)')
        date     = meta.get('date', '')
        from_u   = meta.get('from', '')
        to_u     = meta.get('to', '')
        url      = meta.get('url', '')
        utts     = meta.get('utterances', [])

        sim_color = self.GREEN if sim > 0.8 else self.YELLOW if sim > 0.6 else self.RED

        card = tk.Frame(self.results_frame, bg=self.BG_CARD,
                        pady=10, padx=12, bd=0)
        card.pack(fill='x', pady=(0, 8))

        # header row: rank + score + date + range
        hdr = tk.Frame(card, bg=self.BG_CARD)
        hdr.pack(fill='x')

        tk.Label(hdr, text=f'#{rank}', font=(self.FONT, 11, 'bold'),
                 bg=self.BG_CARD, fg=self.ACCENT).pack(side='left')
        tk.Label(hdr, text=f'  sim={sim:.3f}', font=(self.FONT, 11, 'bold'),
                 bg=self.BG_CARD, fg=sim_color).pack(side='left')
        tk.Label(hdr, text=f'  {date}  utterances {from_u}–{to_u}',
                 font=(self.FONT, 10), bg=self.BG_CARD, fg=self.FG_DIM).pack(side='left')

        # title (Hebrew, RTL)
        tk.Label(card, text=title, font=(self.FONT, 12, 'bold'),
                 bg=self.BG_CARD, fg=self.FG,
                 justify='right', anchor='e', wraplength=820).pack(fill='x', pady=(4, 0))

        if url:
            url_lbl = tk.Label(card, text=url, font=(self.FONT, 9),
                               bg=self.BG_CARD, fg='#89dceb',
                               cursor='hand2', anchor='w')
            url_lbl.pack(fill='x')
            url_lbl.bind('<Button-1>', lambda e, u=url: self._open_url(u))

        # divider
        tk.Frame(card, bg=self.FG_DIM, height=1).pack(fill='x', pady=6)

        # utterances
        for u in utts:
            speaker = u.get('speaker', '').strip()
            text    = u.get('text', '').replace('\n', ' ').strip()

            utt_frame = tk.Frame(card, bg=self.BG_CARD)
            utt_frame.pack(fill='x', pady=2)

            tk.Label(utt_frame, text=speaker, font=(self.FONT, 10, 'bold'),
                     bg=self.BG_CARD, fg=self.ACCENT,
                     anchor='e', justify='right').pack(fill='x')

            tk.Label(utt_frame, text=text, font=(self.FONT, 11),
                     bg=self.BG_CARD, fg=self.FG,
                     justify='right', anchor='e', wraplength=840).pack(fill='x')

    def _open_url(self, url):
        import webbrowser
        webbrowser.open(url)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default='rag_index_dictabert.npz')
    parser.add_argument('--top-k', type=int, default=TOP_K)
    args = parser.parse_args()

    vecs, metadata = load_index(args.index)
    embedder = Embedder()

    root = tk.Tk()
    SearchApp(root, vecs, metadata, embedder, args.top_k)
    root.mainloop()


if __name__ == '__main__':
    main()
