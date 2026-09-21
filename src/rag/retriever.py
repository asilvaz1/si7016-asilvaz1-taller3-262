#!/usr/bin/env python
"""Recuperacion sobre el indice de data/index/: densa, BM25 e hibrida.

- dense:  FAISS con embeddings de sentence-transformers.
- bm25:   busqueda lexica; ayuda con codigos y simbolos (K1, LAeq, 7.3).
- hybrid: fusion por Reciprocal Rank Fusion de las dos listas anteriores.
- rerank: opcional, un CrossEncoder reordena los mejores candidatos.

Uso rapido:
    python src/rag/retriever.py "What background noise criterion does ISO 3746 impose?"
"""
import argparse
import json
import re
import sys

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from index import INDEX_DIR, prefijos

MODOS = ("dense", "bm25", "hybrid")
RERANKER_DEFAULT = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60  # constante estandar de RRF


def tokenizar(texto: str):
    return re.findall(r"\w+", texto.lower())


class Retriever:
    def __init__(self, reranker: str | None = None):
        self.meta = json.loads((INDEX_DIR / "meta.json").read_text(encoding="utf-8"))
        self.chunks = [json.loads(l) for l in
                       (INDEX_DIR / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        self.faiss = faiss.read_index(str(INDEX_DIR / "faiss.index"))
        self.modelo = SentenceTransformer(self.meta["model"])
        self.pref_q, _ = prefijos(self.meta["model"])
        self.bm25 = BM25Okapi([tokenizar(f"{c['norma']} {c['text']}") for c in self.chunks])
        self.reranker = CrossEncoder(reranker) if reranker else None

    def _dense(self, q: str, n: int):
        v = self.modelo.encode([self.pref_q + q], normalize_embeddings=True).astype("float32")
        _, ids = self.faiss.search(v, n)
        return [int(i) for i in ids[0] if i >= 0]

    def _bm25(self, q: str, n: int):
        puntajes = self.bm25.get_scores(tokenizar(q))
        return sorted(range(len(puntajes)), key=lambda i: -puntajes[i])[:n]

    def search(self, q: str, k: int = 5, modo: str = "hybrid", candidatos: int = 20):
        """Devuelve k chunks (dict del chunk + 'rank' y 'score') en orden de
        relevancia. Con reranker se reordenan los `candidatos` mejores."""
        n = max(k, candidatos) if self.reranker else k
        if modo == "dense":
            ids = self._dense(q, n)
        elif modo == "bm25":
            ids = self._bm25(q, n)
        elif modo == "hybrid":
            fusion = {}
            for lista in (self._dense(q, 50), self._bm25(q, 50)):
                for pos, i in enumerate(lista):
                    fusion[i] = fusion.get(i, 0.0) + 1.0 / (RRF_K + pos + 1)
            ids = sorted(fusion, key=lambda i: -fusion[i])[:n]
        else:
            raise ValueError(f"modo desconocido: {modo}")

        if self.reranker:
            puntajes = self.reranker.predict([(q, self.chunks[i]["text"]) for i in ids])
            orden = sorted(zip(ids, puntajes), key=lambda t: -t[1])[:k]
            ids, scores = [i for i, _ in orden], [float(s) for _, s in orden]
        else:
            scores = [None] * len(ids)

        return [{**self.chunks[i], "rank": r + 1, "score": s}
                for r, (i, s) in enumerate(zip(ids, scores))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--mode", choices=MODOS, default="dense")
    ap.add_argument("--rerank", action="store_true")
    args = ap.parse_args()
    r = Retriever(RERANKER_DEFAULT if args.rerank else None)
    for h in r.search(args.query, args.k, args.mode):
        print(f"\n#{h['rank']}  {h['chunk_id']}")
        print(h["text"][:400].replace("\n", " "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
