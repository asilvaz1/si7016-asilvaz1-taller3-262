#!/usr/bin/env python
"""Construye el indice FAISS (denso) sobre los chunks del corpus.

Uso:
    python src/rag/index.py
    python src/rag/index.py --model intfloat/multilingual-e5-small

Deja en data/index/: faiss.index, chunks.jsonl y meta.json. El indice BM25 no
se persiste porque se reconstruye en segundos al cargar el retriever.
"""
import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from chunking import CHUNK_OVERLAP, CHUNK_SIZE, RAIZ, hacer_chunks

INDEX_DIR = RAIZ / "data" / "index"
MODELO_DEFAULT = "intfloat/multilingual-e5-base"


def prefijos(modelo: str):
    """Los modelos E5 exigen 'query: ' / 'passage: '; sin ellos rinden peor."""
    if "e5" in modelo.lower():
        return "query: ", "passage: "
    return "", ""


def texto_a_embeber(chunk: dict) -> str:
    # La norma va como encabezado: las preguntas nombran la norma ("ISO 3746")
    # y el chunk suelto muchas veces no la menciona.
    return f"{chunk['norma']}. {chunk['text']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODELO_DEFAULT)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    chunks, faltan = hacer_chunks()
    if not chunks:
        raise SystemExit("No hay texto en data/processed/. Ver HANDOFF seccion 5.")
    docs = sorted({c["source_file"] for c in chunks})
    print(f"{len(chunks)} chunks de {len(docs)} normas fuente")
    if faltan:
        print(f"AVISO: {len(faltan)} normas fuente sin texto en data/processed/ "
              "(quedan fuera del indice):")
        for f in faltan:
            print(f"   - {f}")

    _, pref_p = prefijos(args.model)
    modelo = SentenceTransformer(args.model)
    emb = modelo.encode([pref_p + texto_a_embeber(c) for c in chunks],
                        batch_size=args.batch, normalize_embeddings=True,
                        show_progress_bar=True).astype("float32")

    # Vectores normalizados: producto interno == similitud coseno.
    indice = faiss.IndexFlatIP(emb.shape[1])
    indice.add(emb)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(indice, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (INDEX_DIR / "meta.json").write_text(json.dumps({
        "model": args.model, "dim": int(emb.shape[1]), "n_chunks": len(chunks),
        "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
        "documentos": docs, "fuente_sin_texto": faltan,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Indice guardado en {INDEX_DIR.relative_to(RAIZ)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
