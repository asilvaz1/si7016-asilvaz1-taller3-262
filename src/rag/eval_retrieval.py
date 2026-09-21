#!/usr/bin/env python
"""Recall@K y MRR de la recuperacion, por modo (dense, bm25, hybrid).

Recall@K a nivel de documento: acierta si el source_file de la pregunta esta
entre los source_file de los primeros K chunks recuperados.

Solo se evaluan las preguntas cuya norma fuente esta en el indice; las demas
no tienen forma de acertar y meterlas solo diluiria la metrica. El resumen
dice cuantas quedaron fuera.

Uso:
    python src/rag/eval_retrieval.py                       # las 20 de eval
    python src/rag/eval_retrieval.py --dataset data/qa/normas_ruido_all.jsonl
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from chunking import RAIZ
from retriever import MODOS, RERANKER_DEFAULT, Retriever

KS = (1, 3, 5, 10)


def cargar(ruta):
    return [json.loads(l) for l in Path(ruta).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(RAIZ / "data" / "qa" / "normas_ruido_eval.jsonl"))
    ap.add_argument("--out", default=str(RAIZ / "results" / "recall-retrieval.csv"))
    ap.add_argument("--rerank", action="store_true",
                    help="Agrega una fila por modo con reranker CrossEncoder")
    args = ap.parse_args()

    filas = cargar(args.dataset)
    r = Retriever()
    en_indice = set(r.meta["documentos"])
    evaluables = [d for d in filas if d["source_file"] in en_indice]
    print(f"{len(filas)} preguntas en el dataset; {len(evaluables)} tienen su norma "
          f"en el indice ({len(en_indice)} normas indexadas)")
    if not evaluables:
        raise SystemExit("Ninguna pregunta es evaluable con el indice actual.")

    variantes = [(m, False) for m in MODOS]
    if args.rerank:
        rr = Retriever(RERANKER_DEFAULT)
        variantes += [(m, True) for m in MODOS]

    resultados = []
    kmax = max(KS)
    for modo, con_rerank in variantes:
        ret = rr if con_rerank else r
        aciertos = {k: 0 for k in KS}
        rr_suma = 0.0
        for d in evaluables:
            fuentes = [h["source_file"] for h in ret.search(d["question"], kmax, modo)]
            for k in KS:
                aciertos[k] += d["source_file"] in fuentes[:k]
            if d["source_file"] in fuentes:
                rr_suma += 1.0 / (fuentes.index(d["source_file"]) + 1)
        n = len(evaluables)
        resultados.append({
            "modo": modo + ("+rerank" if con_rerank else ""),
            **{f"recall@{k}": round(aciertos[k] / n, 3) for k in KS},
            "mrr": round(rr_suma / n, 3),
            "n_preguntas": n,
            "n_normas_indexadas": len(en_indice),
        })

    cols = list(resultados[0])
    print("\n" + "  ".join(f"{c:>14}" for c in cols))
    for x in resultados:
        print("  ".join(f"{str(x[c]):>14}" for c in cols))
    print(f"\nAzar a nivel de documento con {len(en_indice)} normas: "
          f"recall@1 ~ {1/len(en_indice):.2f}")

    destino = Path(args.out)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(resultados)
    print(f"-> {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
