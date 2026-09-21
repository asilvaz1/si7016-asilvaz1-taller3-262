#!/usr/bin/env python
"""Consolida las respuestas de los tres sistemas en una tabla comparativa.

Lee results/respuestas-{base,finetuning,rag}.jsonl (esquema del HANDOFF, 4.2),
calcula ROUGE-1/2/L y, para el RAG, Recall@K, y escribe una fila por
combinacion de system y technique en results/metricas-comparativas.csv.

Uso:
    python src/eval/run_eval.py
    python src/eval/run_eval.py --results-dir <carpeta> --detalle
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from metrics import ROUGE_TIPOS, es_valida, limpiar_prediccion, recall_at_k, rouge

RAIZ = Path(__file__).resolve().parents[2]
SISTEMAS = ("base", "finetuning", "rag")
KS = (1, 3, 5)


def leer_jsonl(ruta: Path):
    return [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]


def media(valores):
    return round(sum(valores) / len(valores), 4) if valores else ""


def evaluar_fila(fila: dict) -> dict:
    """Metricas de una respuesta. Las invalidas quedan sin puntuar."""
    res = {"system": fila["system"], "technique": fila["technique"],
           "question": fila["question"], "valida": es_valida(fila["prediction"])}
    if res["valida"]:
        pred = limpiar_prediccion(fila["prediction"], fila["technique"])
        res.update(rouge(pred, fila["reference"]))
    # Recall@K depende solo de la recuperacion, no de que la generacion salga bien.
    if fila.get("retrieved_sources"):
        for k in KS:
            res[f"recall@{k}"] = recall_at_k(fila["source_file"], fila["retrieved_sources"], k)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(RAIZ / "results"))
    ap.add_argument("--eval-set", default=str(RAIZ / "data" / "qa" / "normas_ruido_eval.jsonl"))
    ap.add_argument("--out", default=None, help="default: <results-dir>/metricas-comparativas.csv")
    ap.add_argument("--detalle", action="store_true",
                    help="Escribe tambien metricas-por-pregunta.csv")
    args = ap.parse_args()

    carpeta = Path(args.results_dir)
    esperadas = {d["question"] for d in leer_jsonl(Path(args.eval_set))}

    evaluadas = []
    for sistema in SISTEMAS:
        ruta = carpeta / f"respuestas-{sistema}.jsonl"
        if not ruta.exists():
            print(f"AVISO: falta {ruta.name}, se omite el sistema '{sistema}'")
            continue
        filas = leer_jsonl(ruta)
        ajenas = {f["question"] for f in filas} - esperadas
        if ajenas:
            print(f"AVISO: {ruta.name} tiene {len(ajenas)} preguntas que no estan en el eval set")
        evaluadas += [evaluar_fila(f) for f in filas]

    if not evaluadas:
        raise SystemExit("No hay respuestas que evaluar.")

    grupos = {}
    for r in evaluadas:
        grupos.setdefault((r["system"], r["technique"]), []).append(r)

    cols = ["system", "technique", "n", "n_validas",
            *ROUGE_TIPOS, *(f"recall@{k}" for k in KS)]
    tabla = []
    for (sistema, tecnica), rs in sorted(grupos.items()):
        validas = [r for r in rs if r["valida"]]
        fila = {"system": sistema, "technique": tecnica, "n": len(rs), "n_validas": len(validas)}
        for t in ROUGE_TIPOS:
            fila[t] = media([r[t] for r in validas])
        for k in KS:
            fila[f"recall@{k}"] = media([r[f"recall@{k}"] for r in rs if f"recall@{k}" in r])
        tabla.append(fila)

    destino = Path(args.out) if args.out else carpeta / "metricas-comparativas.csv"
    with open(destino, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(tabla)

    print("  ".join(f"{c:>16}" for c in cols))
    for fila in tabla:
        print("  ".join(f"{str(fila[c]):>16}" for c in cols))
    print(f"\n-> {destino}")

    if args.detalle:
        det = carpeta / "metricas-por-pregunta.csv"
        campos = ["system", "technique", "question", "valida", *ROUGE_TIPOS,
                  *(f"recall@{k}" for k in KS)]
        with open(det, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
            w.writeheader()
            w.writerows(evaluadas)
        print(f"-> {det}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
