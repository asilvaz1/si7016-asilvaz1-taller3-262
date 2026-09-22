#!/usr/bin/env python
"""Describe COMO responde cada sistema, no solo que tan bien puntua.

ROUGE dice cuanto se parece una respuesta a su referencia, pero no por que. Este
script mide tres rasgos de forma que explican buena parte de las diferencias de
la tabla comparativa:

  - **Idioma.** El corpus mezcla ediciones en ingles (ISO, BS) y en espanol
    (UNE). Una respuesta que cita literalmente un fragmento de la edicion
    espanola puntua casi cero contra una referencia en ingles, por correcta que
    sea. Esto no es un defecto del sistema: es el precio de anclar la respuesta
    en el texto recuperado.
  - **Longitud.** ROUGE es F-measure, asi que penaliza tanto quedarse corto como
    irse largo. Una respuesta del doble de la referencia pierde precision
    aunque contenga la respuesta correcta.
  - **Citas literales.** Comillas y marcadores tipo [1] son la huella de un
    prompt que pide citar. Suben la fidelidad y bajan el parecido con una
    referencia redactada.

Uso:
    python src/eval/analizar_respuestas.py
    python src/eval/analizar_respuestas.py --sistemas rag rag-vertex
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]

# Palabras funcionales, no terminologia: se busca el idioma en que esta escrita
# la frase, no el tema del que habla.
ES = re.compile(r"\b(el|la|los|las|de|que|debe|deben|se|para|con|segun|según|"
                r"norma|medicion|medición|nivel|sonido|ruido|una|del)\b", re.I)
EN = re.compile(r"\b(the|of|shall|is|are|and|to|for|with|level|noise|sound|"
                r"measurement|standard|a|in)\b", re.I)


def en_espanol(texto):
    return len(ES.findall(texto or "")) > len(EN.findall(texto or ""))


def tiene_cita(texto):
    return bool(re.search(r'\[\d+\]|"', texto or ""))


def analizar(filas):
    grupos = defaultdict(list)
    for f in filas:
        grupos[f["technique"]].append(f)
    tabla = []
    for tecnica, rs in sorted(grupos.items()):
        preds = [r["prediction"] or "" for r in rs]
        tabla.append({
            "technique": tecnica,
            "n": len(rs),
            "espanol": sum(1 for p in preds if en_espanol(p)),
            "largo_medio": round(sum(len(p) for p in preds) / len(preds)),
            "con_citas": sum(1 for p in preds if tiene_cita(p)),
        })
    return tabla


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sistemas", nargs="*",
                    default=["base", "finetuning", "rag", "rag-vertex"])
    ap.add_argument("--results-dir", default=str(RAIZ / "results"))
    args = ap.parse_args()
    carpeta = Path(args.results_dir)

    ruta_eval = RAIZ / "data" / "qa" / "normas_ruido_eval.jsonl"
    refs = [json.loads(l)["answer"] for l in
            ruta_eval.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Referencias: {len(refs)}, largo medio {sum(map(len, refs)) // len(refs)} "
          f"caracteres, {sum(1 for r in refs if en_espanol(r))} en espanol\n")

    cab = f"{'sistema':<12}{'tecnica':<18}{'en espanol':>12}{'largo medio':>13}{'con citas':>11}"
    print(cab)
    print("-" * len(cab))
    for sistema in args.sistemas:
        ruta = carpeta / f"respuestas-{sistema}.jsonl"
        if not ruta.exists():
            print(f"{sistema:<12}(falta {ruta.name})")
            continue
        filas = [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]
        for f in analizar(filas):
            print(f"{sistema:<12}{f['technique']:<18}"
                  f"{str(f['espanol']) + '/' + str(f['n']):>12}"
                  f"{f['largo_medio']:>13}"
                  f"{str(f['con_citas']) + '/' + str(f['n']):>11}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
