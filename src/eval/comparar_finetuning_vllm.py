#!/usr/bin/env python
"""Compara las respuestas del Custom Job por lotes contra las del endpoint vLLM.

Lo que se quiere verificar es que el endpoint sirve EL MISMO modelo que se
midio, no que devuelva los mismos caracteres. Son cosas distintas y conviene no
confundirlas:

- La coincidencia literal es baja por construccion. vLLM y transformers usan
  kernels distintos y agrupan las secuencias de forma distinta, asi que en bf16
  un empate de logits en el primer token manda la respuesta por otro camino. En
  preguntas donde el modelo no recuerda la cifra exacta y esta eligiendo entre
  continuaciones igual de plausibles, eso ocurre casi siempre. No es un
  sintoma de nada.
- Lo que si tiene que coincidir es el ROUGE por tecnica. Si el endpoint
  estuviera cargando el modelo base, o los adaptadores sin fusionar, o el
  modelo fusionado a medias, las metricas caerian visiblemente. Que coincidan
  en el tercer decimal es la evidencia real de que el despliegue sirve lo
  mismo que se midio.

Por eso el veredicto de este script sale del ROUGE, no de la identidad textual.

Uso:
    python src/eval/comparar_finetuning_vllm.py
"""
import argparse
import difflib
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import ROUGE_TIPOS, es_valida, limpiar_prediccion, rouge  # noqa: E402

# Umbral del veredicto. Las diferencias observadas entre los dos caminos estan
# en el rango de 0.001 a 0.015; 0.03 deja margen de sobra para el ruido de
# decodificacion y sigue siendo estrecho para detectar otro modelo, que moveria
# el ROUGE en decimas.
TOLERANCIA = 0.03


def leer(ruta):
    return [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]


def por_tecnica(filas):
    grupos = {}
    for f in filas:
        grupos.setdefault(f["technique"], []).append(f)
    tabla = {}
    for tecnica, rs in sorted(grupos.items()):
        validas = [r for r in rs if es_valida(r["prediction"])]
        puntajes = [rouge(limpiar_prediccion(r["prediction"], tecnica), r["reference"])
                    for r in validas]
        tabla[tecnica] = {
            "n": len(rs),
            "n_validas": len(validas),
            "chars": round(sum(len(r["prediction"]) for r in validas) / len(validas))
                     if validas else 0,
            **{t: round(sum(p[t] for p in puntajes) / len(puntajes), 4) if puntajes else 0.0
               for t in ROUGE_TIPOS},
        }
    return tabla


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lotes", default="results/respuestas-finetuning.jsonl")
    ap.add_argument("--endpoint", default="results/respuestas-finetuning-vllm.jsonl")
    ap.add_argument("--mostrar", type=int, default=2,
                    help="Cuantas respuestas divergentes imprimir, como muestra")
    ap.add_argument("--tolerancia", type=float, default=TOLERANCIA)
    args = ap.parse_args()

    filas_a, filas_b = leer(RAIZ / args.lotes), leer(RAIZ / args.endpoint)
    a, b = por_tecnica(filas_a), por_tecnica(filas_b)

    print("=== Metricas por tecnica (esto es lo que decide) ===\n")
    cab = f"{'tecnica':<18}{'fuente':<10}{'n':>4}{'ROUGE-1':>10}{'ROUGE-2':>10}{'ROUGE-L':>10}{'chars':>8}"
    print(cab)
    print("-" * len(cab))
    desvios = []
    for tecnica in sorted(set(a) & set(b)):
        for etiqueta, tabla in (("lotes", a), ("endpoint", b)):
            f = tabla[tecnica]
            print(f"{tecnica:<18}{etiqueta:<10}{f['n_validas']:>4}"
                  f"{f['rouge1']:>10.4f}{f['rouge2']:>10.4f}{f['rougeL']:>10.4f}{f['chars']:>8}")
        peor = max(abs(a[tecnica][t] - b[tecnica][t]) for t in ROUGE_TIPOS)
        desvios.append((tecnica, peor))
        print(f"{'':<28}{'delta maximo':>10} {peor:.4f}\n")

    # --- Coincidencia literal, como contexto y no como veredicto ---
    ia = {(f["technique"], f["question"]): (f["prediction"] or "").strip() for f in filas_a}
    ib = {(f["technique"], f["question"]): (f["prediction"] or "").strip() for f in filas_b}
    comunes = sorted(set(ia) & set(ib))
    iguales = sum(1 for k in comunes if ia[k] == ib[k])
    divergentes = sorted(
        ((k, difflib.SequenceMatcher(None, ia[k], ib[k]).ratio()) for k in comunes if ia[k] != ib[k]),
        key=lambda x: x[1])

    print("=== Coincidencia literal (contexto, no veredicto) ===")
    print(f"  identicas   : {iguales}/{len(comunes)}")
    print(f"  divergentes : {len(divergentes)}  (esperable: decodificacion en bf16)")
    for (tecnica, pregunta), ratio in divergentes[:args.mostrar]:
        print(f"\n  --- {tecnica} | similitud {ratio:.2f} ---\n  {pregunta}")
        print(f"    lotes    : {ia[(tecnica, pregunta)][:180]}")
        print(f"    endpoint : {ib[(tecnica, pregunta)][:180]}")

    peor_global = max(d for _, d in desvios)
    print(f"\n=== Veredicto ===")
    if peor_global <= args.tolerancia:
        print(f"El endpoint sirve el mismo modelo. Desvio maximo de ROUGE: "
              f"{peor_global:.4f} (tolerancia {args.tolerancia}).")
        return 0
    peor_tec = max(desvios, key=lambda x: x[1])[0]
    print(f"REVISAR: el ROUGE se desvia {peor_global:.4f} en '{peor_tec}', por encima de "
          f"la tolerancia {args.tolerancia}. Comprueba que el endpoint apunte al "
          f"modelo fusionado y no al base.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
