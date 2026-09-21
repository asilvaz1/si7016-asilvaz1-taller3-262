"""
validar_respuestas.py - Comprueba que un jsonl de respuestas cumpla el esquema
de la seccion 4.2 del HANDOFF antes de darlo por bueno.

Existe porque un archivo con 60 filas puede estar mal de varias formas que no se
ven a simple vista: una prediccion vacia, una fila con el campo 'system'
equivocado, una respuesta que quedo como texto de error, o preguntas que no
coinciden con las del eval set. Cualquiera de esas envenena ROUGE sin avisar.

Uso (desde 13_nlp, con el .venv activado):
    python .\\si7016-asilvaz1-taller3-262\\src\\finetuning\\validar_respuestas.py `
        --archivo results\\respuestas-finetuning.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

CAMPOS = ["question", "reference", "prediction", "prediction_raw",
          "technique", "system", "source_file", "section_label"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--archivo", default="results/respuestas-finetuning.jsonl")
    p.add_argument("--eval_set", default="data/qa/normas_ruido_eval.jsonl",
                   help="Para comprobar que las preguntas sean las mismas")
    p.add_argument("--esperadas", type=int, default=60)
    return p.parse_args()


def leer(ruta):
    filas = []
    with open(ruta, encoding="utf-8") as f:
        for n, linea in enumerate(f, 1):
            linea = linea.strip()
            if not linea:
                continue
            try:
                filas.append(json.loads(linea))
            except json.JSONDecodeError as e:
                raise SystemExit(f"Linea {n} de {ruta} no es JSON valido: {e}")
    return filas


def main():
    args = parse_args()
    ruta = Path(args.archivo)
    if not ruta.exists():
        raise SystemExit(f"No existe {ruta}. Bajalo del bucket primero.")

    filas = leer(ruta)
    problemas = []

    print(f"Archivo : {ruta}")
    print(f"Filas   : {len(filas)}")
    if len(filas) != args.esperadas:
        problemas.append(f"se esperaban {args.esperadas} filas y hay {len(filas)}")

    faltantes = Counter()
    vacias = 0
    errores = 0
    for x in filas:
        for c in CAMPOS:
            if c not in x:
                faltantes[c] += 1
        pred = x.get("prediction", "")
        if not pred.strip():
            vacias += 1
        elif pred.startswith("__ERROR__"):
            errores += 1

    if faltantes:
        problemas.append("faltan campos: " +
                         ", ".join(f"{c} en {n} filas" for c, n in faltantes.items()))
    if vacias:
        problemas.append(f"{vacias} predicciones vacias")
    if errores:
        problemas.append(f"{errores} predicciones son mensajes de error")

    por_tecnica = Counter(x.get("technique") for x in filas)
    print("Por tecnica:")
    for t, n in sorted(por_tecnica.items()):
        largo = [len(x["prediction"]) for x in filas if x.get("technique") == t]
        media = sum(largo) / max(1, len(largo))
        print(f"  {t:<18} {n:>3} filas, largo medio {media:,.0f} caracteres")

    sistemas = set(x.get("system") for x in filas)
    print(f"Sistemas: {sistemas}")
    if len(sistemas) != 1:
        problemas.append(f"hay mas de un valor de 'system': {sistemas}")

    # Las preguntas tienen que ser exactamente las del eval set, o ROUGE estaria
    # comparando contra referencias de otras preguntas.
    eval_path = Path(args.eval_set)
    if eval_path.exists():
        esperadas = {x["question"] for x in leer(eval_path)}
        vistas = {x.get("question", "") for x in filas}
        sobran = vistas - esperadas
        faltan = esperadas - vistas
        if sobran:
            problemas.append(f"{len(sobran)} preguntas que no estan en el eval set")
        if faltan:
            problemas.append(f"{len(faltan)} preguntas del eval set sin responder")
        if not sobran and not faltan:
            print(f"Preguntas: las {len(esperadas)} del eval set, completas")
    else:
        print(f"Aviso: no encontre {eval_path}, no se comprobaron las preguntas")

    print()
    if problemas:
        print("PROBLEMAS:")
        for p in problemas:
            print(f"  - {p}")
        raise SystemExit(1)
    print("Todo bien: el archivo cumple el esquema del HANDOFF 4.2.")


if __name__ == "__main__":
    main()
