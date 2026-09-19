#!/usr/bin/env python3
"""Verificacion de anclaje: cada numero citado en una respuesta debe aparecer
en el texto extraido de la norma que la respuesta declara como fuente.

Detecta numeros inventados, que es el riesgo principal de un dataset QA
redactado sobre normas tecnicas. Los numeros se buscan en las dos notaciones
decimales (coma y punto) porque el corpus mezcla UNE en espanol con ISO/BS en
ingles.

Uso:  python src/data/verify_qa_grounding.py
"""
import json
import re
import sys
from pathlib import Path

PROCESSED = Path("data/processed")
DATASET = Path("data/qa/normas_ruido_all.jsonl")
# numeros triviales o de contexto que no vale la pena rastrear
IGNORAR = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "0"}


def slug(nombre: str) -> str:
    s = re.sub(r"\.pdf$", "", nombre, flags=re.I)
    s = re.sub(r"[^\w\s.-]", "", s)
    return re.sub(r"[\s_]+", "_", s.strip()).lower()


def variantes(num: str):
    """Formas en que el mismo numero puede aparecer en el PDF extraido."""
    v = {num, num.replace(",", "."), num.replace(".", ","),
         num.replace(" ", ""), num.replace(" ", " ")}
    if " " in num:                      # 101,325 vs 101 325 / 10 000 vs 10000
        v.add(num.replace(" ", ""))
    return {x for x in v if x}


def main() -> int:
    cache = {}
    filas = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    sin_anclar, revisados = [], 0

    for d in filas:
        key = slug(d["source_file"])
        if key not in cache:
            p = PROCESSED / f"{key}.txt"
            if not p.exists():
                sin_anclar.append((d["question"], "SIN TEXTO FUENTE", d["source_file"]))
                continue
            t = p.read_text(encoding="utf-8")
            cache[key] = re.sub(r"[ \t]+", " ", t)
        texto = cache[key]

        # numeros con decimal, con separador de miles, o enteros de 2+ cifras
        nums = re.findall(r"\d{1,3}(?:[ ]\d{3})+|\d+[.,]\d+|\b\d{2,}\b", d["answer"])
        for n in nums:
            if n in IGNORAR:
                continue
            revisados += 1
            if not any(v in texto for v in variantes(n)):
                sin_anclar.append((d["question"], n, d["source_file"]))

    print(f"Pares QA: {len(filas)}   numeros verificados: {revisados}")
    if sin_anclar:
        print(f"\n*** {len(sin_anclar)} numero(s) sin coincidencia en el texto fuente ***")
        print("(revisar a mano: puede ser un error de extraccion del PDF, "
              "una cifra escrita de otra forma, o un dato inventado)\n")
        for q, n, sf in sin_anclar:
            print(f"  [{n}]  {sf}\n      {q[:95]}")
        return 1
    print("Todos los numeros citados aparecen en el texto de su norma fuente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
