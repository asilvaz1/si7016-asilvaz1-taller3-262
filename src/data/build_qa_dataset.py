#!/usr/bin/env python3
"""Combina, valida y divide el dataset QA del taller 3.

Lee los .jsonl por categoria en data/qa/parts/, valida el esquema y la
trazabilidad contra data/processed/, y escribe:
  data/qa/normas_ruido_all.jsonl
  data/qa/normas_ruido_train.jsonl   (~80 %)
  data/qa/normas_ruido_eval.jsonl    (~20 %)
  data/qa/cobertura.csv              (preguntas por norma y categoria)

Uso:  python src/data/build_qa_dataset.py [--seed 42]
"""
import argparse
import csv
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

CAMPOS = ["question", "answer", "source_file", "source_doc_title",
          "section_kind", "section_label"]
PARTS = Path("data/qa/parts")
PROCESSED = Path("data/processed")
OUT = Path("data/qa")

CATEGORIAS = {
    "cat01": "Ruido ambiental: medicion, descriptores y limites",
    "cat02": "Propagacion del sonido al aire libre",
    "cat03": "Potencia sonora de fuentes",
    "cat04": "Ruido industrial y evaluacion de molestia",
    "cat05": "Aislamiento acustico en edificaciones",
    "cat06": "Acustica de salas y reverberacion",
    "cat07": "Exposicion ocupacional y proteccion auditiva",
    "cat08": "Modelos de calculo: eventos impulsivos y ruido de transito",
}


def slug(nombre: str) -> str:
    s = re.sub(r"\.pdf$", "", nombre, flags=re.I)
    s = re.sub(r"[^\w\s.-]", "", s)
    s = re.sub(r"[\s_]+", "_", s.strip())
    return s.lower()


def norm_txt(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\W+", " ", s).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-frac", type=float, default=0.20)
    args = ap.parse_args()

    archivos = sorted(PARTS.glob("cat*.jsonl"))
    if not archivos:
        print("No hay archivos en data/qa/parts/", file=sys.stderr)
        return 1

    filas, errores = [], []
    for f in archivos:
        cat = f.name[:5]
        for i, linea in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not linea.strip():
                continue
            try:
                d = json.loads(linea)
            except json.JSONDecodeError as e:
                errores.append(f"{f.name}:{i} JSON invalido: {e}")
                continue
            faltan = [c for c in CAMPOS if c not in d or not str(d[c]).strip()]
            if faltan:
                errores.append(f"{f.name}:{i} campos vacios/ausentes: {faltan}")
                continue
            if set(d) - set(CAMPOS):
                errores.append(f"{f.name}:{i} campos extra: {sorted(set(d)-set(CAMPOS))}")
            d["_cat"] = cat
            filas.append(d)

    # 1. trazabilidad: el PDF citado debe existir como texto extraido
    disponibles = {p.stem for p in PROCESSED.glob("*.txt")}
    for d in filas:
        if slug(d["source_file"]) not in disponibles:
            errores.append(f"source_file sin texto extraido: {d['source_file']}")

    # 2. duplicados exactos y casi-exactos de pregunta
    vistos = {}
    for d in filas:
        k = norm_txt(d["question"])
        if k in vistos:
            errores.append(f"pregunta duplicada: {d['question'][:70]}...")
        vistos[k] = True

    # 3. longitudes minimas
    for d in filas:
        if len(d["answer"]) < 80:
            errores.append(f"respuesta muy corta ({len(d['answer'])} car.): {d['question'][:60]}")
        if len(d["question"]) < 20:
            errores.append(f"pregunta muy corta: {d['question']}")

    # ---- reporte de cobertura ----
    por_cat = Counter(d["_cat"] for d in filas)
    por_norma = Counter(d["source_file"] for d in filas)
    kinds = Counter(d["section_kind"] for d in filas)

    print(f"Total de pares QA: {len(filas)}")
    print("\nPor categoria:")
    for c in sorted(por_cat):
        print(f"  {c}  {por_cat[c]:3d}  {CATEGORIAS.get(c,'?')}")
    print("\nPor tipo de seccion:")
    for k, v in kinds.most_common():
        print(f"  {k:12s} {v:3d}")
    print(f"\nNormas distintas citadas: {len(por_norma)}")

    with open(OUT / "cobertura.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["categoria", "categoria_nombre", "source_file", "preguntas"])
        agr = defaultdict(Counter)
        for d in filas:
            agr[d["_cat"]][d["source_file"]] += 1
        for c in sorted(agr):
            for sf, n in sorted(agr[c].items()):
                w.writerow([c, CATEGORIAS.get(c, ""), sf, n])

    if errores:
        print(f"\n*** {len(errores)} PROBLEMA(S) DETECTADO(S) ***")
        for e in errores:
            print("  -", e)
        return 2

    # ---- escritura y split estratificado por categoria ----
    def dump(path, rows):
        path.write_text(
            "\n".join(json.dumps({k: v for k, v in r.items() if k != "_cat"},
                                 ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8")

    dump(OUT / "normas_ruido_all.jsonl", filas)

    rnd = random.Random(args.seed)
    train, ev = [], []
    for c in sorted(por_cat):
        grupo = [d for d in filas if d["_cat"] == c]
        rnd.shuffle(grupo)
        n_ev = max(1, round(len(grupo) * args.eval_frac))
        ev.extend(grupo[:n_ev])
        train.extend(grupo[n_ev:])
    rnd.shuffle(train)
    rnd.shuffle(ev)
    dump(OUT / "normas_ruido_train.jsonl", train)
    dump(OUT / "normas_ruido_eval.jsonl", ev)

    print(f"\nSplit (seed={args.seed}): train={len(train)}  eval={len(ev)}")
    print("Validacion OK: esquema, trazabilidad, duplicados y longitudes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
