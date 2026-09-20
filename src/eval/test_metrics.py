#!/usr/bin/env python
"""Pruebas de metrics.py y run_eval.py con respuestas de mentira.

Uso:  python src/eval/test_metrics.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from metrics import es_valida, limpiar_prediccion, recall_at_k, rouge

AQUI = Path(__file__).resolve().parent


def test_rouge():
    assert rouge("the sound level", "the sound level")["rouge1"] == 1.0
    assert rouge("apples", "the sound level")["rouge1"] == 0.0


def test_recall():
    fuentes = ["a.pdf", "b.pdf", "a.pdf"]
    assert recall_at_k("b.pdf", fuentes, 1) == 0.0
    assert recall_at_k("b.pdf", fuentes, 2) == 1.0
    assert recall_at_k("c.pdf", fuentes, 3) == 0.0


def test_limpieza():
    assert limpiar_prediccion("Step 1... Answer: 20 microPa", "chain-of-thought") == "20 microPa"
    assert limpiar_prediccion("sin marcador", "chain-of-thought") == "sin marcador"
    assert limpiar_prediccion("It is 6 dB [2].", "rag-anclado") == "It is 6 dB."
    assert limpiar_prediccion("Answer: x", "zero-shot") == "Answer: x"


def test_validez():
    assert es_valida("algo") and not es_valida("") and not es_valida("__ERROR__: 503")


def test_run_eval_de_punta_a_punta():
    eval_set = [{"question": f"q{i}"} for i in range(3)]

    def fila(i, sistema, tecnica, pred, **extra):
        return {"question": f"q{i}", "reference": "the reference sound pressure is 20 microPa",
                "prediction": pred, "technique": tecnica, "system": sistema,
                "source_file": "a.pdf", "section_label": "1", **extra}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "eval.jsonl").write_text("\n".join(json.dumps(x) for x in eval_set), encoding="utf-8")
        base = [fila(i, "base", "zero-shot", "the reference sound pressure is 20 microPa") for i in range(3)]
        rag = [fila(0, "rag", "rag-anclado", "20 microPa [1]",
                    retrieved_sources=["a.pdf", "b.pdf"]),
               fila(1, "rag", "rag-anclado", "__ERROR__: timeout",
                    retrieved_sources=["b.pdf", "a.pdf"]),
               fila(2, "rag", "rag-anclado", "x", retrieved_sources=["b.pdf", "c.pdf"])]
        (tmp / "respuestas-base.jsonl").write_text("\n".join(json.dumps(x) for x in base), encoding="utf-8")
        (tmp / "respuestas-rag.jsonl").write_text("\n".join(json.dumps(x) for x in rag), encoding="utf-8")

        subprocess.run([sys.executable, str(AQUI / "run_eval.py"), "--results-dir", str(tmp),
                        "--eval-set", str(tmp / "eval.jsonl")], check=True, capture_output=True)
        filas = {(r["system"], r["technique"]): r for r in
                 (dict(zip(cab, v.split(","))) for cab, v in
                  [((tmp / "metricas-comparativas.csv").read_text().splitlines()[0].split(","), l)
                   for l in (tmp / "metricas-comparativas.csv").read_text().splitlines()[1:]])}
        b = filas[("base", "zero-shot")]
        assert b["n"] == "3" and float(b["rouge1"]) == 1.0 and b["recall@1"] == ""
        r = filas[("rag", "rag-anclado")]
        assert r["n"] == "3" and r["n_validas"] == "2"
        assert float(r["recall@1"]) == round(1 / 3, 4) and float(r["recall@3"]) == round(2 / 3, 4)


if __name__ == "__main__":
    for nombre, f in list(globals().items()):
        if nombre.startswith("test_"):
            f()
            print("ok", nombre)
