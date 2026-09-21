#!/usr/bin/env python
"""Metricas del taller: ROUGE para la generacion y Recall@K para la recuperacion.

Funciones puras, sin leer archivos: run_eval.py se encarga de la E/S.
"""
import re

from rouge_score import rouge_scorer

ROUGE_TIPOS = ("rouge1", "rouge2", "rougeL")
_scorer = rouge_scorer.RougeScorer(list(ROUGE_TIPOS), use_stemmer=True)


def es_valida(prediccion: str) -> bool:
    """Una prediccion vacia o con error del endpoint no se puntua: contarla
    como 0 mezclaria un fallo de infraestructura con la calidad del modelo."""
    p = (prediccion or "").strip()
    return bool(p) and not p.startswith("__ERROR__")


def limpiar_prediccion(prediccion: str, tecnica: str) -> str:
    """Deja solo la respuesta, para que ROUGE no puntue el formato del prompt.

    - chain-of-thought: el prompt pide terminar con una linea 'Answer:', asi que
      se puntua solo lo que viene despues de la ultima. Si el modelo no la
      escribio, se usa el texto completo.
    - rag-anclado: se quitan las citas numericas tipo [2].
    """
    p = (prediccion or "").strip()
    if tecnica == "chain-of-thought":
        partes = re.split(r"(?i)\banswer\s*:", p)
        if len(partes) > 1 and partes[-1].strip():
            p = partes[-1].strip()
    if tecnica == "rag-anclado":
        p = re.sub(r"\s*\[\d+\]", "", p).strip()
    return p


def rouge(prediccion: str, referencia: str) -> dict:
    """F-measure de ROUGE-1, ROUGE-2 y ROUGE-L entre una prediccion y su referencia."""
    s = _scorer.score(referencia, prediccion)
    return {t: s[t].fmeasure for t in ROUGE_TIPOS}


def recall_at_k(fuente_esperada: str, fuentes_recuperadas: list, k: int) -> float:
    """1.0 si el source_file esperado esta entre los primeros k documentos
    recuperados (en orden de ranking), 0.0 si no. A nivel de documento."""
    return float(fuente_esperada in fuentes_recuperadas[:k])
