#!/usr/bin/env python
"""Plantillas de generacion del RAG y armado del contexto.

Las tres tecnicas de prompt engineering (zero-shot, few-shot, chain-of-thought)
usan el MISMO texto que las plantillas del modelo base en
src/deploy/03-predict-endpoint.py, con un bloque de contexto recuperado
delante de la pregunta. Asi la unica diferencia entre sistemas es el contexto.
'rag-anclado' es el prompt estricto del ejemplo 5 del enunciado: responder solo
con los fragmentos, citarlos, y admitir cuando no alcanza.

Uso:
    python src/rag/prompts.py --export     # regenera prompts/rag-anclado.md
"""
import argparse
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]

ROL = ("You are an expert on ISO, UNE and BS acoustics standards for noise "
       "measurement.")

NO_INFO = "I do not have enough information."

# Mismo bloque en las tres variantes que reutilizan la plantilla del base.
BLOQUE_CONTEXTO = (
    "Use ONLY the numbered excerpts below, taken from the standards. If the "
    f"answer is not in the excerpts, reply exactly \"{NO_INFO}\"\n\n"
    "Context:\n{context}\n\n"
)

PLANTILLAS_RAG = {
    "rag-anclado": (
        "<start_of_turn>user\n"
        + ROL + " Answer the question using ONLY the excerpts below. "
        "Quote the exact passage you used and cite it by its number, e.g. [2]. "
        f"If the answer is not in the excerpts, reply exactly \"{NO_INFO}\"\n\n"
        "Context:\n{context}\n\n"
        "Question: {q}<end_of_turn>\n<start_of_turn>model\n"
    ),
    "zero-shot": (
        "<start_of_turn>user\n"
        + ROL + " Answer the question precisely and cite the clause when you know it.\n\n"
        + BLOQUE_CONTEXTO
        + "Question: {q}<end_of_turn>\n<start_of_turn>model\n"
    ),
    "few-shot": (
        "<start_of_turn>user\n"
        + ROL + "\n\n"
        "Example 1\n"
        "Question: What is the reference sound pressure used in ISO 1996-1?\n"
        "Answer: The reference sound pressure is 20 microPa, and sound pressure is expressed in pascals.\n\n"
        "Example 2\n"
        "Question: What instrumentation class does ISO 3746 require?\n"
        "Answer: The instrumentation system shall meet IEC 61672-1:2002 class 2, although class 1 is recommended.\n\n"
        + BLOQUE_CONTEXTO
        + "Now answer in the same style.\n"
        "Question: {q}<end_of_turn>\n<start_of_turn>model\n"
    ),
    "chain-of-thought": (
        "<start_of_turn>user\n"
        + ROL + "\n"
        "Think step by step: first identify which excerpt and which clause the question "
        "refers to, then recall what that clause requires, then state the final answer.\n"
        "Finish with a line starting with 'Answer:' that contains only the final answer.\n\n"
        + BLOQUE_CONTEXTO
        + "Question: {q}<end_of_turn>\n<start_of_turn>model\n"
    ),
}

TITULOS = {
    "rag-anclado": "Variante estricta (technique = rag-anclado)",
    "zero-shot": "RAG + zero-shot",
    "few-shot": "RAG + few-shot",
    "chain-of-thought": "RAG + chain-of-thought",
}

DESCRIPCION = (
    "# Tecnica: RAG anclado\n\n"
    "Prompt de generacion del sistema RAG. Antepone a la pregunta los k "
    "fragmentos recuperados (numerados, con el nombre de la norma) y le exige "
    "al modelo responder **solo** con ellos, citar el fragmento por su numero "
    f"y declarar '{NO_INFO}' cuando la respuesta no este en el contexto.\n\n"
    "Ademas de esta variante estricta, las tres tecnicas del taller "
    "(zero-shot, few-shot y chain-of-thought) se aplican a la etapa de "
    "generacion del RAG con el mismo texto que en el modelo base mas el "
    "bloque de contexto. Se distinguen en `results/respuestas-rag.jsonl` por "
    "el campo `technique`; en los cuatro casos `system` es `rag`.\n\n"
    "Los marcadores `{context}` y `{q}` se reemplazan por los fragmentos y la "
    "pregunta. Las etiquetas `<start_of_turn>` y `<end_of_turn>` son el chat "
    "template de Gemma. El texto operativo vive en `src/rag/prompts.py`; este "
    "archivo se regenera con `python src/rag/prompts.py --export`.\n\n"
    "## Formato del contexto\n\n"
    "```text\n[1] (ISO 3746) <texto del chunk>\n\n[2] (BS 4142) <texto del chunk>\n```\n"
)


def formatear_contexto(hits) -> str:
    return "\n\n".join(f"[{h['rank']}] ({h['norma']}) {h['text']}" for h in hits)


def construir_prompt(tecnica: str, pregunta: str, hits) -> str:
    return PLANTILLAS_RAG[tecnica].format(context=formatear_contexto(hits), q=pregunta)


def exportar():
    partes = [DESCRIPCION]
    for clave, plantilla in PLANTILLAS_RAG.items():
        partes.append(f"## Prompt exacto: {TITULOS[clave]}\n\n```text\n{plantilla}\n```\n")
    destino = RAIZ / "prompts" / "rag-anclado.md"
    destino.write_text("\n".join(partes), encoding="utf-8")
    print(f"  {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    if ap.parse_args().export:
        exportar()
