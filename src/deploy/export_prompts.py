#!/usr/bin/env python
"""Exporta a prompts/ el texto exacto de cada tecnica de prompt engineering.

El taller exige documentar el prompt exacto usado en cada tecnica. Para que lo
documentado no se desincronice de lo que realmente se envia, este script lee
las plantillas del propio 03-predict-endpoint.py y las escribe en prompts/.

Uso:
    python src\\deploy\\export_prompts.py
"""
import ast
import pathlib
import sys

AQUI = pathlib.Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
DESTINO = RAIZ / "prompts"

DESCRIPCIONES = {
    "zero-shot": (
        "Zero-shot",
        "Se le da al modelo solo el rol de experto y la pregunta, sin ningun "
        "ejemplo resuelto. Es la linea base mas pura: mide lo que el modelo "
        "sabe del dominio sin ayuda de contexto."),
    "few-shot": (
        "Few-shot",
        "Se anteponen dos pares pregunta-respuesta resueltos, tomados del "
        "dataset de entrenamiento, para fijar el formato y el nivel de detalle "
        "esperado. Mide cuanto mejora la respuesta solo por ver ejemplos del "
        "estilo, sin que el modelo aprenda contenido nuevo."),
    "chain-of-thought": (
        "Chain-of-thought",
        "Se le pide al modelo razonar por pasos antes de responder: primero "
        "identificar la norma y el apartado, luego recordar que exige ese "
        "apartado, y al final dar la respuesta en una linea que empieza con "
        "'Answer:'. Mide si el razonamiento explicito ayuda en un dominio "
        "donde la respuesta correcta depende de ubicar la clausula adecuada."),
}


def main():
    # Se lee el diccionario PLANTILLAS con ast en vez de importar el modulo,
    # para no exigir que el SDK de GCP este instalado solo para exportar texto.
    fuente = (AQUI / "03-predict-endpoint.py").read_text(encoding="utf-8")
    arbol = ast.parse(fuente)
    plantillas = None
    for nodo in arbol.body:
        if isinstance(nodo, ast.Assign) and any(
                getattr(d, "id", None) == "PLANTILLAS" for d in nodo.targets):
            plantillas = ast.literal_eval(nodo.value)
            break
    if not plantillas:
        raise SystemExit("No se encontro el diccionario PLANTILLAS en 03-predict-endpoint.py")

    DESTINO.mkdir(parents=True, exist_ok=True)
    for clave, plantilla in plantillas.items():
        titulo, descripcion = DESCRIPCIONES[clave]
        texto = (
            f"# Tecnica: {titulo}\n\n"
            f"{descripcion}\n\n"
            "Se aplica igual al modelo base, al modelo afinado y a la etapa de\n"
            "generacion del RAG, para que la comparacion sea sobre lo mismo.\n\n"
            "El marcador `{q}` se reemplaza por la pregunta. Las etiquetas\n"
            "`<start_of_turn>` y `<end_of_turn>` son el chat template de Gemma.\n\n"
            "## Prompt exacto\n\n"
            "```text\n" + plantilla + "\n```\n"
        )
        destino = DESTINO / f"{clave}.md"
        destino.write_text(texto, encoding="utf-8")
        print(f"  {destino.relative_to(RAIZ)}")

    rag = DESTINO / "rag-anclado.md"
    if not rag.exists():
        rag.write_text(
            "# Tecnica: RAG anclado\n\n"
            "Pendiente: se completa en la Fase 3, cuando el corpus este en\n"
            "Vertex AI RAG Engine. El prompt antepone los fragmentos\n"
            "recuperados a la pregunta y le pide al modelo responder solo con\n"
            "lo que esta en esos fragmentos.\n", encoding="utf-8")
        print(f"  {rag.relative_to(RAIZ)} (plantilla vacia)")

    print(f"\n{len(plantillas)} prompts exportados a {DESTINO.relative_to(RAIZ)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
