#!/usr/bin/env python
"""Lector de las plantillas de prompt, para quien no quiera importar el SDK.

Las tres plantillas viven en el diccionario PLANTILLAS de
03-predict-endpoint.py, que es el script que de verdad las envia al endpoint
del modelo base. Ese archivo importa google.cloud.aiplatform al principio, asi
que no se puede importar desde un cliente que solo habla HTTP (el de vLLM) ni
desde la app de Streamlit sin arrastrar el SDK entero. Y su nombre empieza por
un digito, que no es un identificador valido de Python.

La solucion es la misma que ya usa export_prompts.py: leer el diccionario con
ast, sin ejecutar el modulo. Asi hay un unico lugar donde se editan los
prompts y los cuatro consumidores (03, 08, export_prompts y Streamlit) leen el
mismo texto.
"""
import ast
from pathlib import Path

FUENTE = Path(__file__).with_name("03-predict-endpoint.py")

SEPARADORES_ECO = ["\nOutput:\n", "\nOutput:", "<start_of_turn>model\n"]
TOKENS_CONTROL = ["<end_of_turn>", "<start_of_turn>", "<eos>", "<bos>", "<pad>"]


def cargar_plantillas(fuente: Path = FUENTE) -> dict:
    arbol = ast.parse(fuente.read_text(encoding="utf-8"))
    for nodo in arbol.body:
        if isinstance(nodo, ast.Assign) and any(
                getattr(d, "id", None) == "PLANTILLAS" for d in nodo.targets):
            return ast.literal_eval(nodo.value)
    raise SystemExit(f"No se encontro el diccionario PLANTILLAS en {fuente.name}")


def limpiar_eco(texto: str) -> str:
    """Misma limpieza que 03-predict-endpoint.py.

    vLLM devuelve solo los tokens generados, asi que aqui normalmente no hay
    nada que cortar; queda por si se cambia a 'echo': true o si el modelo
    emite <end_of_turn> al final, que si pasa.
    """
    for sep in SEPARADORES_ECO:
        if sep in texto:
            texto = texto.split(sep, 1)[1]
            break
    for token in TOKENS_CONTROL:
        texto = texto.replace(token, "")
    return texto.strip()


PLANTILLAS = cargar_plantillas()

if __name__ == "__main__":
    for clave, texto in PLANTILLAS.items():
        print(f"--- {clave} ({len(texto)} caracteres) ---")
        print(texto)
