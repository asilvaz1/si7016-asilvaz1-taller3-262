#!/usr/bin/env python
"""Limpia el eco del prompt en las respuestas del endpoint.

El contenedor de serving de Model Garden devuelve el prompt completo seguido
de la generacion, con esta forma:

    Prompt:
    <start_of_turn>user
    ...la pregunta...<end_of_turn>
    <start_of_turn>model
    Output:
    ...la respuesta del modelo...

Si se calcula ROUGE contra eso, el resultado no mide nada: la prediccion
arrastra cientos de tokens del prompt que la referencia no tiene. Este script
deja en `prediction` solo lo que el modelo genero, y guarda el texto original
en `prediction_raw` para que la evidencia no se pierda.

Uso:
    python src\\eval\\clean_predictions.py results\\respuestas-base-*.jsonl
    python src\\eval\\clean_predictions.py results\\*.jsonl --dry-run
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

SEPARADORES = ["\nOutput:\n", "\nOutput:", "<start_of_turn>model\n"]
TOKENS_CONTROL = ["<end_of_turn>", "<start_of_turn>", "<eos>", "<bos>", "<pad>"]


def limpiar(texto: str) -> str:
    """Se queda con lo generado por el modelo, sin el eco del prompt."""
    for sep in SEPARADORES:
        if sep in texto:
            texto = texto.split(sep, 1)[1]
            break
    for token in TOKENS_CONTROL:
        texto = texto.replace(token, "")
    # El contenedor a veces corta a mitad de palabra al llegar a max_tokens;
    # eso se deja tal cual, es informacion real sobre el limite usado.
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("patrones", nargs="+", help="Archivos jsonl o patrones glob")
    ap.add_argument("--dry-run", action="store_true",
                    help="Muestra el efecto sin escribir nada")
    args = ap.parse_args()

    archivos = []
    for patron in args.patrones:
        encontrados = glob.glob(patron)
        archivos.extend(encontrados or ([patron] if Path(patron).exists() else []))
    archivos = sorted(set(archivos))
    if not archivos:
        print("No se encontro ningun archivo con esos patrones.", file=sys.stderr)
        return 1

    for ruta in archivos:
        p = Path(ruta)
        filas = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        cambiadas = 0
        for d in filas:
            if "prediction_raw" in d:
                continue                      # ya estaba limpio, no re-procesar
            crudo = d.get("prediction", "")
            limpio = limpiar(crudo)
            if limpio != crudo:
                cambiadas += 1
            d["prediction_raw"] = crudo
            d["prediction"] = limpio

        antes = sum(len(d.get("prediction_raw", d["prediction"])) for d in filas) // max(len(filas), 1)
        despues = sum(len(d["prediction"]) for d in filas) // max(len(filas), 1)
        print(f"{p}: {len(filas)} filas, {cambiadas} limpiadas  "
              f"(largo medio {antes} -> {despues} caracteres)")

        if cambiadas and not args.dry_run:
            respaldo = p.with_suffix(p.suffix + ".raw")
            if not respaldo.exists():
                respaldo.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
            p.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in filas) + "\n",
                         encoding="utf-8")

        if filas:
            print("   ejemplo:", repr(filas[0]["prediction"][:180]))

    if args.dry_run:
        print("\n--dry-run: no se escribio nada.")
    else:
        print("\nOriginales respaldados con extension .raw al lado de cada archivo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
