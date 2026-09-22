#!/usr/bin/env python
"""Consulta el endpoint vLLM del modelo afinado que corre en la VM.

Es el gemelo de 03-predict-endpoint.py, pero contra la API OpenAI-compatible
que expone vLLM en vez de contra el endpoint de Vertex. Dos ventajas para el
taller: no depende del SDK de GCP (solo HTTP), y produce un jsonl con el mismo
esquema del HANDOFF, asi que las respuestas del endpoint se pueden comparar
fila a fila contra las que genero el Custom Job por lotes.

CONEXION. vLLM escucha en el puerto 8000 DENTRO de la VM. La forma segura de
llegar desde Windows es un tunel SSH, que no abre ningun puerto a internet:

    gcloud compute ssh NOMBRE_VM --zone=ZONA `
        --ssh-flag="-N" --ssh-flag="-L" --ssh-flag="8000:localhost:8000"

    En PowerShell hay que usar --ssh-flag: con la forma "-- -N -L ..." el shell
    se queda con el -- y gcloud responde "unrecognized arguments".

Deja esa ventana abierta y en otra corre este script contra localhost:8000.
Es ademas la unica ruta que no pelea con la red del campus: viaja por SSH
sobre el puerto 22, no por el DNS de *.goog que alla no resuelve.

Uso:
    # una pregunta suelta
    python src/deploy/08-predict-vllm.py --prompt "What does ISO 3382-1 specify?"

    # las 20 preguntas del eval set, una tecnica
    python src/deploy/08-predict-vllm.py --dataset data/qa/normas_ruido_eval.jsonl \
        --out results/respuestas-finetuning-vllm-zero-shot.jsonl --technique zero-shot

    # las tres tecnicas de un golpe -> 60 filas en un solo archivo
    python src/deploy/08-predict-vllm.py --dataset data/qa/normas_ruido_eval.jsonl \
        --out results/respuestas-finetuning-vllm.jsonl --technique all
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plantillas import PLANTILLAS, limpiar_eco  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]


def pedir(base_url: str, modelo: str, prompt: str, max_tokens: int,
          temperature: float, timeout: float) -> str:
    """Una generacion por /v1/completions.

    Se usa 'completions' y no 'chat/completions' a proposito: las plantillas ya
    traen escritas las etiquetas <start_of_turn> del chat template de Gemma. Si
    se mandaran por el endpoint de chat, vLLM aplicaria el template OTRA VEZ y
    el prompt que llega al modelo dejaria de ser el mismo que vio el modelo
    base en la linea base, que es justo lo que hace comparable la medicion.
    """
    cuerpo = json.dumps({
        "model": modelo,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": ["<end_of_turn>"],
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=cuerpo, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        datos = json.loads(r.read().decode("utf-8"))
    return datos["choices"][0]["text"]


def modelo_servido(base_url: str, timeout: float = 15.0) -> str:
    """vLLM nombra el modelo como se le paso en --served-model-name; si no se
    paso, usa la ruta completa del directorio. Preguntarselo evita un 404 por
    adivinar mal."""
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/v1/models", timeout=timeout) as r:
        datos = json.loads(r.read().decode("utf-8"))
    return datos["data"][0]["id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", default="http://localhost:8000",
                    help="URL de vLLM. Con el tunel SSH, localhost:8000")
    ap.add_argument("--model", default=None,
                    help="Nombre del modelo servido. Por defecto se le pregunta a /v1/models")
    ap.add_argument("--prompt", default=None, help="Una pregunta suelta")
    ap.add_argument("--dataset", default=None, help="jsonl con campo 'question'")
    ap.add_argument("--out", default=None, help="jsonl de salida")
    ap.add_argument("--technique", default="zero-shot",
                    choices=[*PLANTILLAS, "all"])
    ap.add_argument("--system", default="finetuning",
                    choices=["base", "finetuning", "rag", "rag-vertex"])
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--limit", type=int, default=0, help="0 = todas")
    args = ap.parse_args()

    try:
        modelo = args.model or modelo_servido(args.base_url)
    except urllib.error.URLError as e:
        raise SystemExit(
            f"No hay nadie escuchando en {args.base_url} ({e}).\n"
            "Revisa que el tunel SSH este abierto y que vLLM este arriba:\n"
            "  gcloud compute ssh VM --zone=ZONA --ssh-flag=\"-N\" "
            "--ssh-flag=\"-L\" --ssh-flag=\"8000:localhost:8000\"\n"
            "  (dentro de la VM)  ./07-vm-vllm.sh status")
    print(f"Endpoint : {args.base_url}")
    print(f"Modelo   : {modelo}")

    if args.prompt:
        plantilla = PLANTILLAS[args.technique if args.technique != "all" else "zero-shot"]
        t0 = time.time()
        crudo = pedir(args.base_url, modelo, plantilla.format(q=args.prompt),
                      args.max_tokens, args.temperature, args.timeout)
        print(f"\n--- respuesta ({time.time() - t0:.1f} s) ---")
        print(limpiar_eco(crudo))
        return 0

    if not args.dataset or not args.out:
        raise SystemExit("Usa --prompt, o --dataset junto con --out.")

    ruta_ds = Path(args.dataset)
    if not ruta_ds.is_absolute():
        ruta_ds = RAIZ / ruta_ds
    filas = [json.loads(l) for l in ruta_ds.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        filas = filas[:args.limit]

    tecnicas = list(PLANTILLAS) if args.technique == "all" else [args.technique]
    salida = []
    fallos = 0
    for tecnica in tecnicas:
        plantilla = PLANTILLAS[tecnica]
        for i, fila in enumerate(filas, 1):
            pregunta = fila["question"]
            try:
                crudo = pedir(args.base_url, modelo, plantilla.format(q=pregunta),
                              args.max_tokens, args.temperature, args.timeout)
            except Exception as e:  # noqa: BLE001
                # No se aborta la corrida entera por una pregunta: se marca la
                # fila y run_eval.py la cuenta como invalida, que es
                # informacion util y no un archivo a medias.
                print(f"  [{tecnica} {i}/{len(filas)}] ERROR: {e}")
                crudo = ""
                fallos += 1
            salida.append({
                "question": pregunta,
                "reference": fila["answer"],
                "prediction": limpiar_eco(crudo),
                "prediction_raw": crudo,
                "technique": tecnica,
                "system": args.system,
                "source_file": fila["source_file"],
                "section_label": fila["section_label"],
            })
            print(f"  [{tecnica} {i}/{len(filas)}] {len(salida[-1]['prediction'])} caracteres")

    ruta_out = Path(args.out)
    if not ruta_out.is_absolute():
        ruta_out = RAIZ / ruta_out
    ruta_out.parent.mkdir(parents=True, exist_ok=True)
    with ruta_out.open("w", encoding="utf-8") as f:
        for r in salida:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{len(salida)} filas en {ruta_out.relative_to(RAIZ)}"
          + (f"  ({fallos} con error)" if fallos else ""))
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
