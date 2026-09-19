#!/usr/bin/env python
"""Consulta el endpoint desplegado (base o afinado) y guarda las respuestas.

Lee el endpoint de .endpoint_base.json, que deja el script 02. Sirve tanto
para la prueba rapida de humo como para generar el archivo de respuestas del
modelo base que luego compara el script de evaluacion.

Uso (PowerShell, con .venv activado):
    # una pregunta suelta
    python ...\03-predict-endpoint.py --prompt "What is the scope of ISO 1996-1?"

    # las 20 preguntas del split de evaluacion -> results/respuestas-base.jsonl
    python ...\03-predict-endpoint.py --dataset data\qa\normas_ruido_eval.jsonl `
        --out results\respuestas-base.jsonl --technique zero-shot
"""
import argparse
import json
import sys
from pathlib import Path

import os

# --- Resolucion DNS en Windows ------------------------------------------------
# gRPC trae su propio resolvedor DNS (c-ares) que en Windows falla con
#   "UNAVAILABLE: ... getaddrinfo: WSA Error ... 11001"
# en redes con VPN, DNS corporativo o IPv6 a medias, aunque el navegador y
# gcloud si resuelvan el mismo nombre. Esto le dice a gRPC que use el
# resolvedor del sistema operativo, que es el que si funciona ahi.
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")

from google.cloud import aiplatform

ESTADO = Path(__file__).with_name(".endpoint_base.json")

PLANTILLAS = {
    # Las 3 tecnicas de prompt engineering que exige el taller. El prompt exacto
    # de cada una se documenta en prompts/ ; aqui viven las versiones operativas.
    "zero-shot": (
        "<start_of_turn>user\n"
        "You are an expert on ISO, UNE and BS acoustics standards for noise "
        "measurement. Answer the question precisely and cite the clause when you know it.\n\n"
        "Question: {q}<end_of_turn>\n<start_of_turn>model\n"
    ),
    "few-shot": (
        "<start_of_turn>user\n"
        "You are an expert on ISO, UNE and BS acoustics standards for noise measurement.\n\n"
        "Example 1\n"
        "Question: What is the reference sound pressure used in ISO 1996-1?\n"
        "Answer: The reference sound pressure is 20 microPa, and sound pressure is expressed in pascals.\n\n"
        "Example 2\n"
        "Question: What instrumentation class does ISO 3746 require?\n"
        "Answer: The instrumentation system shall meet IEC 61672-1:2002 class 2, although class 1 is recommended.\n\n"
        "Now answer in the same style.\n"
        "Question: {q}<end_of_turn>\n<start_of_turn>model\n"
    ),
    "chain-of-thought": (
        "<start_of_turn>user\n"
        "You are an expert on ISO, UNE and BS acoustics standards for noise measurement.\n"
        "Think step by step: first identify which standard and which clause the question "
        "refers to, then recall what that clause requires, then state the final answer.\n"
        "Finish with a line starting with 'Answer:' that contains only the final answer.\n\n"
        "Question: {q}<end_of_turn>\n<start_of_turn>model\n"
    ),
}


def cargar_estado():
    if not ESTADO.exists():
        raise SystemExit(
            f"No existe {ESTADO.name}. Corre primero 02-deploy-model-garden-base.py, "
            "o pasa --endpoint_id, --project y --region a mano."
        )
    return json.loads(ESTADO.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint_id", default=None)
    ap.add_argument("--project", default=None)
    ap.add_argument("--region", default=None)
    ap.add_argument("--prompt", default=None, help="Una pregunta suelta")
    ap.add_argument("--dataset", default=None, help="jsonl con campo 'question'")
    ap.add_argument("--out", default=None, help="jsonl de salida")
    ap.add_argument("--technique", default="zero-shot", choices=list(PLANTILLAS))
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0, help="0 = todas")
    ap.add_argument("--debug", action="store_true",
                    help="Imprime la respuesta cruda del endpoint. Util la primera "
                         "vez: distintos contenedores de serving devuelven la "
                         "prediccion con nombres de campo distintos.")
    ap.add_argument("--transport", default="rest", choices=["rest", "grpc"],
                    help="rest usa HTTPS normal (como gcloud) y respeta el proxy del "
                         "sistema; grpc abre su propio canal y suele ser lo que bloquean "
                         "los firewalls corporativos. Por eso el default es rest.")

    args = ap.parse_args()

    if args.endpoint_id and args.project and args.region:
        est = {"endpoint_id": args.endpoint_id, "project": args.project, "region": args.region}
    else:
        est = cargar_estado()

    aiplatform.init(project=est["project"], location=est["region"],
                    api_transport=args.transport)
    endpoint = aiplatform.Endpoint(
        endpoint_name=(f"projects/{est['project']}/locations/{est['region']}"
                       f"/endpoints/{est['endpoint_id']}")
    )

    plantilla = PLANTILLAS[args.technique]

    def preguntar(q: str) -> str:
        instancia = {
            "prompt": plantilla.format(q=q),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": 1.0,
            "top_k": -1,
        }
        r = endpoint.predict(instances=[instancia])
        pred = r.predictions[0]
        if args.debug:
            print("--- respuesta cruda del endpoint ---")
            print(f"tipo: {type(pred).__name__}")
            if isinstance(pred, dict):
                print(f"campos: {sorted(pred.keys())}")
            print(json.dumps(pred, ensure_ascii=False, default=str)[:1500])
            print("------------------------------------")
        if isinstance(pred, dict):
            for clave in ("generated_text", "text", "content", "output"):
                if clave in pred:
                    return str(pred[clave])
            return json.dumps(pred, ensure_ascii=False)
        return str(pred)

    if args.prompt:
        print(preguntar(args.prompt))
        return 0

    if not args.dataset:
        raise SystemExit("Pasa --prompt o --dataset.")

    filas = [json.loads(l) for l in Path(args.dataset).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        filas = filas[:args.limit]

    salida = []
    for i, d in enumerate(filas, 1):
        print(f"[{i}/{len(filas)}] {d['question'][:70]}...")
        try:
            resp = preguntar(d["question"])
        except Exception as e:                                   # noqa: BLE001
            resp = f"__ERROR__: {e}"
        salida.append({
            "question": d["question"],
            "reference": d.get("answer", ""),
            "prediction": resp,
            "technique": args.technique,
            "system": "base",
            "source_file": d.get("source_file", ""),
            "section_label": d.get("section_label", ""),
        })

    destino = Path(args.out) if args.out else Path("results/respuestas-base.jsonl")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in salida) + "\n",
        encoding="utf-8")
    errores = sum(1 for x in salida if x["prediction"].startswith("__ERROR__"))
    print(f"\n{len(salida)} respuestas -> {destino}  ({errores} con error)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
