#!/usr/bin/env python
"""Victoria temprana 1: desplegar gemma-7b-it BASE (sin afinar) como endpoint
gestionado en Vertex AI Model Garden.

Este endpoint sirve para dos cosas:
  1. Evaluar el modelo base con las 3 tecnicas de prompt engineering, que es
     el primer paso obligatorio de la evaluacion del taller.
  2. Servir de plantilla ya probada para desplegar despues el modelo afinado.

IMPORTANTE - COSTO: el endpoint cobra por hora mientras este desplegado,
aunque no se le mande ni una sola peticion. Al terminar cada sesion de pruebas:
    python 04-undeploy-cleanup.py

Antes de correr esto, ejecuta 01-list-gemma-model-garden.py y usa los valores
que imprima. Los defaults de abajo son un punto de partida razonable, pero el
contenedor de serving cambia de version con el tiempo.

Uso (PowerShell, con .venv activado):
    python ...\02-deploy-model-garden-base.py --model "google/gemma@gemma-7b-it"
    # opcional: --container <uri>  --machine_type g2-standard-12  --accelerator NVIDIA_L4
    # ensayo en seco, no despliega ni cobra:
    python ...\02-deploy-model-garden-base.py --dry_run
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

import vertexai
from vertexai import model_garden

PROJECT = "si7016-262-nlp"
REGION = "us-central1"
ESTADO = Path(__file__).with_name(".endpoint_base.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--region", default=REGION)
    ap.add_argument("--model", default="google/gemma@gemma-7b-it",
                    help="Id exacto del model card, tal como lo imprime el script 01")
    ap.add_argument("--container", default=None,
                    help="serving_container_image_uri; si se omite, Model Garden "
                         "elige el contenedor verificado por defecto")
    ap.add_argument("--machine_type", default="g2-standard-12")
    ap.add_argument("--accelerator", default="NVIDIA_L4")
    ap.add_argument("--accelerator_count", type=int, default=1)
    ap.add_argument("--endpoint_name", default="gemma-7b-it-base-si7016")
    ap.add_argument("--model_name", default="gemma-7b-it-base-si7016")
    ap.add_argument("--dedicated", action="store_true",
                    help="Endpoint con host dedicado (como en create-model-agent-platform.py)")
    ap.add_argument("--no-dedicated", dest="no_dedicated", action="store_true",
                    help="Fuerza un endpoint SIN host dedicado. Usalo si tu red no "
                         "resuelve el DNS *.prediction.vertexai.goog. OJO: no basta "
                         "con omitir --dedicated, porque Model Garden habilita el "
                         "host dedicado por su cuenta; hay que desactivarlo "
                         "explicitamente con esta opcion.")
    ap.add_argument("--transport", default="rest", choices=["rest", "grpc"],
                    help="rest usa HTTPS normal (como gcloud) y respeta el proxy del "
                         "sistema; grpc abre su propio canal y suele ser lo que bloquean "
                         "los firewalls corporativos. Por eso el default es rest.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Muestra lo que se desplegaria y termina, sin crear nada")
    args = ap.parse_args()

    vertexai.init(project=args.project, location=args.region,
                  api_transport=args.transport)

    kwargs = dict(
        accept_eula=True,                     # acepta la licencia de Gemma para el proyecto
        machine_type=args.machine_type,
        accelerator_type=args.accelerator,
        accelerator_count=args.accelerator_count,
        endpoint_display_name=f"{args.endpoint_name}-mg-deploy",
        model_display_name=args.model_name,
        use_dedicated_endpoint=args.dedicated,
        reservation_affinity_type="NO_RESERVATION",
    )
    # El SDK solo ACTIVA el host dedicado cuando use_dedicated_endpoint=True;
    # pasarlo en False no lo desactiva, y Model Garden lo habilita por defecto
    # para varios modelos. Para desactivarlo de verdad hay que mandar
    # dedicated_endpoint_disabled=True.
    if args.no_dedicated:
        kwargs["dedicated_endpoint_disabled"] = True
    if args.container:
        kwargs["serving_container_image_uri"] = args.container

    print("== Despliegue del modelo BASE ==")
    print(f"  proyecto   : {args.project}")
    print(f"  region     : {args.region}")
    print(f"  transporte : {args.transport}")
    print(f"  modelo     : {args.model}")
    for k, v in kwargs.items():
        print(f"  {k:26s}: {v}")

    if args.dry_run:
        print("\n--dry_run: no se desplego nada y no se genero costo.")
        return 0

    print("\nDesplegando. Un modelo 7B tarda entre 15 y 30 minutos en quedar listo.")
    print("No cierres la terminal: el SDK espera a que el endpoint responda.\n")

    modelo = model_garden.OpenModel(args.model)
    endpoint = modelo.deploy(**kwargs)

    endpoint_id = endpoint.name.split("/")[-1] if hasattr(endpoint, "name") else str(endpoint)
    estado = {
        "project": args.project,
        "region": args.region,
        "model_garden_id": args.model,
        "endpoint_resource_name": getattr(endpoint, "resource_name", str(endpoint)),
        "endpoint_id": endpoint_id,
        "endpoint_display_name": kwargs["endpoint_display_name"],
        "dedicated": args.dedicated,
    }
    ESTADO.write_text(json.dumps(estado, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n== Endpoint listo ==")
    print(json.dumps(estado, indent=2, ensure_ascii=False))
    print(f"\nEstado guardado en {ESTADO.name} (lo leen los scripts 03 y 04).")
    print("\nRECUERDA: cobra por hora mientras siga desplegado.")
    print("Al terminar las pruebas:  python 04-undeploy-cleanup.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
