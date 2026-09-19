#!/usr/bin/env python
"""Apaga el endpoint para que deje de cobrar.

Model Garden y los endpoints de Vertex cobran por hora mientras el modelo este
desplegado, reciba peticiones o no. Este es el script que mas plata ahorra del
repositorio: correrlo al terminar cada sesion de pruebas.

Por defecto solo hace undeploy del modelo, que es lo que detiene el cobro y
deja el endpoint vacio listo para volver a usar. Con --delete-endpoint tambien
borra el endpoint, y con --delete-model borra ademas el modelo del registro.

Uso (PowerShell, con .venv activado):
    python ...\04-undeploy-cleanup.py                 # detiene el cobro
    python ...\04-undeploy-cleanup.py --delete-endpoint --delete-model
    python ...\04-undeploy-cleanup.py --list          # que hay desplegado ahora
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
PROJECT = "si7016-262-nlp"
REGION = "us-central1"


def listar(project, region, transport="rest"):
    aiplatform.init(project=project, location=region, api_transport=transport)
    endpoints = aiplatform.Endpoint.list()
    if not endpoints:
        print("No hay endpoints en esta region. Nada esta cobrando por este concepto.")
        return
    print(f"{'ENDPOINT_ID':<24} {'DISPLAY_NAME':<45} MODELOS_DESPLEGADOS")
    for ep in endpoints:
        desplegados = ep.list_models()
        marca = f"{len(desplegados)}"
        if desplegados:
            marca += "  <-- COBRANDO"
        print(f"{ep.name:<24} {ep.display_name:<45} {marca}")
        for dm in desplegados:
            maq = dm.dedicated_resources.machine_spec if dm.dedicated_resources else None
            detalle = f"{maq.machine_type} {maq.accelerator_type} x{maq.accelerator_count}" if maq else "?"
            print(f"    deployed_model_id={dm.id}  {detalle}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None)
    ap.add_argument("--region", default=None)
    ap.add_argument("--endpoint_id", default=None)
    ap.add_argument("--list", action="store_true", help="Solo listar, no apagar nada")
    ap.add_argument("--delete-endpoint", action="store_true")
    ap.add_argument("--delete-model", action="store_true")
    ap.add_argument("--transport", default="rest", choices=["rest", "grpc"],
                    help="rest usa HTTPS normal (como gcloud) y respeta el proxy del "
                         "sistema; grpc abre su propio canal y suele ser lo que bloquean "
                         "los firewalls corporativos. Por eso el default es rest.")
    ap.add_argument("--all", action="store_true",
                    help="Aplica el undeploy a TODOS los endpoints de la region")
    args = ap.parse_args()

    est = {}
    if ESTADO.exists():
        est = json.loads(ESTADO.read_text(encoding="utf-8"))
    project = args.project or est.get("project", PROJECT)
    region = args.region or est.get("region", REGION)

    if args.list:
        listar(project, region, args.transport)
        return 0

    aiplatform.init(project=project, location=region, api_transport=args.transport)

    if args.all:
        objetivos = aiplatform.Endpoint.list()
    else:
        endpoint_id = args.endpoint_id or est.get("endpoint_id")
        if not endpoint_id:
            raise SystemExit(
                "No se sabe cual endpoint apagar. Usa --endpoint_id, o --all, "
                f"o corre 02-deploy-model-garden-base.py para que genere {ESTADO.name}."
            )
        objetivos = [aiplatform.Endpoint(
            endpoint_name=f"projects/{project}/locations/{region}/endpoints/{endpoint_id}")]

    for ep in objetivos:
        print(f"\n== {ep.display_name} ({ep.name}) ==")
        desplegados = ep.list_models()
        if not desplegados:
            print("  Sin modelos desplegados; ya no cobra.")
        for dm in desplegados:
            print(f"  Retirando deployed_model_id={dm.id} ...")
            ep.undeploy(deployed_model_id=dm.id, sync=True)
            print("  Retirado. El cobro por hora se detuvo.")
            if args.delete_model:
                try:
                    modelo = aiplatform.Model(model_name=dm.model)
                    print(f"  Borrando modelo del registro: {modelo.display_name}")
                    modelo.delete(sync=True)
                except Exception as e:                           # noqa: BLE001
                    print(f"  No se pudo borrar el modelo: {e}")

        if args.delete_endpoint:
            print("  Borrando el endpoint ...")
            ep.delete(force=True, sync=True)
            print("  Endpoint borrado.")

    if args.delete_endpoint and ESTADO.exists():
        ESTADO.unlink()

    print("\n== Estado final ==")
    listar(project, region, args.transport)
    print("\nOjo: esto no toca las VM de Compute Engine ni los datos en GCS.")
    print("  VM activas:   gcloud compute instances list")
    print("  Uso del bucket: gcloud storage du -s gs://asilvaz1taller3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
