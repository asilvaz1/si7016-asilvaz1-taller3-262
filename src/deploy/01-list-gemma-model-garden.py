#!/usr/bin/env python
"""Descubre el nombre exacto del model card de Gemma en Model Garden y sus
opciones de despliegue verificadas.

Esto evita el riesgo que senala la hoja de ruta: adivinar el id del modelo o
copiar un contenedor de serving que ya quedo obsoleto. El SDK devuelve las
combinaciones (contenedor, machine_type, accelerator) que Google tiene
verificadas para ese modelo; se usa una de esas, no una inventada.

Uso (PowerShell, con .venv activado):
    python .\si7016-asilvaz1-taller3-262\src\deploy\01-list-gemma-model-garden.py
    python ...\01-list-gemma-model-garden.py --filter gemma-7b --accelerator L4
"""
import argparse
import sys

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--region", default=REGION)
    ap.add_argument("--filter", default="gemma",
                    help="Texto para filtrar los modelos de Model Garden")
    ap.add_argument("--accelerator", default=None,
                    help="Filtra las opciones de despliegue por acelerador, p.ej. L4")
    ap.add_argument("--transport", default="rest", choices=["rest", "grpc"],
                    help="rest usa HTTPS normal (como gcloud) y respeta el proxy del "
                         "sistema; grpc abre su propio canal y suele ser lo que bloquean "
                         "los firewalls corporativos. Por eso el default es rest.")
    ap.add_argument("--model", default=None,
                    help="Si ya conoces el id exacto, salta el listado y muestra "
                         "solo sus opciones de despliegue")
    args = ap.parse_args()

    vertexai.init(project=args.project, location=args.region,
                  api_transport=args.transport)
    print(f"(transporte: {args.transport})\n")

    if args.model:
        modelos = [args.model]
    else:
        print(f"== Modelos desplegables en Model Garden que contienen '{args.filter}' ==\n")
        modelos = model_garden.list_deployable_models(model_filter=args.filter)
        if not modelos:
            print("Ningun modelo nativo coincide. Probando tambien modelos de Hugging Face...")
            modelos = model_garden.list_deployable_models(
                model_filter=args.filter, list_hf_models=True)
        for m in modelos:
            print(f"  {m}")
        if not modelos:
            print("\nNada encontrado. Revisa el filtro o mira el catalogo en la consola:")
            print("  https://console.cloud.google.com/vertex-ai/model-garden")
            return 1
        print()

    # Opciones de despliegue verificadas de cada candidato relevante
    candidatos = [m for m in modelos if "7b" in m.lower() and "it" in m.lower()] or modelos[:3]
    for nombre in candidatos:
        print("=" * 78)
        print(f"Opciones de despliegue verificadas para: {nombre}")
        print("=" * 78)
        try:
            modelo = model_garden.OpenModel(nombre)
        except Exception as e:                                   # noqa: BLE001
            print(f"  No se pudo abrir el model card: {e}\n")
            continue

        # EULA: Gemma es un modelo con licencia; el proyecto debe aceptarla.
        try:
            aceptado = modelo.check_license_agreement_status()
            print(f"  Licencia aceptada por el proyecto: {aceptado}")
            if not aceptado:
                print("  -> Para aceptarla desde codigo:")
                print("     modelo.accept_model_license_agreement()")
                print("     (o deploy(accept_eula=True), que la acepta al desplegar)")
        except Exception as e:                                   # noqa: BLE001
            print(f"  No se pudo consultar el estado de la licencia: {e}")

        try:
            opciones = modelo.list_deploy_options(
                concise=True,
                accelerator_type_filter=args.accelerator,
            )
            print("\n" + str(opciones) + "\n")
        except Exception as e:                                   # noqa: BLE001
            print(f"  No se pudieron listar las opciones de despliegue: {e}\n")

    print("Copia de la salida de arriba: el id exacto del modelo, el "
          "serving_container_image_uri, el machine_type y el accelerator_type.")
    print("Esos cuatro valores son los argumentos de 02-deploy-model-garden-base.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
