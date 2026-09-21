"""
submit_merge_job.py - Envia merge_lora.py como Custom Job de Vertex, reusando
la misma imagen del entrenamiento con el entrypoint cambiado.

Por que como job y no en tu maquina: la fusion necesita cargar gemma-7b-it en
bf16, o sea ~17 GB de descarga y ~18 GB de RAM, y escribir otros ~17 GB. En
GCP eso pasa en minutos y dentro de la misma red del bucket.

Por que en CPU: la fusion es aritmetica de pesos, no entrenamiento. Corriendola
en n1-highmem-8 no consume la cuota de L4, asi que puede correr aunque el
endpoint del modelo base siga desplegado. Si prefieres GPU, pasa
--accelerator_type NVIDIA_L4 --machine_type g2-standard-12 --device_map auto.

Uso (desde 13_nlp, con .venv activado y $env:HF_TOKEN definido):
    python .\si7016-asilvaz1-taller3-262\src\finetuning\submit_merge_job.py --dry_run
    python .\si7016-asilvaz1-taller3-262\src\finetuning\submit_merge_job.py
"""

import argparse
import os
import sys
from datetime import datetime

try:
    from google.cloud import aiplatform
except ModuleNotFoundError as error:
    if error.name == "google":
        raise SystemExit(
            "Falta google-cloud-aiplatform en el Python que ejecuta este script.\n"
            f"Python actual: {sys.executable}"
        ) from error
    raise

PROJECT = "si7016-262-nlp"
REGION = "us-central1"
BUCKET = "asilvaz1taller3"
AR_REPO = "si7016-taller3"
IMAGE_TAG = "gemma-7b-it-normas-ruido:latest"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=PROJECT)
    p.add_argument("--region", default=REGION)
    p.add_argument("--bucket", default=BUCKET)
    p.add_argument("--image_uri",
                   default=f"{REGION}-docker.pkg.dev/{PROJECT}/{AR_REPO}/{IMAGE_TAG}")
    p.add_argument("--adapter_run", default="normas-ruido-lora",
                   help="Carpeta del bucket con los adaptadores del entrenamiento")
    p.add_argument("--merged_run", default="normas-ruido-merged",
                   help="Carpeta del bucket donde queda el modelo fusionado")
    p.add_argument("--base_model", default="google/gemma-7b-it")
    p.add_argument("--machine_type", default="n1-highmem-8",
                   help="52 GB de RAM, suficiente para gemma-7b en bf16. Sin GPU.")
    p.add_argument("--accelerator_type", default=None,
                   choices=[None, "NVIDIA_L4", "NVIDIA_TESLA_T4"])
    p.add_argument("--accelerator_count", type=int, default=0)
    p.add_argument("--device_map", default=None,
                   help="Dejalo vacio para CPU. 'auto' solo si pediste GPU.")
    p.add_argument("--hf_token", default=os.getenv("HF_TOKEN"))
    p.add_argument("--transport", default="rest", choices=["rest", "grpc"])
    p.add_argument("--no_sync", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.hf_token and not args.dry_run:
        raise SystemExit('Falta el token: $env:HF_TOKEN = "hf_..."')

    bucket_uri = f"gs://{args.bucket}"
    adapter_dir = f"/gcs/{args.bucket}/{args.adapter_run}"
    output_dir = f"/gcs/{args.bucket}/{args.merged_run}"
    sello = datetime.now().strftime("%Y%m%d-%H%M")
    display_name = f"merge-lora-{args.merged_run}-{sello}"

    script_args = [
        f"--base_model={args.base_model}",
        f"--adapter_dir={adapter_dir}",
        f"--output_dir={output_dir}",
    ]
    if args.device_map:
        script_args.append(f"--device_map={args.device_map}")

    acelerador = (f"{args.accelerator_count} x {args.accelerator_type}"
                  if args.accelerator_type else "sin GPU (CPU)")

    print("== Job de fusion ==")
    print(f"  imagen      : {args.image_uri}")
    print(f"  maquina     : {args.machine_type}, {acelerador}")
    print(f"  adaptadores : {bucket_uri}/{args.adapter_run}")
    print(f"  salida      : {bucket_uri}/{args.merged_run}")
    print("  args        : " + " ".join(script_args))

    if args.dry_run:
        print("\n--dry_run: no se envio nada.")
        return

    aiplatform.init(project=args.project, location=args.region,
                    staging_bucket=bucket_uri, api_transport=args.transport)

    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=args.image_uri,
        # La imagen tiene ENTRYPOINT hacia train_gemma.py; aqui se cambia por
        # el script de fusion, que viaja en la misma imagen.
        command=["python3", "/trainer/merge_lora.py"],
    )

    consola = (f"https://console.cloud.google.com/vertex-ai/training/custom-jobs"
               f"?project={args.project}")
    print(f"\nSeguimiento: {consola}\n")

    run_kwargs = dict(
        args=script_args,
        replica_count=1,
        machine_type=args.machine_type,
        environment_variables={
            "HF_HOME": "/root/.cache/huggingface",
            # Sin esto Python bufferea stdout cuando no hay terminal: los
            # print() no llegan a los logs hasta que el proceso termina o se
            # llena el buffer, y el job se ve congelado sin estarlo.
            "PYTHONUNBUFFERED": "1",
            "HF_TOKEN": args.hf_token,
        },
        base_output_dir=f"{bucket_uri}/{args.merged_run}-job-artifacts",
        sync=not args.no_sync,
    )
    if args.accelerator_type:
        run_kwargs["accelerator_type"] = args.accelerator_type
        run_kwargs["accelerator_count"] = max(1, args.accelerator_count)

    job.run(**run_kwargs)

    if args.no_sync:
        job.wait_for_resource_creation()
        print(f"Job creado: {job.resource_name}")
    else:
        print(f"\nModelo fusionado en: {bucket_uri}/{args.merged_run}")
        print("Comprobar:")
        print(f"  gcloud storage ls {bucket_uri}/{args.merged_run}/")


if __name__ == "__main__":
    main()
