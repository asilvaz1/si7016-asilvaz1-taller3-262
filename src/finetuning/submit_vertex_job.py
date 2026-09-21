"""
submit_vertex_job.py - Envia train_gemma.py como Custom Training Job de
Vertex AI, con la imagen construida por 00-build-image.ps1.

Base: ft-gemma-vertex/submit_vertex_job.py del repositorio del curso. Cambios:

1. Los valores del proyecto (proyecto, region, bucket, imagen) vienen cableados
   como default, igual que en src/deploy, para no tener que pasarlos cada vez.

2. `--transport rest` por defecto. El SDK de Vertex usa gRPC de fabrica y la red
   del campus lo bloquea junto con el TLD .goog (ver src/deploy/README.md). Con
   REST el envio viaja por HTTPS normal, igual que gcloud.

3. `--smoke_test` deja listo el ensayo corto que pide la hoja de ruta: 20 pasos,
   ~15 minutos, centavos de costo, y escribe en una carpeta aparte del bucket
   para no pisar los adaptadores buenos.

4. `--no_sync` para lanzar y soltar la terminal. Con esta opcion el script
   espera solo a que el job quede creado y luego imprime el enlace de la consola.

Por que Custom*Container*TrainingJob y no autopackaging: la imagen de Hugging
Face no esta en la lista blanca de imagenes que Vertex sabe empaquetar solo.
Es la misma razon que da el script del curso.

Uso (desde 13_nlp, con el entorno .venv activado y $env:HF_TOKEN definido):
    python .\si7016-asilvaz1-taller3-262\src\finetuning\submit_vertex_job.py --smoke_test
    python .\si7016-asilvaz1-taller3-262\src\finetuning\submit_vertex_job.py
"""

import argparse
import json
import os
import sys
from datetime import datetime

try:
    from google.cloud import aiplatform
except ModuleNotFoundError as error:
    if error.name == "google":
        raise SystemExit(
            "Falta google-cloud-aiplatform en el Python que ejecuta este script.\n"
            f"Python actual: {sys.executable}\n"
            "Activa el entorno y ejecuta:\n"
            "  .\\.venv\\Scripts\\Activate.ps1\n"
            "  python -m pip install --upgrade google-cloud-aiplatform"
        ) from error
    raise

PROJECT = "si7016-262-nlp"
REGION = "us-central1"          # cuota L4 aprobada; alterna: us-west1
BUCKET = "asilvaz1taller3"
AR_REPO = "si7016-taller3"
IMAGE_TAG = "gemma-7b-it-normas-ruido:latest"
AQUI = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=PROJECT)
    p.add_argument("--region", default=REGION)
    p.add_argument("--bucket", default=BUCKET, help="sin gs://; regional y en --region")
    p.add_argument("--image_uri",
                   default=f"{REGION}-docker.pkg.dev/{PROJECT}/{AR_REPO}/{IMAGE_TAG}",
                   help="La que construye 00-build-image.ps1")
    p.add_argument("--run_name", default=None,
                   help="Nombre de la carpeta de salida en el bucket. Por defecto "
                        "'normas-ruido-lora', o 'smoke-test' con --smoke_test.")
    p.add_argument("--machine_type", default="g2-standard-12",
                   help="g2-standard-12 + NVIDIA_L4 es la combinacion con cuota aprobada")
    p.add_argument("--accelerator_type", default="NVIDIA_L4",
                   choices=["NVIDIA_L4", "NVIDIA_TESLA_T4", "NVIDIA_TESLA_A100"])
    p.add_argument("--accelerator_count", type=int, default=1)
    p.add_argument("--model_name", default="google/gemma-7b-it",
                   help="Alternativa mas liviana si falta memoria: google/gemma-2b-it")
    p.add_argument("--hf_token", default=os.getenv("HF_TOKEN"),
                   help="Token de HF con la licencia de gemma-7b-it aceptada")
    p.add_argument("--max_steps", type=int, default=-1,
                   help="-1 entrena las epocas completas")
    p.add_argument("--num_train_epochs", type=float, default=10)
    p.add_argument("--smoke_test", action="store_true",
                   help="Atajo: max_steps=20 y carpeta de salida aparte")
    p.add_argument("--transport", default="rest", choices=["rest", "grpc"],
                   help="rest usa HTTPS normal (como gcloud) y atraviesa el firewall "
                        "del campus, que bloquea gRPC y el TLD .goog")
    p.add_argument("--no_sync", action="store_true",
                   help="Lanza el job y devuelve la terminal en vez de esperarlo")
    p.add_argument("--dry_run", action="store_true",
                   help="Solo imprime lo que enviaria. No cuesta nada.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.smoke_test:
        args.max_steps = 20
        if args.run_name is None:
            args.run_name = "smoke-test"
    if args.run_name is None:
        args.run_name = "normas-ruido-lora"

    if not args.hf_token and not args.dry_run:
        raise SystemExit(
            "Falta el token de Hugging Face.\n"
            '  $env:HF_TOKEN = "hf_..."   (o pasa --hf_token)\n'
            "Recuerda aceptar la licencia en huggingface.co/google/gemma-7b-it "
            "con la cuenta dueña de ese token."
        )

    bucket_uri = f"gs://{args.bucket}"
    output_dir = f"/gcs/{args.bucket}/{args.run_name}"     # ruta FUSE dentro del contenedor
    output_gs = f"{bucket_uri}/{args.run_name}"            # la misma, vista desde afuera
    sello = datetime.now().strftime("%Y%m%d-%H%M")
    display_name = f"gemma-7b-it-{args.run_name}-{sello}"

    script_args = [
        f"--model_name={args.model_name}",
        "--dataset_path=/trainer/normas_ruido_train.jsonl",
        f"--output_dir={output_dir}",
        f"--max_steps={args.max_steps}",
        f"--num_train_epochs={args.num_train_epochs}",
    ]

    print("== Job que se va a enviar ==")
    print(f"  proyecto   : {args.project}")
    print(f"  region     : {args.region}")
    print(f"  imagen     : {args.image_uri}")
    print(f"  maquina    : {args.machine_type} + {args.accelerator_count} x {args.accelerator_type}")
    print(f"  nombre     : {display_name}")
    print(f"  salida     : {output_gs}")
    print(f"  transporte : {args.transport}")
    print(f"  modo       : {'SMOKE TEST (20 pasos)' if args.smoke_test else 'entrenamiento completo'}")
    print("  args       : " + " ".join(script_args))

    if args.dry_run:
        print("\n--dry_run: no se envio nada.")
        return

    aiplatform.init(project=args.project, location=args.region,
                    staging_bucket=bucket_uri, api_transport=args.transport)

    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=args.image_uri,
    )

    env_vars = {
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_LOG_LEVEL": "INFO",
        # Sin esto Python bufferea stdout cuando no hay terminal: los
        # print() no llegan a los logs hasta que el proceso termina o se
        # llena el buffer, y el job se ve congelado sin estarlo.
        "PYTHONUNBUFFERED": "1",
        "HF_TOKEN": args.hf_token,
    }

    consola = (f"https://console.cloud.google.com/vertex-ai/training/custom-jobs"
               f"?project={args.project}")
    print(f"\nSeguimiento en la consola: {consola}")
    if not args.no_sync:
        print("No cierres esta terminal: el SDK se queda esperando a que el job termine.\n")

    job.run(
        args=script_args,
        replica_count=1,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        environment_variables=env_vars,
        base_output_dir=f"{output_gs}-job-artifacts",
        sync=not args.no_sync,
    )

    if args.no_sync:
        # Con sync=False el envio ocurre en un hilo aparte: hay que esperar a
        # que el recurso exista antes de que el proceso termine.
        job.wait_for_resource_creation()
        print(f"Job creado: {job.resource_name}")
        print("Sigue corriendo en GCP. Puedes cerrar la terminal.")
    else:
        print(f"\nJob terminado. Adaptadores LoRA en: {output_gs}")

    estado = {
        "display_name": display_name,
        "resource_name": getattr(job, "resource_name", None),
        "project": args.project,
        "region": args.region,
        "output_gs": output_gs,
        "smoke_test": args.smoke_test,
        "enviado": sello,
    }
    with open(os.path.join(AQUI, ".job_finetuning.json"), "w", encoding="utf-8") as fh:
        json.dump(estado, fh, indent=2, ensure_ascii=False)

    print("\nComprobar que los adaptadores quedaron en el bucket:")
    print(f"  gcloud storage ls {output_gs}/")


if __name__ == "__main__":
    main()
