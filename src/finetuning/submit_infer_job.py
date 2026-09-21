"""
submit_infer_job.py - Envia batch_infer.py como Custom Job de Vertex, reusando
la misma imagen del entrenamiento con el entrypoint cambiado.

Es el mismo patron de submit_merge_job.py. La diferencia es que este si pide
GPU: generar 60 respuestas de 256 tokens con un modelo de 7B en CPU tardaria
horas, y la L4 lo hace en minutos.

Por que un Custom Job y no el endpoint. El despliegue del modelo fusionado con
vLLM se trabo por memoria de GPU y por timeout de despliegue. El Custom Job usa
la cuota custom_model_training_nvidia_l4_gpus, que es la aprobada y la que ya
funciono en el entrenamiento y en la fusion, no pasa por health checks, y se
apaga solo al terminar. Las 60 respuestas quedan en el bucket.

Costo tipico: g2-standard-12 con 1 x L4 durante ~25 minutos, unos US$0.35.

Uso (desde 13_nlp, con .venv activado y $env:HF_TOKEN definido):
    python .\si7016-asilvaz1-taller3-262\src\finetuning\submit_infer_job.py --dry_run
    python .\si7016-asilvaz1-taller3-262\src\finetuning\submit_infer_job.py

Si el modelo fusionado del bucket estuviera incompleto, el job falla al cargarlo
y lo dice. En ese caso, sin reconstruir la imagen:
    python .\...\submit_infer_job.py --from_adapters

Al terminar, bajar el resultado al repo:
    gcloud storage cp gs://asilvaz1taller3/inferencia/respuestas-finetuning.jsonl results\
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
    p.add_argument("--merged_run", default="normas-ruido-merged",
                   help="Carpeta del bucket con el modelo fusionado")
    p.add_argument("--adapter_run", default="normas-ruido-lora",
                   help="Carpeta del bucket con los adaptadores (solo con --from_adapters)")
    p.add_argument("--out_run", default="inferencia",
                   help="Carpeta del bucket donde quedan los jsonl de respuestas")
    p.add_argument("--from_adapters", action="store_true",
                   help="Carga el modelo base en 4 bits + adaptadores, en vez del fusionado")
    p.add_argument("--base_model", default="google/gemma-7b-it")
    p.add_argument("--system", default="finetuning",
                   choices=["base", "finetuning", "rag"])
    p.add_argument("--techniques", default="zero-shot,few-shot,chain-of-thought")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=0,
                   help="0 = las 20 preguntas. Usa 2 para una prueba de humo barata.")
    # g2-standard-12: 1 x L4 (24 GB de VRAM) y 48 GB de RAM de host. La misma
    # maquina del despliegue que fallaba, pero aqui sin vLLM reservando cache KV
    # ni capturando grafos CUDA, que es lo que no cabia junto a los 17 GB de pesos.
    p.add_argument("--machine_type", default="g2-standard-12")
    p.add_argument("--accelerator_type", default="NVIDIA_L4",
                   choices=["NVIDIA_L4", "NVIDIA_TESLA_T4"])
    p.add_argument("--accelerator_count", type=int, default=1)
    # Ultimo recurso si ninguna region tiene L4 libre. La generacion en CPU de
    # un modelo de 7B es de horas, no de minutos: ver el comentario en main().
    p.add_argument("--no_gpu", action="store_true",
                   help="Corre en CPU. Fuerza --machine_type n1-highmem-16 si no se pide otra.")
    # El bucket del proyecto es regional (us-central1). Si el job se lanza en
    # otra region conviene un bucket de staging alla, para no depender de que
    # Vertex acepte un bucket de staging de otra region.
    p.add_argument("--staging_bucket", default=None,
                   help="Por defecto, el mismo --bucket.")
    p.add_argument("--hf_token", default=os.getenv("HF_TOKEN"))
    p.add_argument("--transport", default="rest", choices=["rest", "grpc"])
    p.add_argument("--no_sync", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    # El modelo fusionado se lee del bucket y no necesita token. El modo
    # adaptadores descarga gemma-7b-it, que sigue siendo gated.
    if args.from_adapters and not args.hf_token and not args.dry_run:
        raise SystemExit('Falta el token: $env:HF_TOKEN = "hf_..."')

    bucket_uri = f"gs://{args.bucket}"
    staging_uri = f"gs://{args.staging_bucket or args.bucket}"
    out_dir = f"/gcs/{args.bucket}/{args.out_run}"

    # En CPU: fp32, porque los procesadores de las n1 no tienen bf16 nativo y
    # torch lo emula, que es peor que usar fp32 directamente. gemma-7b en fp32
    # son ~34 GB de RAM, asi que la maquina tiene que ser highmem.
    if args.no_gpu and args.machine_type == "g2-standard-12":
        args.machine_type = "n1-highmem-16"
    sello = datetime.now().strftime("%Y%m%d-%H%M")
    display_name = f"infer-{args.system}-{sello}"

    script_args = [
        f"--out_dir={out_dir}",
        f"--system={args.system}",
        f"--techniques={args.techniques}",
        f"--max_new_tokens={args.max_new_tokens}",
    ]
    if args.limit:
        script_args.append(f"--limit={args.limit}")
    if args.no_gpu:
        script_args.append("--dtype=float32")
    elif args.accelerator_type == "NVIDIA_TESLA_T4":
        # Turing no tiene bf16. Y una T4 son 16 GB: solo cabe la ruta de
        # adaptadores en 4 bits, no el fusionado en bf16 (~17 GB).
        script_args.append("--dtype=float16")
    if args.from_adapters:
        script_args.append("--from_adapters")
        script_args.append(f"--adapter_dir=/gcs/{args.bucket}/{args.adapter_run}")
        script_args.append(f"--base_model={args.base_model}")
    else:
        script_args.append(f"--model_dir=/gcs/{args.bucket}/{args.merged_run}")

    origen = (f"{bucket_uri}/{args.adapter_run} (adaptadores en 4 bits)"
              if args.from_adapters
              else f"{bucket_uri}/{args.merged_run} (fusionado en bf16)")

    hardware = ("sin GPU (CPU)" if args.no_gpu
                else f"{args.accelerator_count} x {args.accelerator_type}")

    print("== Job de inferencia ==")
    print(f"  imagen   : {args.image_uri}")
    print(f"  region   : {args.region}")
    print(f"  maquina  : {args.machine_type}, {hardware}")
    print(f"  staging  : {staging_uri}")
    print(f"  modelo   : {origen}")
    print(f"  salida   : {bucket_uri}/{args.out_run}")
    print("  args     : " + " ".join(script_args))

    if args.dry_run:
        print("\n--dry_run: no se envio nada.")
        return

    aiplatform.init(project=args.project, location=args.region,
                    staging_bucket=staging_uri, api_transport=args.transport)

    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=args.image_uri,
        # La imagen tiene ENTRYPOINT hacia train_gemma.py; aqui se cambia por el
        # script de inferencia, que viaja en la misma imagen.
        command=["python3", "/trainer/batch_infer.py"],
    )

    consola = (f"https://console.cloud.google.com/vertex-ai/training/custom-jobs"
               f"?project={args.project}")
    print(f"\nSeguimiento: {consola}\n")

    entorno = {
        "HF_HOME": "/root/.cache/huggingface",
        # Sin esto los print() no llegan a los logs hasta que el proceso termina
        # y el job se ve congelado sin estarlo.
        "PYTHONUNBUFFERED": "1",
    }
    if args.hf_token:
        entorno["HF_TOKEN"] = args.hf_token

    run_kwargs = dict(
        args=script_args,
        replica_count=1,
        machine_type=args.machine_type,
        environment_variables=entorno,
        base_output_dir=f"{staging_uri}/{args.out_run}-job-artifacts",
        sync=not args.no_sync,
    )
    if not args.no_gpu:
        run_kwargs["accelerator_type"] = args.accelerator_type
        run_kwargs["accelerator_count"] = max(1, args.accelerator_count)

    job.run(**run_kwargs)

    if args.no_sync:
        job.wait_for_resource_creation()
        print(f"Job creado: {job.resource_name}")
    else:
        print(f"\nRespuestas en: {bucket_uri}/{args.out_run}")
        print("Bajarlas al repo:")
        print(f"  gcloud storage cp {bucket_uri}/{args.out_run}/respuestas-{args.system}.jsonl results\\")


if __name__ == "__main__":
    main()
