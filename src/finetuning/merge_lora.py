"""
merge_lora.py - Fusiona los adaptadores LoRA del fine-tuning con los pesos de
google/gemma-7b-it y deja el modelo fusionado en GCS, listo para que vLLM lo
sirva desde un endpoint de Vertex.

Es el paso que el enunciado del taller describe como "modelo congelado +
adaptadores, estan separados, mezclar".

Base: la logica de fusion de c-deploy-merged.ipynb del repositorio del curso.
Cambios respecto al notebook:

1. Corre como Custom Job de Vertex, no dentro del contenedor JupyterHub. Lo
   envia submit_merge_job.py y se apaga solo al terminar.

2. Corre en CPU por defecto. La fusion es aritmetica de pesos (W = W + BA*s),
   no necesita GPU, y asi no compite por la unica cuota de L4 del proyecto: se
   puede fusionar mientras el endpoint del modelo base sigue desplegado, o
   mientras otro entrenamiento corre.

3. Escribe primero en el disco local del contenedor y despues copia a GCS. El
   modelo fusionado pesa ~17 GB repartidos en varios shards; escribirlos
   directo sobre el montaje FUSE de GCS es mas fragil que copiarlos ya
   completos.

4. Verifica al final que el destino tenga config.json, los shards y el
   tokenizer, porque un modelo incompleto en GCS falla mucho despues, cuando
   vLLM intenta arrancar, y ahi el error no dice que falto un archivo.

Requiere HF_TOKEN: gemma-7b-it sigue siendo gated aunque solo se lea.
"""

import argparse
import os
import shutil
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="google/gemma-7b-it")
    p.add_argument("--adapter_dir", required=True,
                   help="Ruta FUSE de los adaptadores, ej. /gcs/asilvaz1taller3/normas-ruido-lora")
    p.add_argument("--output_dir", required=True,
                   help="Ruta FUSE del modelo fusionado, ej. /gcs/asilvaz1taller3/normas-ruido-merged")
    p.add_argument("--local_dir", default="/tmp/merged",
                   help="Staging en el disco del contenedor antes de copiar a GCS")
    p.add_argument("--device_map", default=None,
                   help="None = CPU (default). Usa 'auto' solo si lanzaste el job con GPU.")
    p.add_argument("--max_shard_size", default="4GB")
    return p.parse_args()


def copiar_a_gcs(origen, destino):
    os.makedirs(destino, exist_ok=True)
    archivos = sorted(os.listdir(origen))
    total = len(archivos)
    for i, nombre in enumerate(archivos, 1):
        ruta_o = os.path.join(origen, nombre)
        if os.path.isdir(ruta_o):
            continue
        mb = os.path.getsize(ruta_o) / 1e6
        t0 = time.time()
        shutil.copy2(ruta_o, os.path.join(destino, nombre))
        print(f"  [{i}/{total}] {nombre}  {mb:,.0f} MB  en {time.time()-t0:,.0f}s", flush=True)


def main():
    args = parse_args()
    print(f"Modelo base : {args.base_model}")
    print(f"Adaptadores : {args.adapter_dir}")
    print(f"Destino     : {args.output_dir}")
    print(f"CUDA        : {torch.cuda.is_available()}")

    if not os.path.isdir(args.adapter_dir):
        raise SystemExit(
            f"No existe {args.adapter_dir}. Revisa que el entrenamiento haya terminado "
            "y que la ruta del bucket sea la correcta."
        )
    contenido = os.listdir(args.adapter_dir)
    print("Contenido del directorio de adaptadores:", contenido)
    if not any(n.startswith("adapter_model") for n in contenido):
        raise SystemExit(
            "No encuentro adapter_model.safetensors en el directorio de adaptadores. "
            "Eso normalmente significa que el entrenamiento no llego al paso de guardado."
        )

    # --- 1. Tokenizer: el que guardo el entrenamiento, no el del Hub ---------
    # Asi el modelo fusionado queda con exactamente el mismo tokenizer con que
    # se entreno, incluido cualquier pad_token que se haya fijado.
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)

    # --- 2. Modelo base en bf16 (no en 4-bit: se va a servir, no a entrenar) -
    print("Cargando el modelo base... (descarga ~17 GB la primera vez)", flush=True)
    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=args.device_map,
    )
    print(f"Modelo base cargado en {time.time()-t0:,.0f}s", flush=True)

    # --- 3. Fusion ------------------------------------------------------------
    print("Aplicando los adaptadores...", flush=True)
    modelo_con_lora = PeftModel.from_pretrained(base_model, args.adapter_dir)
    print("Fusionando en los pesos del modelo base...", flush=True)
    t0 = time.time()
    modelo_fusionado = modelo_con_lora.merge_and_unload()
    print(f"Fusion completa en {time.time()-t0:,.0f}s", flush=True)

    # --- 4. Guardar local y copiar a GCS -------------------------------------
    if os.path.isdir(args.local_dir):
        shutil.rmtree(args.local_dir)
    os.makedirs(args.local_dir, exist_ok=True)

    print(f"Guardando en {args.local_dir}...", flush=True)
    modelo_fusionado.save_pretrained(
        args.local_dir, safe_serialization=True, max_shard_size=args.max_shard_size
    )
    tokenizer.save_pretrained(args.local_dir)
    print("Archivos generados:", sorted(os.listdir(args.local_dir)), flush=True)

    print(f"Copiando a {args.output_dir}...", flush=True)
    copiar_a_gcs(args.local_dir, args.output_dir)

    # --- 5. Verificacion -----------------------------------------------------
    # Un modelo incompleto en GCS no falla aqui: falla 15 minutos despues,
    # cuando vLLM intenta arrancar, y con un mensaje que no menciona el archivo
    # que falta. Por eso se comprueba ahora.
    destino = sorted(os.listdir(args.output_dir))
    print("\nContenido final en GCS:", destino)

    problemas = []
    if "config.json" not in destino:
        problemas.append("falta config.json")
    if not any(n.endswith(".safetensors") for n in destino):
        problemas.append("no hay shards .safetensors")
    if not any(n.startswith("tokenizer") for n in destino):
        problemas.append("faltan los archivos del tokenizer")

    origen_bytes = sum(os.path.getsize(os.path.join(args.local_dir, n))
                       for n in os.listdir(args.local_dir)
                       if os.path.isfile(os.path.join(args.local_dir, n)))
    destino_bytes = sum(os.path.getsize(os.path.join(args.output_dir, n))
                        for n in destino
                        if os.path.isfile(os.path.join(args.output_dir, n)))
    print(f"Bytes local: {origen_bytes:,} | en GCS: {destino_bytes:,}")
    if destino_bytes != origen_bytes:
        problemas.append("el tamano en GCS no coincide con el local")

    if problemas:
        raise SystemExit("La copia quedo incompleta: " + "; ".join(problemas))

    print(f"\nListo. Modelo fusionado en: {args.output_dir}")
    print("Siguiente paso: subirlo al Model Registry y desplegarlo con vLLM.")


if __name__ == "__main__":
    main()
