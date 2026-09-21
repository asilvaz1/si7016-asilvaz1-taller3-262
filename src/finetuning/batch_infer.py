"""
batch_infer.py - Genera las respuestas del modelo AFINADO sin necesidad de un
endpoint de Vertex.

Por que existe. El despliegue del modelo fusionado en un endpoint con vLLM se
trabo dos veces: unas por "Model server exited unexpectedly" (la L4 de 24 GB no
alcanza para 17 GB de pesos bf16 mas los grafos CUDA y la cache KV de 2048
tokens) y otras por timeout de despliegue. Pero el entregable del taller no es
un endpoint vivo: son las 60 respuestas del modelo afinado sobre las 20
preguntas de evaluacion con las tres tecnicas de prompt engineering. Eso se
puede producir con un Custom Job de Vertex, que usa la cuota
custom_model_training_nvidia_l4_gpus (la aprobada, la que ya funciono en el
entrenamiento y en la fusion) y no pasa por health checks ni por el reloj de
despliegue.

Aqui no hay servidor de inferencia: se carga el modelo con transformers y se
genera en un solo proceso, secuencia por secuencia. Mas lento por token que
vLLM, irrelevante para 60 generaciones.

Dos modos de carga:

  1. Por defecto, el modelo FUSIONADO desde GCS en bf16. Son ~17.1 GB en una L4
     de 24 GB: con batch 1 y secuencias cortas entra con holgura, porque aqui no
     hay cache KV persistente ni grafos CUDA compitiendo por la memoria.

  2. --from_adapters: el modelo base en 4 bits mas los adaptadores LoRA. Son
     ~200 MB desde el bucket en vez de 17 GB, y no toca el modelo fusionado.
     Es la salida de emergencia si la copia fusionada quedara incompleta.

El prompt de cada tecnica es copia literal de PLANTILLAS en
src/deploy/03-predict-endpoint.py, que es con lo que se midio el modelo base. Si
el prompt cambiara, parte de la diferencia medida entre base y afinado vendria
del prompt y no del fine-tuning.

El esquema de salida es el de la seccion 4.2 del HANDOFF. Como se decodifican
solo los tokens nuevos, no hay eco del prompt que limpiar: prediction y
prediction_raw salen iguales, que es lo que el HANDOFF preve para un sistema sin
eco.

Uso dentro del job de Vertex (lo arma submit_infer_job.py):
    python3 batch_infer.py --model_dir /gcs/<bucket>/normas-ruido-merged \
        --out_dir /gcs/<bucket>/inferencia
"""

import argparse
import json
import os
import shutil
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Plantillas: copia literal de src/deploy/03-predict-endpoint.py -----------
# No editar aqui sin editar alla, prompts/ y volver a medir la linea base.
PLANTILLAS = {
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

# Mismos tokens de control que limpia 03-predict-endpoint.py. skip_special_tokens
# ya deberia quitarlos, pero si el tokenizer del modelo fusionado no marcara
# alguno como especial, apareceria en el texto y ROUGE lo contaria.
TOKENS_CONTROL = ["<end_of_turn>", "<start_of_turn>", "<eos>", "<bos>", "<pad>"]

# Nombre corto de archivo por tecnica, igual que en results/ de la linea base.
ALIAS = {"zero-shot": "zero-shot", "few-shot": "few-shot", "chain-of-thought": "cot"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="/gcs/asilvaz1taller3/normas-ruido-merged",
                   help="Ruta FUSE del modelo fusionado. Ignorada con --from_adapters.")
    p.add_argument("--from_adapters", action="store_true",
                   help="Carga el modelo base en 4 bits y le aplica los adaptadores.")
    p.add_argument("--base_model", default="google/gemma-7b-it")
    p.add_argument("--adapter_dir", default="/gcs/asilvaz1taller3/normas-ruido-lora")
    p.add_argument("--dataset", default="/trainer/normas_ruido_eval.jsonl")
    p.add_argument("--out_dir", default="/gcs/asilvaz1taller3/inferencia")
    p.add_argument("--local_dir", default="/tmp/inferencia",
                   help="Se escribe aqui y despues se copia al bucket.")
    p.add_argument("--system", default="finetuning",
                   choices=["base", "finetuning", "rag"])
    p.add_argument("--techniques", default="zero-shot,few-shot,chain-of-thought")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="float32 para la ruta de CPU (las n1 no tienen bf16 nativo y "
                        "torch lo emula). float16 para una T4: Turing no tiene bf16.")
    p.add_argument("--limit", type=int, default=0, help="0 = todas las preguntas")
    return p.parse_args()


def limpiar(texto):
    for token in TOKENS_CONTROL:
        texto = texto.replace(token, "")
    return texto.strip()


def cargar_dataset(ruta, limite):
    if not os.path.isfile(ruta):
        raise SystemExit(f"No existe el dataset {ruta}")
    filas = []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                filas.append(json.loads(linea))
    if limite:
        filas = filas[:limite]
    if not filas:
        raise SystemExit(f"{ruta} no tiene filas")
    return filas


def cargar_modelo(args):
    if args.from_adapters:
        from peft import PeftModel
        from transformers import BitsAndBytesConfig

        print(f"Modo adaptadores: {args.base_model} en 4 bits + {args.adapter_dir}",
              flush=True)
        if not os.path.isdir(args.adapter_dir):
            raise SystemExit(f"No existe {args.adapter_dir}")
        tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
        # El entrenamiento uso NF4 con computo en bf16. En una T4 hay que bajar a
        # fp16: Turing no tiene bf16 y torch lo emula.
        computo = getattr(torch, args.dtype if args.dtype != "float32" else "bfloat16")
        cuantizacion = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=computo,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=cuantizacion,
            torch_dtype=computo,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        modelo = PeftModel.from_pretrained(base, args.adapter_dir)
        return tokenizer, modelo

    print(f"Modo fusionado: {args.model_dir} en {args.dtype}", flush=True)
    if not os.path.isdir(args.model_dir):
        raise SystemExit(
            f"No existe {args.model_dir}. Si el modelo fusionado no esta en el "
            "bucket, relanza este job con --from_adapters."
        )
    contenido = sorted(os.listdir(args.model_dir))
    print("Contenido del modelo:", contenido, flush=True)
    faltantes = []
    if "config.json" not in contenido:
        faltantes.append("config.json")
    if not any(n.endswith(".safetensors") for n in contenido):
        faltantes.append("shards .safetensors")
    if not any(n.startswith("tokenizer") for n in contenido):
        faltantes.append("archivos del tokenizer")
    if faltantes:
        raise SystemExit(
            "El modelo fusionado esta incompleto (falta: " + ", ".join(faltantes) +
            "). Relanza este job con --from_adapters."
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    modelo = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=getattr(torch, args.dtype),
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return tokenizer, modelo


def main():
    args = parse_args()
    tecnicas = [t.strip() for t in args.techniques.split(",") if t.strip()]
    for t in tecnicas:
        if t not in PLANTILLAS:
            raise SystemExit(f"Tecnica desconocida: {t}. Validas: {list(PLANTILLAS)}")

    print(f"CUDA disponible: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    datos = cargar_dataset(args.dataset, args.limit)
    print(f"Preguntas: {len(datos)} | tecnicas: {tecnicas} | "
          f"generaciones totales: {len(datos) * len(tecnicas)}", flush=True)

    t0 = time.time()
    tokenizer, modelo = cargar_modelo(args)
    modelo.eval()
    print(f"Modelo cargado en {time.time() - t0:,.0f}s", flush=True)

    dispositivo = next(modelo.parameters()).device
    todas = []
    errores = 0

    for tecnica in tecnicas:
        plantilla = PLANTILLAS[tecnica]
        filas = []
        t_tec = time.time()
        for i, d in enumerate(datos, 1):
            prompt = plantilla.format(q=d["question"])
            try:
                entradas = tokenizer(prompt, return_tensors="pt").to(dispositivo)
                with torch.no_grad():
                    salida = modelo.generate(
                        **entradas,
                        max_new_tokens=args.max_new_tokens,
                        # Greedy: equivale al temperature=0.0 con que se midio la
                        # linea base. Sin muestreo la corrida es reproducible.
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                # Solo los tokens NUEVOS: asi no hay eco del prompt que limpiar.
                nuevos = salida[0][entradas["input_ids"].shape[1]:]
                texto = limpiar(tokenizer.decode(nuevos, skip_special_tokens=True))
            except Exception as error:  # noqa: BLE001
                errores += 1
                texto = f"__ERROR__ {type(error).__name__}: {error}"
                print(f"  [{tecnica} {i}/{len(datos)}] ERROR: {error}", flush=True)

            filas.append({
                "question": d["question"],
                "reference": d.get("answer", ""),
                "prediction": texto,
                "prediction_raw": texto,
                "technique": tecnica,
                "system": args.system,
                "source_file": d.get("source_file", ""),
                "section_label": d.get("section_label", ""),
            })
            if i % 5 == 0 or i == len(datos):
                print(f"  [{tecnica}] {i}/{len(datos)} "
                      f"({time.time() - t_tec:,.0f}s)", flush=True)

        todas.extend(filas)
        print(f"{tecnica}: {len(filas)} respuestas en {time.time() - t_tec:,.0f}s",
              flush=True)

    # --- Escritura: local primero, despues copia al bucket -------------------
    os.makedirs(args.local_dir, exist_ok=True)
    salidas = []

    for tecnica in tecnicas:
        filas = [x for x in todas if x["technique"] == tecnica]
        nombre = f"respuestas-{args.system}-{ALIAS[tecnica]}.jsonl"
        ruta = os.path.join(args.local_dir, nombre)
        with open(ruta, "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(x, ensure_ascii=False) for x in filas) + "\n")
        salidas.append(ruta)

    consolidado = os.path.join(args.local_dir, f"respuestas-{args.system}.jsonl")
    with open(consolidado, "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(x, ensure_ascii=False) for x in todas) + "\n")
    salidas.append(consolidado)

    os.makedirs(args.out_dir, exist_ok=True)
    for ruta in salidas:
        destino = os.path.join(args.out_dir, os.path.basename(ruta))
        shutil.copy2(ruta, destino)
        print(f"  copiado {destino}", flush=True)

    vacias = sum(1 for x in todas if not x["prediction"].strip())
    largo = sum(len(x["prediction"]) for x in todas) / max(1, len(todas))
    print("\n== Resumen ==")
    print(f"  filas            : {len(todas)}")
    print(f"  errores de gen.  : {errores}")
    print(f"  respuestas vacias: {vacias}")
    print(f"  largo medio      : {largo:,.0f} caracteres")
    print(f"  bucket           : {args.out_dir}")
    if errores:
        raise SystemExit(f"Termino con {errores} generaciones fallidas.")


if __name__ == "__main__":
    main()
