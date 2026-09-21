"""
train_gemma.py - Fine-tuning QLoRA/SFT de google/gemma-7b-it sobre el dataset
de 100 pares pregunta-respuesta construido desde las normas ISO/UNE/BS de
medicion de ruido (taller 3, SI7016).

Base: ft-gemma-vertex/train_gemma.py del repositorio del curso
(si7016-262/class05/apoyo-al-taller3). Se conserva la misma imagen base, el
mismo stack (transformers 4.42 + peft + trl) y la misma estructura del
pipeline. Lo que cambia respecto al original esta listado abajo, para que la
declaracion de etica pueda decir con precision que se reuso y que se aporto.

CAMBIOS RESPECTO AL SCRIPT DEL CURSO
------------------------------------
1. Dataset: en vez de `knkarthick/samsum` descargado de Hugging Face, lee un
   JSONL local con el esquema del taller (question, answer, source_file,
   source_doc_title, section_kind, section_label). El archivo viaja dentro de
   la imagen Docker, asi que el job no depende de la red ni de permisos de
   lectura sobre el bucket.

2. `format_example`: el turno de usuario reproduce EXACTAMENTE el prompt
   zero-shot con el que ya se midio el modelo base
   (prompts/zero-shot.md -> results/respuestas-base.jsonl). Esto importa para
   la comparacion del taller: si el modelo afinado se entrenara con un prompt
   distinto al que se uso para evaluar el modelo base, parte de la diferencia
   medida seria por el cambio de prompt y no por el fine-tuning.

3. Hiperparametros ajustados al tamano real del dataset. Con 80 ejemplos de
   entrenamiento, el default del curso (batch 2 x acumulacion 8 = lote efectivo
   16, 3 epocas) da 5 pasos de optimizador por epoca y 15 en total: no alcanza
   para que los adaptadores LoRA aprendan nada. Aqui el default es lote
   efectivo 8 y 10 epocas -> 10 pasos por epoca, 100 en total. Sigue costando
   pocos minutos de L4.

4. Se agrega el token de fin de secuencia al final de cada ejemplo, para que el
   modelo aprenda a terminar la respuesta en vez de seguir generando.

5. Al terminar escribe `training_config.json` junto a los adaptadores, con los
   hiperparametros y el conteo de ejemplos: es la evidencia de la corrida para
   el README de la entrega.

Requisito: gemma-7b-it es un modelo "gated" en Hugging Face. Hay que aceptar la
licencia en https://huggingface.co/google/gemma-7b-it con la cuenta dueña del
token, y pasar el token al job con HF_TOKEN (lo hace submit_vertex_job.py).

Uso dentro del job de Vertex AI (lo arma submit_vertex_job.py):
    python3 train_gemma.py --output_dir /gcs/<bucket>/gemma-7b-it-normas-ruido-lora
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# Texto del turno de usuario, identico al de prompts/zero-shot.md. No editar
# aqui sin editar tambien ese archivo y src/deploy/03-predict-endpoint.py: los
# tres tienen que decir lo mismo o la comparacion base vs FT deja de ser justa.
SYSTEM_INSTRUCTION = (
    "You are an expert on ISO, UNE and BS acoustics standards for noise "
    "measurement. Answer the question precisely and cite the clause when you "
    "know it."
)


def build_user_turn(question: str) -> str:
    """Turno de usuario en el formato exacto del prompt zero-shot."""
    return f"{SYSTEM_INSTRUCTION}\n\nQuestion: {question}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="google/gemma-7b-it",
                   help="Alternativa mas liviana si falta memoria: google/gemma-2b-it")
    p.add_argument("--dataset_path", default="/trainer/normas_ruido_train.jsonl",
                   help="JSONL con las columnas question/answer. Por defecto el que "
                        "viene dentro de la imagen; se puede apuntar a "
                        "/gcs/<bucket>/... para iterar sin reconstruir la imagen.")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--num_train_epochs", type=float, default=10,
                   help="10 epocas sobre 80 ejemplos = 100 pasos con el lote efectivo "
                        "por defecto. Ver el punto 3 del encabezado.")
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--max_steps", type=int, default=-1,
                   help="-1 entrena las epocas completas. Usa 20 para el smoke test.")
    p.add_argument("--seed", type=int, default=42,
                   help="El mismo seed del split del dataset, por consistencia.")
    p.add_argument("--output_dir",
                   default=os.environ.get("AIP_MODEL_DIR", "./gemma-7b-it-normas-ruido-lora"))
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Modelo base  : {args.model_name}")
    print(f"Dataset      : {args.dataset_path}")
    print(f"Salida (LoRA): {args.output_dir}")

    print("CUDA disponible:", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # --- 1. Modelo base en 4-bit (identico al script del curso) --------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    print("Cargando y cuantizando el modelo base a 4 bits. Este paso tarda\n"
          "varios minutos y no imprime nada mientras trabaja.", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Gemma trae padding_side="left", que es lo correcto para generar pero no
    # para entrenar: SFTTrainer lo advierte en el log. Con relleno a la
    # izquierda, las etiquetas se desplazan respecto a las posiciones que el
    # modelo predice y el entrenamiento aprende de tokens corridos.
    tokenizer.padding_side = "right"

    print("Memoria GPU tras cargar el modelo (GB):",
          round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else "N/A",
          flush=True)

    # --- 2. Adaptadores LoRA (mismos target_modules que el script del curso) -
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 3. Dataset de normas + chat template de Gemma -----------------------
    if not os.path.exists(args.dataset_path):
        raise SystemExit(
            f"No existe el dataset en {args.dataset_path}. Si construiste la imagen "
            "sin copiar el JSONL, vuelve a correr 00-build-image.ps1."
        )

    raw_dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    print(f"Ejemplos de entrenamiento: {len(raw_dataset)}", flush=True)

    faltantes = [c for c in ("question", "answer") if c not in raw_dataset.column_names]
    if faltantes:
        raise SystemExit(
            f"Al dataset le faltan las columnas {faltantes}. Columnas encontradas: "
            f"{raw_dataset.column_names}"
        )

    def format_example(example):
        # Gemma no admite rol 'system' en su chat template: la instruccion va
        # dentro del turno de usuario. El rol 'assistant' lo renderiza la
        # plantilla como 'model', que es la etiqueta propia de Gemma.
        messages = [
            {"role": "user", "content": build_user_turn(example["question"])},
            {"role": "assistant", "content": example["answer"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        # La plantilla cierra con <end_of_turn> pero no con <eos>. Sin el token
        # de fin, el modelo afinado tiende a seguir escribiendo despues de la
        # respuesta, que es justo el ruido que castiga ROUGE.
        if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
            text = text + tokenizer.eos_token
        return {"text": text}

    dataset = raw_dataset.map(format_example, remove_columns=raw_dataset.column_names)
    print("\n--- Ejemplo formateado (completo) ---")
    print(dataset[0]["text"])
    print("--- fin del ejemplo ---\n", flush=True)

    # --- 4. Entrenamiento con SFTTrainer -------------------------------------
    lote_efectivo = args.per_device_train_batch_size * args.gradient_accumulation_steps
    pasos_por_epoca = max(1, len(dataset) // lote_efectivo)
    print(f"Lote efectivo: {lote_efectivo} | pasos por epoca: ~{pasos_por_epoca} "
          f"| pasos totales estimados: ~{int(pasos_por_epoca * args.num_train_epochs)}",
          flush=True)

    local_ckpt_dir = "/tmp/gemma-7b-it-normas-ruido-ckpts"
    sft_config = SFTConfig(
        output_dir=local_ckpt_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        seed=args.seed,
        bf16=True,
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
    )
    resultado = trainer.train()

    # --- 5. Guardar SOLO los adaptadores LoRA, directo a GCS -----------------
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # --- 6. Evidencia de la corrida para el README de la entrega -------------
    metadatos = {
        "model_name": args.model_name,
        "dataset_path": args.dataset_path,
        "n_ejemplos": len(dataset),
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lote_efectivo": lote_efectivo,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "prompt_usuario": build_user_turn("{q}"),
    }
    if resultado is not None and getattr(resultado, "metrics", None):
        metadatos["metricas_entrenamiento"] = resultado.metrics
    with open(os.path.join(args.output_dir, "training_config.json"), "w", encoding="utf-8") as fh:
        json.dump(metadatos, fh, indent=2, ensure_ascii=False)

    print(f"Adaptadores guardados en: {args.output_dir}", flush=True)
    print("Contenido esperado: adapter_model.safetensors, adapter_config.json, "
          "tokenizer.* y training_config.json")


if __name__ == "__main__":
    main()
