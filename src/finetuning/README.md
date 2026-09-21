# Fine-tuning (Fase 2)

Runbook del entrenamiento. Continuación de `src/deploy/README.md`, que dejó
medida la línea base del modelo puro.

## Por qué Vertex y no una VM

Tres razones, en orden de peso:

1. El Custom Training Job se apaga solo cuando termina. Una VM con L4 cobra
   US$0.56/hora hasta que alguien la apague, y ese es el único riesgo real de
   costo del taller.
2. La cuota aprobada es `custom_model_training_nvidia_l4_gpus`, que es de
   Vertex Training. Compute Engine tiene su propia cuota de GPU, distinta y no
   solicitada.
3. La imagen base de Hugging Face para GCP ya trae torch, transformers, peft,
   trl y bitsandbytes. El camino de VM del repo del curso (`ft-gemma-on-vm/`)
   es en realidad el contenedor JupyterHub de la clase 04, con la imagen
   `tensorflow-notebook`, que no trae nada de eso y hay que instalarlo a mano.

La VM con vLLM sigue siendo relevante, pero en la Fase 3 y para **servir** el
modelo afinado, no para entrenarlo.

## Qué hay en esta carpeta

| Archivo | Qué es |
| --- | --- |
| `train_gemma.py` | El entrenamiento. Adaptado de `ft-gemma-vertex/train_gemma.py` del curso; el encabezado lista los cinco cambios |
| `Dockerfile` | Imagen base de HF + el script + el dataset adentro |
| `00-build-image.ps1` | Prepara el contexto de build, llama a Cloud Build y limpia |
| `submit_vertex_job.py` | Envía el job a Vertex |

## Valores del proyecto

Ya están cableados como default, no hay que pasarlos.

| | |
| --- | --- |
| Proyecto | `si7016-262-nlp` |
| Región | `us-central1` (cuota L4 aprobada; alterna `us-west1`) |
| Bucket | `gs://asilvaz1taller3` |
| Imagen | `us-central1-docker.pkg.dev/si7016-262-nlp/si7016-taller3/gemma-7b-it-normas-ruido:latest` |

## Paso 0: terminal y token

Desde PowerShell, en la carpeta `13_nlp`:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
$env:HF_TOKEN = "hf_..."
```

El token nunca va escrito dentro de un archivo. Y la licencia de
`google/gemma-7b-it` tiene que estar aceptada en Hugging Face con la cuenta
dueña de ese token, o la descarga falla con 401/403.

## Paso 1: construir la imagen

```powershell
powershell -ExecutionPolicy Bypass -File .\si7016-asilvaz1-taller3-262\src\finetuning\00-build-image.ps1
```

Tarda entre 5 y 12 minutos la primera vez. El build ocurre en GCP, no en tu
máquina, y viaja por HTTPS, así que funciona aunque la red del campus bloquee
gRPC.

Hay que repetirlo cada vez que cambie `train_gemma.py` o el dataset.

## Paso 2: ensayo en seco

No envía nada y no cuesta nada. Solo imprime el job que armaría:

```powershell
python .\si7016-asilvaz1-taller3-262\src\finetuning\submit_vertex_job.py --dry_run
```

## Paso 3: smoke test de 20 pasos

Este paso existe para descubrir los errores baratos: que la imagen arranque,
que la GPU aparezca, que el token de HF sirva, que el dataset se lea y que los
adaptadores lleguen al bucket. Cuesta centavos.

```powershell
python .\si7016-asilvaz1-taller3-262\src\finetuning\submit_vertex_job.py --smoke_test
```

Escribe en `gs://asilvaz1taller3/smoke-test/`, aparte de la corrida buena.

Pasa si al final ves los adaptadores en el bucket:

```powershell
gcloud storage ls gs://asilvaz1taller3/smoke-test/
```

Deberían estar `adapter_model.safetensors`, `adapter_config.json`, los archivos
del tokenizer y `training_config.json`.

## Paso 4: entrenamiento completo

```powershell
python .\si7016-asilvaz1-taller3-262\src\finetuning\submit_vertex_job.py
```

Entre 30 y 60 minutos. Si prefieres soltar la terminal y seguir el avance desde
la consola, agrega `--no_sync`.

Los adaptadores quedan en `gs://asilvaz1taller3/normas-ruido-lora/`.

Consola: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=si7016-262-nlp

## Hiperparámetros y por qué estos

El dataset tiene 80 ejemplos de entrenamiento y 20 de evaluación
(`data/qa/`, split estratificado con seed 42).

| Parámetro | Valor | Razón |
| --- | --- | --- |
| LoRA r / alpha / dropout | 16 / 32 / 0.05 | Igual que el script del curso |
| Lote efectivo | 8 (batch 2 x acumulación 4) | Con el 16 del curso serían 5 pasos por época |
| Épocas | 10 | 10 pasos por época, 100 pasos en total |
| Learning rate | 2e-4, cosine, warmup 3% | Igual que el curso, con scheduler |
| Cuantización | NF4 de 4 bits, cómputo bf16 | Igual que el curso |

El default del curso (3 épocas, lote efectivo 16) daría 15 pasos de optimizador
en total sobre este dataset. No alcanza para que los adaptadores aprendan nada,
y el resultado sería un fine-tuning que se ve idéntico al modelo base.

Longitud de los ejemplos formateados: entre 121 y 330 tokens aproximados,
mediana 223. Cabe de sobra en la ventana por defecto.

## El prompt tiene que ser el mismo de la línea base

El turno de usuario con que se entrena es, palabra por palabra, el prompt
zero-shot con que ya se midió el modelo base (`prompts/zero-shot.md`):

```text
You are an expert on ISO, UNE and BS acoustics standards for noise measurement. Answer the question precisely and cite the clause when you know it.

Question: {q}
```

Si se entrenara con un prompt distinto al de la evaluación, parte de la
diferencia medida entre base y afinado vendría del cambio de prompt y no del
fine-tuning. Cambiar el texto aquí obliga a cambiarlo también en
`prompts/zero-shot.md` y en `src/deploy/03-predict-endpoint.py`.

## Si algo falla

**`COPY failed: file not found in build context` durante el build.** Falta el
`.gcloudignore` de esta carpeta, o alguien le agregó una línea con `.jsonl`.
Cuando `gcloud builds submit` no encuentra un `.gcloudignore`, fabrica uno a
partir del `.gitignore`, y el `.gitignore` de aquí excluye el dataset a
propósito. El `.gcloudignore` existe justamente para anular ese
comportamiento. `00-build-image.ps1` ya comprueba las dos cosas antes de
enviar nada.

**El job se queda callado 20 o 30 minutos después de arrancar.** Mira la línea
`Downloading shards` en los logs. En el smoke test del 20 de septiembre, un
solo shard tardó 26 minutos (698 s/it) porque la descarga cayó al transporte
HTTP normal. La imagen ya incluye `hf_xet`, que es el transporte rápido, así
que no debería repetirse. La GPU cobra mientras espera, así que si vuelve a
pasar vale la pena cancelar y relanzar.

**401 o 403 "gated repo" en el log del job.** El token de Hugging Face no
llegó, o la cuenta dueña del token no aceptó la licencia de
`google/gemma-7b-it`. Comprueba con `echo $env:HF_TOKEN` en la misma ventana
desde la que lanzas el job.

**`WSA Error 11001` o `Failed to resolve ...goog` al enviar.** Es el bloqueo de
gRPC de la red del campus. Los scripts ya usan `--transport rest` por defecto;
si aun así falla, lanza con `--no_sync` y sigue el avance desde la consola web.

**El job se queda esperando GPU.** Congestión de L4 en la región. Tienes cuota
aprobada también en `us-west1`: `--region us-west1`. Ojo, el bucket es regional
y está en `us-central1`, así que en ese caso hay que pasar también un bucket de
la otra región o aceptar la transferencia entre regiones.

## Antes de cerrar el día

El entrenamiento se apaga solo, pero el endpoint del modelo base no:

```powershell
python .\si7016-asilvaz1-taller3-262\src\deploy\04-undeploy-cleanup.py --list
```

## Fase 3: las respuestas del modelo afinado, sin endpoint

El despliegue del modelo fusionado con vLLM se trabó dos veces: por memoria de
GPU (*Model server exited unexpectedly*) y por timeout de despliegue. Pero el
entregable del taller no es un endpoint vivo, son las 60 respuestas del modelo
afinado. Eso se produce con un Custom Job, que usa la cuota de entrenamiento que
ya funcionó dos veces y no pasa por health checks.

| Archivo | Qué |
| --- | --- |
| `batch_infer.py` | Carga el modelo fusionado desde `/gcs` en bf16 y genera las 20 preguntas del eval con las 3 técnicas |
| `submit_infer_job.py` | Lo envía como Custom Job en `g2-standard-12` + 1 L4 |

```powershell
# 1. Reconstruir la imagen: ahora lleva batch_infer.py adentro
powershell -ExecutionPolicy Bypass -File .\si7016-asilvaz1-taller3-262\src\finetuning\00-build-image.ps1

# 2. Ensayo en seco, no envía nada
python .\si7016-asilvaz1-taller3-262\src\finetuning\submit_infer_job.py --dry_run

# 3. El job de verdad: ~25 min, ~US$0.35
python .\si7016-asilvaz1-taller3-262\src\finetuning\submit_infer_job.py

# 4. Bajar el resultado. OJO: results\ es relativo al repo, no a 13_nlp, asi que
#    hay que entrar primero o gcloud responde "Destination URL must name an
#    existing directory".
cd .\si7016-asilvaz1-taller3-262
gcloud storage cp gs://asilvaz1taller3/inferencia/*.jsonl results\

# 5. Validar el esquema del HANDOFF 4.2 antes de darlo por bueno
python .\src\finetuning\validar_respuestas.py --archivo results\respuestas-finetuning.jsonl
```

Tres cosas que este script cuida y que hay que respetar si alguien lo toca:

- **Las plantillas de prompt son copia literal** de `PLANTILLAS` en
  `src/deploy/03-predict-endpoint.py`, que es con lo que se midió el modelo base
  y con lo que se entrenó. Si cambian aquí, parte de la diferencia medida entre
  base y afinado vendría del prompt y no del fine-tuning.
- **Se decodifican solo los tokens nuevos**, así que no hay eco del prompt que
  limpiar. `prediction` y `prediction_raw` salen iguales, que es lo que la
  sección 4.2 del HANDOFF prevé para un sistema sin eco. No hay que pasar el
  resultado por `clean_predictions.py`.
- **`do_sample=False`** equivale al `temperature=0.0` de la línea base y hace la
  corrida reproducible.

Salida de emergencia si el modelo fusionado del bucket estuviera incompleto: el
job lo detecta al cargarlo y lo dice. Relanzar con `--from_adapters` carga
`gemma-7b-it` en 4 bits y le aplica los adaptadores (200 MB en vez de 17 GB), sin
reconstruir la imagen.

## Fase 3b: el endpoint, para la evidencia

`src/deploy/06-deploy-finetuned.ps1` quedó corregido con las cuatro cosas que
faltaban. Las cuentas de por qué no cabía están en el comentario del propio
script, arriba de `$vllmArgs`:

| Concepto | GB |
| --- | --- |
| Memoria de la L4 | 24.0 |
| Presupuesto con `--gpu-memory-utilization=0.85` | ~20.4 |
| Pesos de gemma-7b-it en bf16 (8.54 B parámetros) | ~17.1 |
| Sobra para caché KV, activaciones y grafos CUDA | ~3.3 |

En esos 3.3 GB no caben la captura de grafos CUDA (1 a 2 GB) más la caché KV de
2048 tokens. Ahora va con `--enforce-eager`, `--max-model-len=1024`,
`--gpu-memory-utilization=0.92` y `--container-deployment-timeout-seconds=7200`.

**La spec del contenedor de un modelo del Model Registry es inmutable**: cambiar
los argumentos de vLLM obliga a subir un modelo nuevo. El endpoint sí se
reutiliza, y `create-endpoint` ahora es idempotente para que no queden dos.

```powershell
powershell -ExecutionPolicy Bypass -File .\si7016-asilvaz1-taller3-262\src\deploy\06-deploy-finetuned.ps1 upload
powershell -ExecutionPolicy Bypass -File .\si7016-asilvaz1-taller3-262\src\deploy\06-deploy-finetuned.ps1 deploy
# desde otra ventana, para ver el arranque de vLLM en vivo:
powershell -ExecutionPolicy Bypass -File .\si7016-asilvaz1-taller3-262\src\deploy\06-deploy-finetuned.ps1 logs
```

Si en los logs aparece `CUDA out of memory` o `No available memory for the cache
blocks`, es la VRAM otra vez y el único ajuste que vale otro intento es bajar a
`--max-model-len=768` y `--max-num-seqs=2`.

**`cleanup` el mismo día, sin excepción.** Una L4 de serving olvidada es el único
riesgo real de costo del taller.
