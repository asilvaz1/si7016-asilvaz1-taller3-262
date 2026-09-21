# Imagen del Custom Training Job del taller 3 (Gemma-7B-it + QLoRA sobre las
# normas de medicion de ruido).
#
# Misma imagen base que el Dockerfile del curso (ft-gemma-vertex/Dockerfile) y
# por la misma razon: transformers 4.42 ya soporta Gemma v1 de fabrica, asi que
# no hay que tocar torch ni transformers. Menos cambios = menos superficie de
# error.
#
# Unica diferencia: ademas del script, la imagen lleva adentro el dataset de
# preguntas y respuestas, para que el job no dependa de la red ni de permisos
# de lectura sobre el bucket en tiempo de entrenamiento.
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-training-cu121.2-3.transformers.4-42.ubuntu2204.py310

WORKDIR /trainer

COPY train_gemma.py /trainer/train_gemma.py
# El mismo contenedor sirve para la fusion de adaptadores (Fase 3): lleva
# torch, transformers y peft, que es todo lo que merge_lora.py necesita.
COPY merge_lora.py /trainer/merge_lora.py
# Y para la inferencia por lotes (Fase 3, ruta sin endpoint): genera las
# respuestas del modelo afinado con un Custom Job en vez de un endpoint de
# vLLM, que es lo que se trabo por memoria de GPU y por timeout de despliegue.
COPY batch_infer.py /trainer/batch_infer.py

# Los copia 00-build-image.ps1 desde data/qa/ justo antes del build. No estan
# versionados en esta carpeta (ver .gitignore): la copia canonica sigue siendo
# data/qa/, que es carpeta congelada segun el HANDOFF.
COPY normas_ruido_train.jsonl /trainer/normas_ruido_train.jsonl
COPY normas_ruido_eval.jsonl /trainer/normas_ruido_eval.jsonl

# Paquetes de la evaluacion ROUGE que la imagen base no trae. Nada mas.
# hf_xet: sin el, la descarga del modelo cae al transporte HTTP normal. En
# el smoke test un solo shard tardo 26 minutos por esa razon, con la GPU
# encendida y cobrando mientras esperaba.
RUN pip install --no-cache-dir evaluate rouge-score hf_xet

ENTRYPOINT ["python3", "/trainer/train_gemma.py"]
