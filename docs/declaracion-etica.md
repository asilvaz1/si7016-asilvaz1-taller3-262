# Declaración de ética y transparencia

Taller 3, SI7016 NLP Aplicado 2026-2. Documento exigido por la regla de ética y
transparencia del enunciado.

## 1. Integrantes y aporte específico de cada uno

El taller se desarrolló en equipo, con un reparto por carpetas acordado por
escrito antes de empezar (`HANDOFF.md` en la raíz del repositorio), para que
nadie editara los archivos de otro. El historial de git refleja ese reparto: dos
ramas de trabajo, `finetuning-deploy` y `rag-eval`, con commits separados y
autoría verificable con `git log --format='%an'`.

**Alejandro Silva Zuluaga**

- Extracción del corpus: 41 PDF a texto, inventario y detección de duplicados
  (`src/data/extract_pdf.py`).
- Construcción y verificación del dataset de 100 pares pregunta-respuesta,
  incluida la verificación automática de que los 244 números citados en las
  respuestas aparecen en el texto de la norma que cada respuesta declara como
  fuente (`src/data/build_qa_dataset.py`, `verify_qa_grounding.py`).
- Montaje de GCP: APIs, bucket regional, Artifact Registry, diagnóstico de la
  red del campus.
- Despliegue del modelo base en Model Garden y medición de la línea base
  (`src/deploy/`).
- Fine-tuning QLoRA en Vertex, fusión de adaptadores e inferencia por lotes del
  modelo afinado (`src/finetuning/`).
- README, runbooks y esta declaración.

**Isabel Jurado**

- Sistema RAG completo: chunking, embeddings, índice FAISS, recuperación densa,
  BM25 e híbrida, y reranker CrossEncoder (`src/rag/`).
- Prompt de generación anclada (`prompts/rag-anclado.md`, `src/rag/prompts.py`).
- Métricas y evaluación: ROUGE-1/2/L y Recall@K, con sus pruebas
  (`src/eval/metrics.py`, `run_eval.py`, `test_metrics.py`).
- Generación de las 80 respuestas del sistema RAG y las primeras métricas
  comparativas.

**David Botero**

- [PENDIENTE: una o dos líneas concretas y verificables sobre lo que hizo. Por
  ejemplo: revisión del código, pruebas de que los scripts corren en otra
  máquina, montaje de la interfaz de consulta, o apoyo en la depuración de un
  problema puntual.]

Nota de transparencia sobre la autoría: a la fecha de esta entrega, el historial
de git registra commits de Alejandro Silva Zuluaga (con dos identidades de git,
`Alejandro Silva Zuluaga` y `asilvaz1`) y de Isabel Jurado. El aporte de David
Botero se hizo por fuera del control de versiones, y por eso se describe aquí de
forma explícita en vez de dejar que el historial hable por él.

## 2. Código reutilizado y de dónde salió

**Repositorio del curso** `https://github.com/si7016eafit/si7016-262`, que es la
fuente principal de reúso y está declarado como tal:

- `ft-gemma-vertex/train_gemma.py` es la base de nuestro `train_gemma.py`. Los
  cambios propios están listados en el encabezado del archivo: el
  `format_example` (el original arma el chat template sobre el dataset samsum y
  el nuestro sobre nuestros pares pregunta-respuesta), los hiperparámetros
  ajustados al tamaño real del dataset, el token de fin de secuencia al final de
  cada ejemplo y el volcado de `training_config.json`.
- `ft-gemma-vertex/Dockerfile` es la base de nuestra imagen, con la misma imagen
  base de Hugging Face para GCP.
- `c-deploy-merged.ipynb` aportó la lógica de fusión de adaptadores que se
  reescribió como `merge_lora.py` para correr como Custom Job en vez de dentro
  de un notebook.
- `ft-gemma-vertex/c-deploy-gcloud.sh` es el origen de
  `src/deploy/06-deploy-finetuned.ps1`, portado a PowerShell.
- `0-gcp-agent-platform-model-garden/create-model-agent-platform.py` inspiró el
  despliegue del modelo base, aunque no se copió: apunta a Qwen con un
  contenedor fijo, y en su lugar se escribió `01-list-gemma-model-garden.py`,
  que le pregunta al SDK las combinaciones verificadas para Gemma.
- Los notebooks `class05/2-rag/class05b-*.ipynb` son la referencia del pipeline
  de RAG.

**Modelos, imágenes y bibliotecas de terceros**, usados bajo sus licencias:

- `google/gemma-7b-it` de Hugging Face, con la licencia de Gemma aceptada por la
  cuenta que ejecutó el entrenamiento.
- `intfloat/multilingual-e5-base` para los embeddings del RAG.
- Imagen base `huggingface-pytorch-training-cu121` de Google Deep Learning
  Platform, y el contenedor de serving `pytorch-vllm-serve` de Vertex Model
  Garden.
- `transformers`, `peft`, `trl`, `bitsandbytes`, `faiss`, `sentence-transformers`,
  `rank_bm25` y `rouge-score`.

**Documentación consultada**: la referencia de `gcloud ai` de Google Cloud (de
ahí salieron los nombres exactos de las banderas de contenedor, como
`--container-deployment-timeout-seconds`) y los notebooks públicos de Vertex AI
Model Garden, de donde se tomó el criterio de subir el timeout de despliegue a
7200 segundos para modelos grandes.

**Datos**: las normas ISO, UNE y BS son material con derechos de autor, obtenido
a través del acceso institucional de la universidad. Por eso `data/raw/` y
`data/processed/` están fuera del control de versiones, y el repositorio es
privado. El dataset de preguntas y respuestas es de elaboración propia: cada
respuesta se redactó a partir del texto de su norma fuente y se verificó contra
ese texto.

## 3. Uso de inteligencia artificial generativa

Se usó **Claude (Anthropic)**, en la modalidad de asistente con acceso a la
terminal y a los archivos del proyecto, de forma intensiva y a lo largo de todo
el taller. Declararlo con precisión es parte de la regla de transparencia, así
que este es el detalle:

**En qué ayudó de forma sustantiva:**

- Redacción inicial de los scripts de `src/deploy/`, `src/finetuning/` y
  `src/data/`, a partir de los scripts del curso y de las indicaciones sobre qué
  debía hacer cada uno.
- Diagnóstico de los tres problemas de red del campus (bloqueo de gRPC, TCP al
  443 y el TLD `.goog`) y de la bandera `dedicated_endpoint_disabled` del SDK.
- Diagnóstico del fallo de despliegue del modelo afinado: el cálculo de memoria
  de la L4 que aparece en el README y en `06-deploy-finetuned.ps1` salió de esa
  conversación, igual que la distinción entre el error de cuota y el de
  capacidad que llevó a mover el job a `us-west1`.
- La decisión de generar las respuestas del modelo afinado con un Custom Job en
  vez de un endpoint, que es lo que desbloqueó la entrega.
- Redacción del README, de los runbooks y de esta declaración.
- Apoyo en la redacción de parte de las 100 preguntas y respuestas del dataset, a
  partir del texto extraído de cada norma.

**Qué hicimos nosotros y no se delegó:**

- Toda la ejecución en GCP: los comandos, las credenciales, las decisiones de
  gasto y el apagado de recursos.
- La selección del corpus, la resolución de los duplicados del corpus (qué
  edición de cada norma usar como fuente) y la validación de las preguntas
  contra el texto de las normas.
- La verificación de los resultados. Ninguna cifra de este repositorio viene de
  un modelo de lenguaje: las métricas las calcula `src/eval/run_eval.py` sobre
  los archivos de respuestas, y los archivos de respuestas los produjeron los
  modelos desplegados.
- Las decisiones de diseño del experimento: modelo base, enfoque instructivo,
  reparto de las 100 preguntas por categoría, split con semilla fija y el
  criterio de usar el mismo prompt en entrenamiento y evaluación.

**Una precisión que importa para la honestidad del experimento:** el dataset de
preguntas y respuestas se construyó a partir del texto de las normas, no del
conocimiento previo de ningún modelo, y cada número citado se verificó
automáticamente contra el texto fuente antes de entrenar. Si las respuestas de
referencia hubieran sido generadas libremente por un modelo, la comparación de
esta entrega no mediría nada.
