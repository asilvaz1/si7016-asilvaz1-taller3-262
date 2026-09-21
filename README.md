# si7016-asilvaz1-taller3-262

Taller 3 de SI7016 (NLP): comparación cuantitativa de **fine-tuning** frente a
**RAG + prompt engineering** sobre el mismo LLM abierto, con el mismo corpus y
las mismas tareas.

- **Modelo base**: `google/gemma-7b-it` (8.54 B parámetros). El mismo modelo en
  los tres sistemas, para que la comparación sea justa.
- **Dominio**: 41 documentos ISO / UNE / BS y el informe CNOSSOS-EU sobre
  medición de ruido. De los 41 se usan 33 como fuente y 8 se descartan por
  duplicado o edición superada (ver `data/qa/corpus_map.csv`). No se versionan,
  por derechos de autor.
- **Enfoque**: instructivo, con 100 pares pregunta-respuesta que cubren las 33
  normas fuente, divididos 80/20 con semilla 42.
- **Tarea evaluada**: QA sobre el contenido normativo.
- **Infraestructura**: GCP. Vertex AI (Gemini Enterprise Agent Platform) para
  entrenamiento, fusión, inferencia y el endpoint del modelo base.

## Resultados

Las 20 preguntas del conjunto de evaluación, con tres técnicas de prompt
engineering por sistema. Ninguna respuesta quedó vacía o con error.
Fuente: `results/metricas-comparativas.csv`.

| Sistema | Técnica | ROUGE-1 | ROUGE-2 | ROUGE-L | Recall@1 | Recall@5 |
| --- | --- | --- | --- | --- | --- | --- |
| base | chain-of-thought | 0.2578 | 0.0733 | 0.1820 | | |
| base | few-shot | 0.2818 | 0.0768 | 0.1872 | | |
| base | zero-shot | 0.3251 | 0.0792 | 0.2088 | | |
| fine-tuning | chain-of-thought | 0.3622 | 0.1137 | 0.2341 | | |
| fine-tuning | few-shot | 0.3910 | 0.1379 | 0.2535 | | |
| **fine-tuning** | **zero-shot** | **0.4075** | **0.1486** | **0.2709** | | |
| rag | chain-of-thought | 0.3243 | 0.1636 | 0.2527 | 0.90 | 0.95 |
| rag | few-shot | 0.2550 | 0.1169 | 0.1846 | 0.90 | 0.95 |
| rag | rag-anclado | 0.4248 | 0.1997 | 0.2913 | 0.90 | 0.95 |
| **rag** | **zero-shot** | **0.4423** | **0.2170** | **0.3065** | 0.90 | 0.95 |

**El fine-tuning le gana al modelo base en las tres técnicas, sin excepción.**
La peor configuración del afinado (0.3622) supera a la mejor del base (0.3251).
En zero-shot la mejora es de 0.3251 a 0.4075 en ROUGE-1, un 25% relativo.

**Donde más se nota es en ROUGE-2, que casi se duplica** (0.0792 a 0.1486).
ROUGE-1 mide palabras sueltas y ROUGE-2 pares consecutivos, así que ese salto
dice que el afinado no solo acierta el vocabulario del dominio: reproduce la
forma de redactar de las normas.

**El afinado rinde mejor justo con el prompt con el que se entrenó.** Su mejor
técnica es zero-shot y la peor chain-of-thought. Tiene sentido: el turno de
usuario del entrenamiento es, palabra por palabra, el prompt zero-shot de
`prompts/zero-shot.md`, y pedirle que razone paso a paso lo saca de eso. En el
modelo base el orden de las técnicas es el mismo, así que la comparación no
está sesgada por el prompt.

**El RAG gana en fidelidad literal.** Su ROUGE-2 en zero-shot es 0.2170 contra
0.1486 del afinado, un 46% más. Coincide con lo que se ve al leer las
respuestas: el afinado recuerda de memoria, el RAG copia del texto recuperado.

**Una anomalía que no escondemos:** el RAG con few-shot (0.2550) rinde peor que
el modelo base con few-shot (0.2818), y es la única celda donde eso pasa. La
hipótesis es que los dos ejemplos fijos del prompt compiten con el contexto
recuperado, y el modelo termina imitando el estilo de los ejemplos en vez de
anclarse en la norma. Queda como trabajo futuro.

**Recall@5 de 0.95** sobre las 20 preguntas de evaluación (0.87 sobre las 100
del dataset completo, `results/recall-retrieval-all.csv`): el recuperador
encuentra la norma correcta en 19 de 20. El techo del RAG no está en la
recuperación.

### La evidencia cualitativa

Dos casos del conjunto de evaluación, zero-shot, que explican la tabla mejor
que los números.

**"What does ISO 3382-1 specify?"**

- *Base*: "Sure, here is the answer to the question: ISO 3382-1 specifies the
  requirements for the measurement of sound power levels of noise sources using
  sound intensity methods...", y sigue citando una cláusula 4.1 que no existe.
  La norma es de **acústica de salas**, no de potencia sonora.
- *Afinado*: "ISO 3382-1 specifies methods for measuring the reverberation time
  in ordinary rooms. It describes the measurement procedure, the equipment
  needed, the number of measurement positions required, and the method for
  evaluating the data and presenting the test report." Correcto.

**"What background noise criterion does ISO 3744 impose, and how is the
correction K1 applied?"**

- *Referencia*: al menos 6 dB por debajo, preferiblemente más de 15 dB.
- *Afinado*: "The background noise level should not exceed 30 dB... If the
  background noise level is between 20 dB and 30 dB, a correction K1 is
  applied...". Estructura correcta, **números inventados**.

La conclusión del taller sale de juntar los dos casos: el fine-tuning corrige
el dominio y la forma, pero no la memoria de valores concretos. Con 80 ejemplos
el modelo aprende de qué trata cada norma y cómo se redacta una respuesta, no
las cifras. Esa es justamente la debilidad que cubre el RAG, y por eso su
ROUGE-2 es más alto.

Dato de apoyo: las referencias del conjunto de evaluación promedian 568
caracteres. El modelo base respondía con 742 en zero-shot, arrastrando
preámbulos de chatbot como "Sure, here is the answer to the question:". El
afinado bajó a 490 y dejó de escribirlos.

## Estado por fase

| Fase | Contenido | Estado |
| --- | --- | --- |
| 0 | Extracción del corpus y dataset QA de 100 preguntas | hecho |
| 0 | APIs de GCP, bucket regional y Artifact Registry | hecho |
| 0 | Endpoint base `gemma-7b-it` en Model Garden y línea base evaluada | hecho |
| 1 | RAG local (FAISS + BM25 + reranker) y evaluación | hecho |
| 2 | Fine-tuning QLoRA en Vertex (prueba de humo + entrenamiento completo) | hecho |
| 3 | Fusión de adaptadores y respuestas del modelo afinado | hecho |
| 3 | Evaluación comparativa de los tres sistemas | hecho |
| 3 | Endpoint del modelo afinado con vLLM | no logrado, ver Limitaciones |
| 3 | RAG migrado a Vertex AI RAG Engine | no realizado, ver Limitaciones |
| 4 | Limpieza de recursos y declaración de ética | ver `docs/` |

## Qué se ejecutó en GCP

Proyecto `si7016-262-nlp`, bucket regional `gs://asilvaz1taller3`, Artifact
Registry `si7016-taller3`.

**Endpoint del modelo base** (`src/deploy/`, scripts 00 a 05). Model Garden con
`google/gemma@gemma-7b-it`, desplegado sin host dedicado y consultado por REST.
De ahí salen las 60 respuestas de la línea base. El endpoint se apagó al
terminar.

**Entrenamiento** (`src/finetuning/train_gemma.py`). Vertex AI Custom Training
Job, `g2-standard-12` con 1 x NVIDIA L4, imagen propia sobre la de Hugging Face
para GCP. QLoRA 4-bit NF4, LoRA r=16 / alpha=32 / dropout=0.05, lote efectivo 8,
10 épocas, 100 pasos, learning rate 2e-4 con scheduler coseno. Duró 847 s y el
`train_loss` promedio fue 1.1671. La curva completa y su lectura están en
`evidencia/curva-loss-entrenamiento.md`. Los adaptadores quedan en
`gs://asilvaz1taller3/normas-ruido-lora/`.

Los hiperparámetros no son los del script del curso: con 80 ejemplos, el default
(3 épocas, lote efectivo 16) da 15 pasos de optimizador en total, que no alcanza
para que los adaptadores aprendan nada.

**Fusión de adaptadores** (`src/finetuning/merge_lora.py`). Custom Job en CPU
(`n1-highmem-8`), porque la fusión es aritmética de pesos y así no compite por
la única L4 del proyecto. El modelo fusionado queda en
`gs://asilvaz1taller3/normas-ruido-merged/`. Esto es lo que el enunciado pide
como "modelo congelado + adaptadores, están separados, mezclar".

**Inferencia del modelo afinado** (`src/finetuning/batch_infer.py`). Custom Job
con 1 x L4 en `us-west1`, que carga el modelo fusionado desde GCS y genera las
60 respuestas con las tres técnicas. Ver Limitaciones para por qué no fue por un
endpoint.

**Generación del RAG**. La recuperación corre local sobre FAISS; la generación
se ancla al mismo `gemma-7b-it` servido en Vertex, para que la comparación sea
sobre el mismo LLM en los tres sistemas.

## Estructura

```text
data/raw/         los 41 PDF de normas (ignorado por git)
data/processed/   texto extraído de cada norma (ignorado por git)
data/qa/          dataset QA, inventario, mapa del corpus y cobertura
src/data/         extracción de PDF, construcción y verificación del dataset
src/finetuning/   entrenamiento, fusión, inferencia por lotes y validación
src/deploy/       despliegue y consulta de los endpoints de Vertex
src/rag/          chunking, índice FAISS, recuperación y generación anclada
src/eval/         limpieza de respuestas, ROUGE y Recall@K
prompts/          el prompt exacto de cada técnica de prompt engineering
results/          respuestas de los tres sistemas y métricas comparativas
evidencia/        capturas del despliegue en GCP y curva de entrenamiento
docs/             declaración de ética y costos
```

Cada carpeta de `src/` con pasos en GCP trae su propio runbook:
`src/deploy/README.md` y `src/finetuning/README.md`.

## Reproducir

```bash
pip install -r requirements.txt
```

**Dataset**

```bash
python src/data/extract_pdf.py --src "../talleres/taller3/NORMAS ESTANDARES" \
    --out data/processed --inventory data/qa/inventario_corpus.csv
python src/data/build_qa_dataset.py --seed 42
python src/data/verify_qa_grounding.py
```

Ver `data/qa/README.md` para el esquema, el reparto de las 100 preguntas por
categoría y el criterio con que se resolvieron los duplicados del corpus.

**Fine-tuning y respuestas del modelo afinado** (PowerShell, desde `13_nlp`,
con `$env:HF_TOKEN` definido). El runbook completo está en
`src/finetuning/README.md`.

```powershell
powershell -File .\si7016-asilvaz1-taller3-262\src\finetuning\00-build-image.ps1
python .\si7016-asilvaz1-taller3-262\src\finetuning\submit_vertex_job.py --smoke_test
python .\si7016-asilvaz1-taller3-262\src\finetuning\submit_vertex_job.py
python .\si7016-asilvaz1-taller3-262\src\finetuning\submit_merge_job.py
python .\si7016-asilvaz1-taller3-262\src\finetuning\submit_infer_job.py
```

**RAG local**

```bash
python src/rag/index.py                       # chunking + embeddings + FAISS -> data/index/
python src/rag/eval_retrieval.py --dataset data/qa/normas_ruido_all.jsonl --rerank
python src/rag/run_rag.py --generator none    # recuperación y prompts, sin GPU
python src/rag/run_rag.py --generator endpoint
```

Chunks de 1000 caracteres con solape de 150, embeddings
`intfloat/multilingual-e5-base`, recuperación densa (default), BM25 o híbrida
(RRF), y reranker CrossEncoder opcional. Con las 33 normas la densa gana:
Recall@5 de 0.87 sobre las 100 preguntas, frente a 0.83 híbrida y 0.51 BM25.

**Evaluación**

```bash
python src/eval/test_metrics.py     # pruebas de las métricas
python src/eval/run_eval.py         # results/respuestas-*.jsonl -> metricas-comparativas.csv
python src/finetuning/validar_respuestas.py --archivo results/respuestas-finetuning.jsonl
```

ROUGE-1/2/L (F-measure) para la generación y Recall@1/3/5 a nivel de documento
para la recuperación. Las respuestas vacías o con `__ERROR__` no se puntúan y se
cuentan aparte (`n` frente a `n_validas`). En chain-of-thought se puntúa solo lo
que sigue a la última línea `Answer:`, y en rag-anclado se quitan las citas
`[n]`.

## Prompt engineering

Las tres técnicas que exige el taller están en `prompts/`, con el texto exacto
que se envía: `zero-shot.md`, `few-shot.md` y `chain-of-thought.md`, más
`rag-anclado.md` para la generación anclada del RAG. Los tres primeros no se
escriben a mano: `src/deploy/export_prompts.py` los genera leyéndolos del script
de predicción, para que lo documentado no se desincronice de lo que de verdad se
manda.

Las mismas tres técnicas se aplican a los tres sistemas, que es lo que permite
comparar fila contra fila en la tabla de resultados.

## Limitaciones, y qué se hizo en su lugar

**El endpoint del modelo afinado no levantó.** El modelo fusionado en bf16 pesa
unos 17.1 GB y la L4 tiene 24 GB. Con `--gpu-memory-utilization=0.85` el
presupuesto es de ~20.4 GB, así que quedan ~3.3 GB para la caché KV, las
activaciones y los grafos CUDA de vLLM. No alcanza: la captura de grafos ya
gasta 1 a 2 GB, y una caché KV para 2048 tokens pesa ~0.9 GB en Gemma 7B (~0.44
MB por token). El contenedor moría con *Model server exited unexpectedly*. A eso
se sumó que `gcloud ai models upload` usaba el timeout de despliegue por defecto
(1800 s), insuficiente para traer 17 GB desde GCS y arrancar vLLM.

`src/deploy/06-deploy-finetuned.ps1` quedó corregido con los cuatro ajustes que
lo harían viable (`--enforce-eager`, `--max-model-len=1024`,
`--gpu-memory-utilization=0.92` y `--container-deployment-timeout-seconds=7200`),
documentados en el propio script.

Como el entregable evaluable son las respuestas y no un endpoint vivo, se
generaron con un Custom Job, que usa la cuota de entrenamiento, no pasa por
health checks ni por el reloj de despliegue, y se apaga solo.

**Hubo un tercer obstáculo, de capacidad y no de configuración.** El job de
inferencia se quedó en `PENDING` con `Resources are insufficient in region:
us-central1` en tres intentos seguidos. No era cuota: el límite es 1 L4 y el uso
estaba en 0. Simplemente no había tarjetas libres. Se relanzó en `us-west1`,
donde también hay cuota aprobada, leyendo el modelo fusionado del bucket de
`us-central1`. Ahí corrió sin problema.

**El RAG no se migró a Vertex AI RAG Engine.** Funciona con FAISS local y la
generación anclada al endpoint de Vertex. La migración al corpus administrado
quedó fuera por tiempo.

**No se montó la VM con Ollama/vLLM.** Todo el serving fue por Vertex AI. La
razón es de cuota: la aprobada es `custom_model_training_nvidia_l4_gpus`, que es
de Vertex Training, mientras que Compute Engine tiene su propia cuota de GPU,
distinta y nunca solicitada.

**No hay interfaz de consulta tipo Streamlit.** Las consultas se hacen por
script (`src/deploy/03-predict-endpoint.py` y `src/rag/run_rag.py`).

## Costos

Ver `docs/costos-gcp.md` para el detalle por recurso. El gasto real se concentró
en tres jobs de Vertex y unas horas de endpoint del modelo base, todos apagados
al terminar.

## Ética y transparencia

Ver `docs/declaracion-etica.md`: código reutilizado del repositorio del curso,
aporte específico de cada integrante y uso de IA generativa.
