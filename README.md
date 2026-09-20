# si7016-asilvaz1-taller3-262

Taller 3 de SI7016 (NLP): comparación cuantitativa de **fine-tuning** frente a
**RAG + prompt engineering** sobre el mismo LLM abierto, con el mismo corpus y
las mismas tareas.

- **Modelo base**: `google/gemma-7b-it` (mismo modelo en los dos sistemas, para
  que la comparación sea justa).
- **Dominio**: 41 normas ISO / UNE / BS y el informe CNOSSOS-EU sobre medición
  de ruido, en `talleres/taller3/NORMAS ESTANDARES` (no versionadas, por
  derechos de autor).
- **Enfoque**: instructivo, con 100 pares pregunta-respuesta que cubren las 33
  normas fuente del corpus.
- **Infraestructura**: GCP — Vertex AI Custom Training Job para el
  entrenamiento, Model Garden / Vertex Endpoints para el despliegue,
  Vertex AI RAG Engine para el RAG administrado.

## Estado

| Fase | Contenido | Estado |
| --- | --- | --- |
| 0 | Extracción del corpus y dataset QA de 100 preguntas | hecho |
| 0 | Habilitar APIs de GCP, bucket, endpoint base en Model Garden | pendiente |
| 1 | Prototipo RAG local (FAISS + BM25, CPU) | hecho: indice de 33 normas, Recall@5 = 0.87 (denso) |
| 2 | Fine-tuning QLoRA en Vertex (prueba de humo + entrenamiento completo) | pendiente |
| 3 | Despliegue, RAG en Vertex AI RAG Engine y evaluación comparativa | pendiente |
| 4 | Entrega, limpieza de recursos y declaración de ética | pendiente |

## Estructura

```text
data/raw/         los 41 PDF de normas (ignorado por git)
data/processed/   texto extraído de cada norma (ignorado por git)
data/qa/          dataset QA, inventario, mapa del corpus y cobertura
src/data/         extracción de PDF, construcción y verificación del dataset
src/finetuning/   train_gemma.py, Dockerfile, submit_vertex_job.py
src/deploy/       despliegue del modelo base y del modelo afinado
src/rag/          índice FAISS local y corpus en Vertex AI RAG Engine
src/eval/         ROUGE y Recall@K
prompts/          el prompt exacto de cada técnica de prompt engineering
results/          respuestas y métricas comparativas
evidencia/        capturas del despliegue en GCP
docs/             declaración de ética y costos
```

## Dataset

Ver `data/qa/README.md` para el esquema, el reparto de las 100 preguntas por
categoría, el criterio con el que se resolvieron los duplicados del corpus
(41 PDF → 33 normas fuente) y los comandos para reproducirlo.

## Reproducir el dataset

```bash
pip install -r requirements.txt
python src/data/extract_pdf.py --src "../talleres/taller3/NORMAS ESTANDARES" \
    --out data/processed --inventory data/qa/inventario_corpus.csv
python src/data/build_qa_dataset.py --seed 42
python src/data/verify_qa_grounding.py
```

## RAG local (Fase 1)

```bash
python src/rag/index.py                                   # chunking + embeddings + FAISS -> data/index/
python src/rag/eval_retrieval.py --dataset data/qa/normas_ruido_all.jsonl --rerank
python src/rag/run_rag.py --generator none                # recuperacion + prompts, sin GPU
python src/rag/run_rag.py --generator endpoint            # generacion real -> results/respuestas-rag.jsonl
```

Chunks de 1000 caracteres con solape de 150, embeddings `intfloat/multilingual-e5-base`,
recuperacion densa (default), BM25 o hibrida (RRF) y reranker CrossEncoder opcional. Con las 33 normas, la
densa gana (Recall@5 = 0.87 sobre las 100 preguntas frente a 0.83 hibrida y 0.51 BM25); ver `results/recall-retrieval-all.csv`. Los prompts
del RAG estan en `src/rag/prompts.py` y documentados en `prompts/rag-anclado.md`.

## Evaluacion

```bash
python src/eval/test_metrics.py     # pruebas con respuestas de mentira
python src/eval/run_eval.py         # results/respuestas-*.jsonl -> results/metricas-comparativas.csv
```

ROUGE-1/2/L (F-measure) sobre la respuesta; Recall@1/3/5 a nivel de documento para el RAG.
Las respuestas vacias o con `__ERROR__` no se puntuan y se cuentan aparte (`n` frente a
`n_validas`). En chain-of-thought se puntua solo lo que sigue a la ultima linea `Answer:`,
y en rag-anclado se quitan las citas `[n]`.
