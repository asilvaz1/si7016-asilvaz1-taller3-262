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
| 0 | APIs de GCP, bucket regional y Artifact Registry | hecho |
| 0 | Endpoint base `gemma-7b-it` en Model Garden y línea base evaluada | hecho |
| 1 | Prototipo RAG local (FAISS/Chroma, CPU) | pendiente |
| 2 | Fine-tuning QLoRA en Vertex (prueba de humo + entrenamiento completo) | pendiente |
| 3 | Despliegue del afinado, RAG en Vertex AI RAG Engine y evaluación comparativa | pendiente |
| 4 | Entrega, limpieza de recursos y declaración de ética | pendiente |

### Línea base ya medida

60 respuestas del modelo base puro sobre las 20 preguntas del split de
evaluación, con las tres técnicas de prompt engineering, sin ningún error de
llamada:

| Archivo | Técnica |
| --- | --- |
| `results/respuestas-base-zero-shot.jsonl` | Zero-shot |
| `results/respuestas-base-few-shot.jsonl` | Few-shot |
| `results/respuestas-base-cot.jsonl` | Chain-of-thought |

Cada fila trae `prediction` (solo lo generado) y `prediction_raw` (la respuesta
tal cual la devolvió el contenedor, con el eco del prompt). Los `.raw` al lado
son el respaldo previo a la limpieza.

El modelo base inventa cláusulas y confunde normas con seguridad: por ejemplo
afirma que ISO 3382-1 trata de potencia acústica de vehículos, cuando es de
acústica de salas. Eso no es un fallo del montaje, es la evidencia que el
taller busca, y es contra lo que se compararán el fine-tuning y el RAG.

## Estructura

```text
data/raw/         los 41 PDF de normas (ignorado por git)
data/processed/   texto extraído de cada norma (ignorado por git)
data/qa/          dataset QA, inventario, mapa del corpus y cobertura
src/data/         extracción de PDF, construcción y verificación del dataset
src/finetuning/   train_gemma.py, Dockerfile, submit_vertex_job.py
src/deploy/       despliegue del modelo base y del modelo afinado
src/rag/          índice FAISS local y corpus en Vertex AI RAG Engine
src/eval/         limpieza de respuestas, ROUGE y Recall@K
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
