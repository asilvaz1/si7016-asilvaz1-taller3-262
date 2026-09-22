> **Documento histórico.** Es el acuerdo de trabajo en pareja del 19 de septiembre; los estados
> "pendiente" que aparecen aquí ya se cerraron. El estado final está en `README.md`.

# HANDOFF — Taller 3 NLP (SI7016), trabajo en pareja

Documento de arranque para quien entra a colaborar. Entrega: **martes 22 de septiembre**.
Lee esto completo antes de tocar un archivo.

## 1. Qué es el taller en una frase

Comparar cuantitativamente **fine-tuning** contra **RAG + prompt engineering** usando el
mismo LLM abierto (`google/gemma-7b-it`), el mismo corpus (41 normas ISO/UNE/BS de medición
de ruido) y las mismas 20 preguntas de evaluación. Se evalúa primero el modelo base puro,
después el afinado y el RAG.

## 2. Lo que ya está hecho (no rehacer)

- Corpus extraído: 41 PDF a texto limpio, ninguno necesitó OCR.
- De los 41, **33 son fuente y 8 se descartaron** por duplicado o edición superada. Las
  decisiones están una por una en `data/qa/corpus_map.csv`.
- Dataset QA de **100 pares** listo y verificado: esquema, trazabilidad, sin duplicados, y
  los 244 números citados en las respuestas verificados contra el texto de su norma fuente.
  Split estratificado 80/20 con seed 42.
- Scripts de despliegue del modelo base en `src/deploy/` (00 a 04), validados contra el SDK
  instalado. Ver `src/deploy/README.md` para el runbook.

## 3. Reparto por carpeta

El corte es por carpeta para que no haya dos personas editando el mismo archivo.

**Alejo (ruta crítica, requiere credenciales y cuota de GPU):**

- `src/deploy/` — correr 00 a 04, endpoint del modelo base en Model Garden
- `src/finetuning/` — adaptar `format_example`, Docker, smoke test, entrenamiento en Vertex
- `evidencia/` — capturas del despliegue en GCP
- Fusión de adaptadores LoRA y despliegue del modelo afinado

**Colaborador (independiente, sin costo de GPU):**

- `src/rag/` — Fase 1 completa: chunking, embeddings, índice FAISS o Chroma sobre
  `data/processed/`, y `prompts.py`
- `src/eval/` — `metrics.py` (ROUGE y Recall@K) y `run_eval.py`
- `prompts/` — redactar y documentar el prompt exacto de las 3 técnicas
- `docs/declaracion-etica.md` y `docs/costos-gcp.md`
- `notebooks/05-rag-prototipo-faiss.ipynb` y `notebooks/07-evaluacion-comparativa.ipynb`

**Por qué funciona:** el RAG local y el evaluador no dependen de que el fine-tuning exista.
El evaluador se escribe y se prueba contra `data/qa/normas_ruido_eval.jsonl`, que ya está
congelado, usando respuestas de mentira como prueba. Queda listo para el lunes, cuando
lleguen las respuestas reales de los tres sistemas.

## 4. Contrato entre los dos bloques

Esto es lo único que hay que respetar al pie de la letra para que el merge del lunes sea
trivial.

### 4.1 Archivos congelados

**Nadie modifica** `data/qa/*.jsonl`, `data/qa/parts/*.jsonl`, `data/qa/corpus_map.csv` ni
`data/qa/cobertura.csv`. Si aparece un error en una pregunta, se reporta y lo corrige Alejo,
porque tocarlas invalida la verificación de grounding y el split con seed.

### 4.2 Esquema de `results/respuestas-*.jsonl`

Una línea JSON por pregunta. Los tres sistemas escriben **los mismos campos**, para que
`run_eval.py` lea los tres archivos con el mismo parser. Este esquema ya es el que produce
`src/deploy/03-predict-endpoint.py`:

```json
{
  "question": "texto de la pregunta, idéntico al del eval set",
  "reference": "la respuesta de referencia del dataset",
  "prediction": "lo que generó el sistema",
  "technique": "zero-shot | few-shot | chain-of-thought | rag-anclado",
  "system": "base | finetuning | rag",
  "source_file": "nombre del PDF fuente, copiado del dataset",
  "section_label": "etiqueta de sección, copiada del dataset"
}
```

**Solo para RAG**, dos campos adicionales al final del mismo objeto:

```json
{
  "retrieved_sources": ["archivo1.pdf", "archivo2.pdf", "..."],
  "retrieved_chunk_ids": ["archivo1.pdf#0012", "..."]
}
```

`retrieved_sources` va **en orden de ranking**, del más relevante al menos. Recall@K se
calcula comparando `source_file` contra los primeros K de `retrieved_sources`.

Archivos de salida, con estos nombres exactos:

| Sistema | Archivo |
| --- | --- |
| Modelo base | `results/respuestas-base.jsonl` |
| Fine-tuning | `results/respuestas-finetuning.jsonl` |
| RAG | `results/respuestas-rag.jsonl` |

`run_eval.py` consolida los tres en `results/metricas-comparativas.csv`, con una fila por
combinación de `system` y `technique`.

### 4.3 Identificador de chunk

Formato `<source_file>#<índice de 4 dígitos>`, por ejemplo `UNE-EN ISO 3382-1.pdf#0012`.
El índice es la posición del chunk dentro de ese documento, empezando en 0. Sirve para que
Recall@K se pueda calcular a nivel de documento o de chunk sin rehacer el índice.

## 5. Cómo montar el entorno

```bash
git clone <url del repo privado>
cd si7016-asilvaz1-taller3-262
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**Ojo con `data/`:** `data/raw/` (los PDF) y `data/processed/` (el texto extraído) están
fuera de git a propósito, por derechos de autor y por tamaño. Sin ellos el RAG no arranca.
Dos opciones:

1. Bajar ambas carpetas de la carpeta de Drive compartida (recomendado, son ~47 MB).
2. Bajar solo `data/raw/` y regenerar el texto con `python src/data/extract_pdf.py`, que
   produce exactamente lo mismo.

Si eliges la opción 2, verifica contra `data/qa/inventario_corpus.csv`: ahí están las
páginas y los caracteres esperados de cada PDF.

## 6. Reglas de GCP

El proyecto es `si7016-262-nlp`, región `us-central1` (alterna con cuota aprobada:
`us-west1`), bucket `gs://asilvaz1taller3`.

- **Solo Alejo despliega y hace undeploy de endpoints.** Un endpoint olvidado prendido cobra
  por hora y es el único riesgo real de costo del taller. El presupuesto completo estimado
  es de US$7 a 16, y se dispara solo por ahí.
- Antes de cerrar el día, `python src/deploy/04-undeploy-cleanup.py --list` muestra qué está
  cobrando.
- El token de Hugging Face va **siempre** por variable de entorno `$env:HF_TOKEN`, nunca
  escrito dentro de un archivo. El repo es privado, pero la entrega va a Drive y a GitHub.
- La Fase 1 (RAG local con FAISS o Chroma) corre en CPU y **no necesita GCP en absoluto**.
  Se puede avanzar completa sin credenciales.

## 7. Flujo de git

- Rama `main` protegida por acuerdo: nadie hace push directo.
- Una rama por bloque: `rag-eval` para el colaborador, `finetuning-deploy` para Alejo.
- Commits pequeños y push frecuente. Como los bloques no comparten archivos, los merges
  deberían ser automáticos.
- Los notebooks son la excepción: generan conflictos feos. Si dos personas van a tocar el
  mismo notebook, avisar antes.

## 8. Orden sugerido para el colaborador

1. Montar el entorno y confirmar que `data/processed/` tiene los 33 `.txt` de las normas fuente.
2. `src/eval/metrics.py` con ROUGE y Recall@K, probado con un jsonl falso de 3 líneas.
   Es lo más independiente de todo y desbloquea el lunes entero.
3. Chunking y embeddings sobre `data/processed/`, índice FAISS, y Recall@K sobre las 20
   preguntas del eval set. Esta es la primera métrica cuantitativa del taller.
4. Redactar los 4 archivos de `prompts/` con el prompt exacto de cada técnica.
5. `docs/declaracion-etica.md`, que debe decir explícitamente quién hizo qué.

## 9. Pendiente administrativo

Confirmar con el profesor (Edwin Montoya) que acepta trabajo en pareja. El nombre de la
carpeta de entrega lleva un usuario individual. En cualquier caso, la declaración de ética
tiene que detallar el aporte de cada uno y el uso de GenAI.
