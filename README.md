# si7016-asilvaz1-taller3-262

Taller 3 de SI7016 (NLP): comparación cuantitativa de **fine-tuning** frente a
**RAG + prompt engineering** sobre el mismo LLM abierto, con el mismo corpus y
las mismas tareas.

- **Modelo base**: `google/gemma-7b-it` (8.54 B parámetros). El mismo modelo en
  los tres primeros sistemas, para que la comparación sea justa. El cuarto,
  `rag-vertex`, genera con `gemini-2.5-flash`, y eso condiciona cómo se lee su
  fila (ver Resultados).
- **Dominio**: 41 documentos ISO / UNE / BS y el informe CNOSSOS-EU sobre
  medición de ruido. De los 41 se usan 33 como fuente y 8 se descartan por
  duplicado o edición superada (ver `data/qa/corpus_map.csv`). No se versionan,
  por derechos de autor.
- **Enfoque**: instructivo, con 100 pares pregunta-respuesta que cubren las 33
  normas fuente, divididos 80/20 con semilla 42.
- **Tarea evaluada**: QA sobre el contenido normativo.
- **Infraestructura**: GCP. Vertex AI para entrenamiento, fusión, inferencia por
  lotes y el endpoint del modelo base; una VM de Compute Engine con vLLM para
  servir el modelo afinado; Vertex AI RAG Engine para el corpus administrado.

## Resultados

Las 20 preguntas del conjunto de evaluación, con tres técnicas de prompt
engineering por sistema, más `rag-anclado` en los dos sistemas RAG. Ninguna
respuesta quedó vacía o con error. Fuente:
`results/metricas-comparativas.csv`.

Son **cuatro sistemas**, y conviene tener claro en qué se diferencian antes de
leer la tabla:

| Sistema | Recuperador | Generador |
| --- | --- | --- |
| `base` | ninguno | `gemma-7b-it` sin afinar |
| `finetuning` | ninguno | `gemma-7b-it` + QLoRA sobre las normas |
| `rag` | FAISS local sobre los 33 documentos | `gemma-7b-it` sin afinar |
| `rag-vertex` | Vertex AI RAG Engine | `gemini-2.5-flash` |

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
| rag-vertex | chain-of-thought | 0.4214 | 0.2812 | 0.3515 | 0.95 | 1.00 |
| rag-vertex | rag-anclado | 0.3362 | 0.2218 | 0.2803 | 0.95 | 1.00 |
| rag-vertex | zero-shot | 0.5156 | 0.3397 | 0.4210 | 0.95 | 1.00 |
| **rag-vertex** | **few-shot** | **0.5362** | **0.3855** | **0.4512** | **0.95** | **1.00** |

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

### Leer la fila de `rag-vertex` sin pasarse de frenada

`rag-vertex` tiene los mejores números de la tabla, pero **cambia dos cosas a la
vez** respecto de `rag`: el recuperador (RAG Engine en vez de FAISS) y el
generador (`gemini-2.5-flash` en vez de `gemma-7b-it`). Así que sus ROUGE no
dicen que el recuperador administrado sea mejor: dicen que el sistema completo
lo es, y lo más probable es que el mérito sea sobre todo del generador, que es
un modelo bastante más capaz.

**Hay una comparación que sí es limpia, y es la del recuperador**, porque
Recall@K no depende de quién redacte:

| Recuperador | Recall@1 | Recall@3 | Recall@5 |
| --- | --- | --- | --- |
| FAISS local (chunks de 1000, `multilingual-e5-base`) | 0.90 | 0.95 | 0.95 |
| Vertex AI RAG Engine (chunks de 512, `text-embedding-005`) | 0.95 | 1.00 | 1.00 |

RAG Engine encuentra el documento correcto para **las 20 preguntas** a partir de
k=3, y la única que no queda en primer lugar la recupera en segundo. Es una
mejora real y modesta, sobre un recuperador local que ya iba bien.

Para aislar el efecto del generador haría falta correr
`vertex_rag_engine.py run --generator vllm`, que recupera de RAG Engine y genera
con el modelo afinado de la VM. Queda como el experimento pendiente más obvio, y
está implementado.

### Por qué `rag-anclado` es la peor técnica de `rag-vertex` y la segunda mejor de `rag`

Es la inversión más llamativa de la tabla, y no es ruido. Sale de mirar **cómo**
responde cada sistema, no cuánto puntúa (`src/eval/analizar_respuestas.py`):

| Sistema | Técnica | Respuestas en español | Largo medio | Con citas literales |
| --- | --- | --- | --- | --- |
| rag | rag-anclado | 0/20 | 500 | 6/20 |
| rag-vertex | rag-anclado | **10/20** | 630 | 20/20 |
| rag-vertex | few-shot | 0/20 | 367 | 1/20 |
| rag-vertex | chain-of-thought | 1/20 | **1080** | 20/20 |

Las referencias del dataset están **todas en inglés** y promedian 568
caracteres.

El prompt `rag-anclado` pide citar el pasaje exacto. Gemini obedece esa
instrucción al pie de la letra, y **la mitad de los fragmentos que recupera
vienen de las ediciones UNE, que están en español**. El resultado son respuestas
correctas, literales y bien ancladas, que puntúan bajo porque están en otro
idioma que la referencia. `gemma-7b-it` no producía ese efecto porque obedecía
la instrucción de citar mucho menos: solo 6 de 20 de sus respuestas traen
comillas, frente a 20 de 20 en Gemini.

La misma lógica explica chain-of-thought: 1080 caracteres contra 568 de la
referencia. ROUGE es F-measure, así que irse al doble de largo cuesta precisión
aunque la respuesta correcta esté dentro. Y explica por qué gana few-shot: los
dos ejemplos resueltos fijan el registro y la longitud, 367 caracteres, sin
comillas, que es lo más parecido a cómo está redactada una referencia.

**La lectura honesta:** parte de la diferencia entre técnicas en `rag-vertex`
mide el idioma y la verbosidad de la respuesta, no su corrección. Un corpus
bilingüe evaluado con referencias en un solo idioma penaliza justamente al
sistema que mejor se ancla en el texto original. Es una limitación del diseño de
la evaluación, no del sistema, y lo correcto es decirlo en vez de presentar 0.53
como si fuera una medida limpia de calidad.

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

### El endpoint sirve lo mismo que se midió

Las 60 respuestas de la tabla las generó un Custom Job por lotes. Para
comprobar que el endpoint de vLLM en la VM no sirve otra cosa, se repitieron las
mismas 60 por el endpoint (`results/respuestas-finetuning-vllm.jsonl`) y se
compararon con `src/eval/comparar_finetuning_vllm.py`:

| Técnica | ROUGE-1 lotes | ROUGE-1 endpoint | Desvío máximo |
| --- | --- | --- | --- |
| zero-shot | 0.4075 | 0.4059 | 0.0023 |
| few-shot | 0.3910 | 0.4020 | 0.0110 |
| chain-of-thought | 0.3622 | 0.3773 | 0.0151 |

El resultado interesante no es que coincidan, sino **cómo** coinciden: solo 7 de
las 60 respuestas son idénticas carácter a carácter, y aun así las métricas se
mueven en milésimas. En bf16 y con decodificación voraz, un empate de logits en
el primer token manda la respuesta por otro camino, y vLLM y `transformers`
empatan distinto porque usan kernels distintos y agrupan las secuencias de otra
forma. Donde el modelo no recuerda la cifra exacta y elige entre continuaciones
igual de plausibles, eso pasa casi siempre.

La lectura para el taller: **la conclusión no depende del camino de
inferencia**. El texto concreto de una respuesta sí, y por eso la evaluación se
hace sobre 20 preguntas y tres técnicas, no sobre un ejemplo.

## Estado por fase

| Fase | Contenido | Estado |
| --- | --- | --- |
| 0 | Extracción del corpus y dataset QA de 100 preguntas | hecho |
| 0 | APIs de GCP, bucket regional y Artifact Registry | hecho |
| 0 | Endpoint base `gemma-7b-it` en Model Garden y línea base evaluada | hecho |
| 1 | RAG local (FAISS + BM25 + reranker) y evaluación | hecho |
| 2 | Fine-tuning QLoRA en Vertex (prueba de humo + entrenamiento completo) | hecho |
| 3 | Fusión de adaptadores y respuestas del modelo afinado | hecho |
| 3 | Evaluación comparativa de base, fine-tuning y RAG | hecho |
| 3 | Endpoint del modelo afinado con vLLM en una VM | hecho |
| 3 | App de consulta tipo Streamlit | hecho |
| 3 | RAG migrado a Vertex AI RAG Engine | hecho |
| 3 | Evaluación comparativa de los cuatro sistemas | hecho |
| 4 | Limpieza de recursos y declaración de ética | ver `docs/` |

La Fase 3 se ejecutó con el plazo ampliado al miércoles 23. El runbook completo,
incluido el criterio de qué se hizo a mano y qué quedó en un script, está en
`docs/fase3-runbook.md`.

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

**Endpoint del modelo afinado con vLLM** (`src/deploy/07-vm-vllm.sh`). VM de
Compute Engine con 1 x NVIDIA L4, vLLM instalado en un entorno virtual y el
modelo fusionado sincronizado desde GCS. Sirve una API compatible con OpenAI en
el puerto 8000, a la que se llega por un túnel SSH. Los argumentos salen de la
cuenta de memoria de la L4: 17.1 GB de pesos en bf16 sobre 22.1 GB de
presupuesto con `--gpu-memory-utilization=0.92`, más `--enforce-eager` para no
gastar 1 a 2 GB en grafos CUDA y `--max-model-len=1024`, que sobra para el
prompt más largo (few-shot, unos 250 tokens) más 256 de generación.

Esta es la ruta que el enunciado pide como "correr el modelo en la VM usando
vLLM". Se eligió sobre el endpoint de Vertex por una razón práctica: en la VM el
log de arranque se ve en vivo y un reintento con otros argumentos cuesta
segundos, mientras que en Vertex cada intento son 30 minutos y un veredicto sin
causa (ver Limitaciones).

**Interfaz de consulta** (`app/streamlit_app.py`). Corre local contra el
endpoint de la VM y contra el corpus de RAG Engine, con las dos respuestas lado
a lado, los fragmentos recuperados y el ROUGE-1 de cada respuesta contra la
referencia del dataset. Las consultas guardadas desde ahí están en
`evidencia/consultas-streamlit.md`.

**Corpus en Vertex AI RAG Engine** (`src/rag/vertex_rag_engine.py`). Los 33
documentos fuente en `us-west1`, troceados en chunks de 512 con solape de 100 y
vectorizados con `text-embedding-005`. La región no es la del resto del
proyecto a propósito: el modo *Spanner* de RAG Engine está restringido a
proyectos en lista blanca en `us-central1`, `us-east1` y `us-east4`, y
`us-west1` queda fuera de esa restricción.

**Generación del RAG**. La recuperación corre local sobre FAISS; la generación
se ancla al mismo `gemma-7b-it` servido en Vertex, para que la comparación sea
sobre el mismo LLM que el base y el afinado.

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
app/              interfaz de consulta en Streamlit (Fase 3)
prompts/          el prompt exacto de cada técnica de prompt engineering
results/          respuestas de los cuatro sistemas y métricas comparativas
evidencia/        capturas del despliegue, curva de entrenamiento y consultas
docs/             declaración de ética, costos y runbook de la Fase 3
requirements-fase3.txt  entorno aparte para el SDK de RAG Engine
```

Cada carpeta de `src/` con pasos en GCP trae su propio runbook:
`src/deploy/README.md` y `src/finetuning/README.md`. La Fase 3 (VM con vLLM,
app de consulta y RAG Engine) tiene el suyo en `docs/fase3-runbook.md`.

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

**Fase 3: servir el modelo afinado con vLLM en una VM**

El runbook con los prerrequisitos y el criterio de qué va a mano está en
`docs/fase3-runbook.md`.

```bash
# dentro de la VM
./07-vm-vllm.sh check && ./07-vm-vllm.sh install && ./07-vm-vllm.sh sync
./07-vm-vllm.sh serve && ./07-vm-vllm.sh status && ./07-vm-vllm.sh test
```

```powershell
# en Windows, con el tunel abierto
gcloud compute ssh VM --zone=ZONA --ssh-flag="-N" --ssh-flag="-L" --ssh-flag="8000:localhost:8000"
python src\deploy\08-predict-vllm.py --dataset data\qa\normas_ruido_eval.jsonl `
    --out results\respuestas-finetuning-vllm.jsonl --technique all
python src\eval\comparar_finetuning_vllm.py
streamlit run app\streamlit_app.py
```

**Fase 3: RAG en Vertex AI RAG Engine**

Entorno aparte, porque `google-cloud-agentplatform` sube el SDK de aiplatform a
la línea 2.x y el resto del repositorio está fijado en 1.75.0.

```powershell
python -m venv .venv-rag; .venv-rag\Scripts\activate
pip install -r requirements-fase3.txt
python src\rag\vertex_rag_engine.py upload
python src\rag\vertex_rag_engine.py create
python src\rag\vertex_rag_engine.py import
python src\rag\vertex_rag_engine.py status
python src\rag\vertex_rag_engine.py ask --q "What does ISO 3382-1 specify?"
python src\rag\vertex_rag_engine.py run --technique all
```

Entre `create` e `import` hay un paso manual: darle al agente de servicio de RAG
Engine permiso de lectura sobre el bucket. Sin él la importación no falla, solo
deja el corpus vacío. El comando exacto lo imprime `upload`.

**Apagar lo que cobra**

```powershell
powershell -File src\deploy\09-apagar-todo.ps1 estado
powershell -File src\deploy\09-apagar-todo.ps1 apagar
```

**Evaluación**

```bash
python src/eval/test_metrics.py     # pruebas de las métricas
python src/eval/run_eval.py         # results/respuestas-*.jsonl -> metricas-comparativas.csv
python src/eval/analizar_respuestas.py   # idioma, longitud y citas por sistema
python src/finetuning/validar_respuestas.py --archivo results/respuestas-finetuning.jsonl
```

ROUGE-1/2/L (F-measure) para la generación y Recall@1/3/5 a nivel de documento
para la recuperación. `analizar_respuestas.py` no calcula calidad: describe la
forma de las respuestas (idioma, longitud, citas literales), que es lo que
explica varias de las diferencias entre técnicas. Las respuestas vacías o con `__ERROR__` no se puntúan y se
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

Las mismas tres técnicas se aplican a los cuatro sistemas, que es lo que
permite comparar fila contra fila en la tabla de resultados. Los dos sistemas
RAG suman además `rag-anclado`, que no tiene equivalente sin contexto
recuperado.

## Limitaciones, y qué se hizo en su lugar

Esta sección describe el estado **medido** del taller, que es el que sostiene la
tabla de resultados. Las dos primeras limitaciones son históricas: describen por
qué el endpoint de Vertex no levantó, que es lo que llevó a servir el modelo
afinado desde una VM en la Fase 3. Se dejan escritas porque el diagnóstico es
parte del trabajo, no un borrador superado.


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

**`rag-vertex` cambia el recuperador y el generador a la vez.** Se midió con
`--generator gemini`, que es el pipeline administrado completo y lo que pide el
enunciado, pero eso significa que sus ROUGE no aíslan el efecto de ninguno de
los dos. La comparación limpia entre recuperadores es la de Recall@K, que está
arriba. El experimento que sí aislaría el generador,
`vertex_rag_engine.py run --generator vllm`, está implementado y no se corrió
por tiempo: requiere la VM encendida y el túnel abierto.

**La evaluación penaliza el anclaje en un corpus bilingüe.** Las referencias del
dataset están todas en inglés, y 10 de las 20 respuestas de `rag-vertex` con
`rag-anclado` salen en español porque el fragmento recuperado viene de la
edición UNE. Son respuestas correctas que puntúan bajo. Arreglarlo de verdad
pediría o referencias en los dos idiomas, o una métrica que no dependa del
idioma, o restringir el corpus a una sola edición por norma; las tres son
decisiones de diseño del dataset, no parches de código. Está medido en
`src/eval/analizar_respuestas.py` y explicado en Resultados.

**El Recall@K a nivel de chunk no es comparable entre los dos RAG.** RAG Engine
no expone la posición del chunk dentro del documento, solo un identificador
opaco. El Recall a nivel de documento sí es comparable, y es el que reporta la
tabla.

### Tres tropiezos del SDK que vale la pena dejar escritos

Ninguno era lo que decía el mensaje de error, y los tres están ahora
diagnosticados en el código:

- **`Bucket ... does not belong to project ...`** al importar. El bucket estaba
  bien. `import_files` comprueba la propiedad del bucket traduciendo el project
  ID a número con la Cloud Resource Manager API, y toda esa lógica vive dentro
  de un `except Exception: return False`, así que un paquete ausente o una API
  sin habilitar salen como un problema de bucket. `vertex_rag_engine.py` hace
  ahora esa comprobación por su cuenta antes de llamar, y distingue los casos.
- **Importación con comodín que devuelve cero sin fallar.** `gs://bucket/pref/*`
  importa 0 archivos, 0 fallidos, sin error. El propio notebook de la clase
  tiene una nota diciendo que al final los documentos se cargaron a mano por la
  consola. El script enumera los objetos y pasa las URIs explícitas, en lotes de
  25, que es el tope de la API.
- **`status` que mostraba 25 de 33.** `list_files` pagina de 25 en 25, y leer
  solo la primera página coincidía justo con el tamaño del primer lote de
  importación, así que parecía que el segundo lote se había perdido. Dos errores
  distintos disfrazados del mismo síntoma.

## Costos

Ver `docs/costos-gcp.md` para el detalle por recurso. El gasto real se concentró
en tres jobs de Vertex y unas horas de endpoint del modelo base, todos apagados
al terminar.

## Ética y transparencia

Ver `docs/declaracion-etica.md`: código reutilizado del repositorio del curso,
aporte específico de cada integrante y uso de IA generativa.
