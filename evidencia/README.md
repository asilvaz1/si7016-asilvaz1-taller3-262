# Evidencia del despliegue en GCP

El entregable 1.d del taller pide capturas de pantalla de la ejecución en GCP.
Esta es la lista de lo que hay que capturar, con el nombre de archivo y qué tiene
que verse en cada una para que sirva como evidencia.

Todas corresponden a recursos que siguen existiendo en el proyecto
`si7016-262-nlp`, salvo donde se indique.

| Archivo | Dónde | Qué tiene que verse |
| --- | --- | --- |
| `01-cuota-gpu.png` | IAM y administración → Cuotas, filtro `custom_model_training_nvidia_l4_gpus` | La cuota aprobada en `us-central1` y en `us-west1`, con su límite |
| `02-endpoint-model-garden.png` | Vertex AI → Predicción en línea → Endpoints, o la captura guardada de la sesión de la línea base | El endpoint del modelo base `gemma-7b-it`. **Ya se apagó**, así que sirve una captura anterior o el archivo `src/deploy/.endpoint_base.json` |
| `03-vertex-training-job.png` | Vertex AI → Entrenamiento → Trabajos personalizados, `us-central1` | El job del entrenamiento completo, en estado terminado, con su duración |
| `04-curva-loss.png` | Explorador de registros, filtro `resource.type="ml_job"` | Las líneas de `loss` del entrenamiento. El texto completo ya está transcrito en `curva-loss-entrenamiento.md` |
| `05-job-fusion.png` | Vertex AI → Entrenamiento → Trabajos personalizados | El job `merge-lora-...`, que es el paso de "modelo congelado + adaptadores, mezclar" del enunciado |
| `06-bucket-artefactos.png` | Cloud Storage → `asilvaz1taller3` | Las carpetas `normas-ruido-lora/` (adaptadores), `normas-ruido-merged/` (modelo fusionado) e `inferencia/` (respuestas) |
| `07-model-registry.png` | Vertex AI → Model Registry | El modelo `gemma-7b-it-normas-ruido-merged` registrado |
| `08-job-inferencia-uswest1.png` | Vertex AI → Entrenamiento → Trabajos personalizados, **`us-west1`** | El job `infer-finetuning-...` terminado. Ojo con el selector de región |
| `09-capacidad-insuficiente.png` | Explorador de registros | El error `Resources are insufficient in region: us-central1` junto a la cuota en 0% de uso. Documenta que el problema fue capacidad y no configuración, y justifica el cambio de región |

La 09 es opcional pero vale la pena: muestra un problema real diagnosticado y
resuelto, que es más interesante que una captura de algo que funcionó a la
primera.

## Fase 3: VM con vLLM, app de consulta y RAG Engine

Estas son nuevas. Las tres primeras son las que cierran el requisito del
enunciado de "correr el modelo en la VM usando vLLM".

| Archivo | Dónde | Qué tiene que verse |
| --- | --- | --- |
| `10-vm-instancia.png` | Compute Engine → Instancias de VM | La VM con su tipo de máquina y la GPU L4 asociada. Tomarla **antes** de apagarla |
| `11-nvidia-smi-modelo-cargado.png` | Terminal SSH de la VM, con vLLM sirviendo | `nvidia-smi` mostrando el proceso de Python y la VRAM ocupada. Es la prueba de que el modelo de 17 GB está de verdad en la tarjeta |
| `12-vllm-arranque.png` | `tail ~/vllm.log` en la VM | Las líneas de arranque, en particular la de `KV cache size`. Documenta que los argumentos de memoria funcionaron |
| `13-vllm-models.png` | Navegador o consola, `http://localhost:8000/v1/models` con el túnel abierto | El JSON con `gemma-7b-it-normas-ruido` y `max_model_len: 1024`. Prueba a la vez el serving y el túnel |
| `14-streamlit-lado-a-lado.png` | La app, pestaña "Lado a lado" | Las dos respuestas, sus métricas y los fragmentos recuperados. **Es la captura más informativa del taller entero** |
| `15-streamlit-fragmentos.png` | La app, pestaña "RAG Engine" | Los fragmentos con su norma y su score, que es lo que hace visible por qué el RAG acierta las cifras |
| `16-rag-engine-corpus.png` | Agent Platform → RAG Engine → el corpus → Files | El corpus `normas-ruido-taller3` en `us-west1` con sus 33 archivos indexados |
| `17-bucket-corpus.png` | Cloud Storage → `asilvaz1taller3-west/normas-ruido/` | Los 33 `.txt` que alimentan el corpus |
| `18-vm-apagada.png` | Compute Engine → Instancias de VM | La VM en `TERMINATED`. Es la prueba de que se controló el costo, y cierra el relato que abre la 01 |

Opcionales, pero cada una cuenta algo que no se ve en las demás:

| Archivo | Qué documenta |
| --- | --- |
| `19-ask-con-y-sin-rag.png` | La salida de `vertex_rag_engine.py ask`: la misma pregunta respondida con el corpus y sin él. Es la demostración directa de que el conocimiento viene del corpus y no del modelo |
| `20-comparar-endpoint.png` | La salida de `comparar_finetuning_vllm.py`: el endpoint y el job por lotes dan el mismo ROUGE aunque el texto difiera |
| `21-analisis-forma.png` | La salida de `analizar_respuestas.py`: es la tabla que explica por qué `rag-anclado` puntúa bajo en `rag-vertex`, y la que sostiene la sección de Resultados sobre el corpus bilingüe |

## Evidencia en texto, que es mejor que una captura

Tres archivos del repositorio valen más que sus capturas equivalentes, porque se
pueden citar, buscar y comparar entre versiones:

- **`consultas-streamlit.md`**: consultas reales a los dos sistemas desplegados,
  guardadas desde la app con la pregunta, las dos respuestas, la referencia del
  dataset y las normas recuperadas. La entrada sobre la instrumentación de ISO
  1996-2 es un caso de manual: el modelo afinado responde con una clase y una
  tolerancia inventadas, y el RAG cita literalmente el pasaje de la norma, en
  español, porque el fragmento recuperado viene de la edición UNE.
- **`curva-loss-entrenamiento.md`**: la curva transcrita, porque los logs de GCP
  no se guardan para siempre.
- **`results/*.jsonl` y `results/metricas-comparativas.csv`**: las respuestas y
  las métricas de los cuatro sistemas, que es lo que de verdad se evalúa.
  `metricas-por-pregunta.csv` tiene el detalle fila a fila, por si el profesor
  quiere verificar una pregunta concreta en vez de un promedio.

## Comandos que generan evidencia en texto

Si se prefiere adjuntar salidas de consola en vez de capturas, estos comandos
generan lo equivalente:

```powershell
gcloud ai custom-jobs list --project=si7016-262-nlp --region=us-central1 `
    --format="table(displayName,state,createTime,endTime)"
gcloud ai custom-jobs list --project=si7016-262-nlp --region=us-west1 `
    --format="table(displayName,state,createTime,endTime)"
gcloud storage ls gs://asilvaz1taller3/normas-ruido-lora/ `
    gs://asilvaz1taller3/normas-ruido-merged/ gs://asilvaz1taller3/inferencia/
gcloud ai models list --project=si7016-262-nlp --region=us-central1 `
    --format="table(displayName,name.basename(),createTime)"

# Fase 3
gcloud compute instances list --project=si7016-262-nlp `
    --format="table(name,zone,machineType.basename(),status)"
gcloud storage ls gs://asilvaz1taller3-west/normas-ruido/
python src\rag\vertex_rag_engine.py status
python src\eval\comparar_finetuning_vllm.py
curl.exe http://localhost:8000/v1/models
```

## Lo que no hay, y por qué

No hay captura de un endpoint del modelo afinado **en Vertex**: nunca llegó a
desplegarse. La razón está explicada con las cuentas de memoria de la L4 en el
README, en la sección de Limitaciones, y en los comentarios de
`src/deploy/06-deploy-finetuned.ps1`. El requisito de servir el modelo afinado
se cumplió por la otra ruta, la VM con vLLM, que es la que documentan las
capturas 10 a 13.

No hay medición de `rag-vertex` con el modelo afinado como generador. Se midió
con Gemini, que es el pipeline administrado que pide el enunciado, pero eso
cambia recuperador y generador a la vez. La variante que aísla el recuperador
(`vertex_rag_engine.py run --generator vllm`) está implementada y quedó sin
correr por tiempo; requiere la VM encendida.
