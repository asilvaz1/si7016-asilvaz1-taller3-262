# Evidencia del despliegue en GCP

Entregable 1.d del taller. La evidencia está en tres formatos:

- **`Evidencias-Taller3.docx`**: el documento con todas las capturas y las
  salidas de consola, en orden de ejecución (Fase 0, fine-tuning, inferencia,
  Fase 3 con vLLM en la VM, RAG Engine y Streamlit).
- **`capturas/`**: las mismas 18 capturas extraídas del documento, con nombre
  descriptivo, para verlas sin abrir Word.
- **Archivos de texto** (sección siguiente), que se pueden citar y buscar.

Proyecto `si7016-262-nlp`.

## Capturas

| Archivo | Qué muestra |
| --- | --- |
| `01-vertex-endpoint-modelo-base.png` | Vertex AI → Endpoints: el endpoint `gemma-7b-it-base-si7016-mg-deploy` del modelo base (Model Garden), `us-central1`. Ya sin modelo desplegado, porque se hizo undeploy al terminar la línea base |
| `02-model-registry-gemma-base.png` | Model Registry: las versiones del modelo base importadas desde Model Garden |
| `03-custom-jobs-entrenamiento-fusion-inferencia.png` | Custom Jobs finalizados: prueba de humo, entrenamiento QLoRA (`gemma-7b-it-normas-ruido-lora`, 17 min 38 s), fusión de adaptadores (`merge-lora-...`) e inferencia (`infer-finetuning-...`) |
| `04-custom-jobs-otra-region.png` | Selector de región de Custom Jobs en otra región, sin trabajos |
| `05-model-registry-modelo-afinado.png` | Model Registry: `gemma-7b-it-normas-...` de origen *Entrenamiento personalizado*, el modelo fusionado |
| `06-bucket-artefactos.png` | Bucket `asilvaz1taller3`: `normas-ruido-lora/` (adaptadores), `normas-ruido-merged/` (modelo fusionado), `inferencia/`, `datasets/`, `smoke-test/` |
| `07-custom-jobs-detalle.png` | La misma lista de jobs con duración y fechas |
| `08-logs-curva-loss.png` | Cloud Logging del entrenamiento: `loss` de 7.33 a 0.16 en 10 épocas y `train_runtime` 847 s. Transcrita en `curva-loss-entrenamiento.md` |
| `09-logs-adaptadores-guardados.png` | Los adaptadores guardados en `gs://asilvaz1taller3/normas-ruido-lora` y `Job completed successfully` |
| `10-logs-inferencia-modelo-afinado.png` | Logs del job de inferencia: 20 respuestas por técnica y copia de los jsonl a `gs://asilvaz1taller3/inferencia` |
| `11-logs-capacidad-insuficiente.png` | `Resources are insufficient in region: us-central1`: el problema de capacidad de GPU, documentado en el README |
| `12-streamlit-afinado-few-shot.png` | App de consulta: el modelo afinado servido con vLLM en la VM, técnica few-shot, con latencia y ROUGE-1 |
| `13-streamlit-prompt-exacto-few-shot.png` | La referencia del dataset y el prompt exacto enviado al modelo |
| `14-streamlit-rag-engine-respuesta.png` | Pestaña RAG Engine: respuesta anclada desde el corpus `normas-ruido-taller3` en `us-west1` |
| `15-streamlit-rag-engine-fragmentos.png` | Los fragmentos recuperados por RAG Engine, con norma y score |
| `16-streamlit-lado-a-lado-pregunta.png` | Pestaña "Lado a lado": la misma pregunta a los dos sistemas desplegados |
| `17-streamlit-lado-a-lado-respuestas.png` | Las dos respuestas con sus métricas: afinado (vLLM) contra RAG Engine |
| `18-streamlit-lado-a-lado-recuperados.png` | Las normas recuperadas por el RAG para esa pregunta |

El `.docx` trae además, como texto, la salida de consola de la VM con vLLM
(`07-vm-vllm.sh serve`, `status` con `/v1/models` mostrando
`gemma-7b-it-normas-ruido` y `max_model_len: 1024`, y `test`), el túnel SSH y la
consulta por `08-predict-vllm.py`, la comparación endpoint contra lotes, la
importación de los 33 documentos a RAG Engine y la salida de
`vertex_rag_engine.py ask` con y sin RAG.

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
