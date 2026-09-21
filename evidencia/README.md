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

## Evidencia en texto

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
```

## Lo que no hay, y por qué

No hay captura del endpoint del **modelo afinado**: nunca llegó a desplegarse.
La razón está explicada con las cuentas de memoria de la L4 en el README, en la
sección de Limitaciones, y en los comentarios de
`src/deploy/06-deploy-finetuned.ps1`.

Tampoco hay captura de una VM con Ollama o vLLM: todo el serving se hizo con
Vertex AI, porque la cuota aprobada es de Vertex Training y Compute Engine tiene
su propia cuota de GPU, que no se solicitó.
