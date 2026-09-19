# Despliegue en GCP

Runbook de la parte de infraestructura.

## Paso 0: abrir la terminal en el sitio correcto

Abre **PowerShell** y ponte en la carpeta `13_nlp`. Todos los comandos de este
documento asumen que estás ahí.

```powershell
cd D:\EAFIT\maestria_eafit\13_nlp
```

Activa el entorno virtual. Si PowerShell se queja con *"la ejecución de scripts
está deshabilitada en este sistema"*, corre primero la línea del
`Set-ExecutionPolicy`, que solo afecta a esta ventana y se revierte al cerrarla:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Sabes que funcionó porque el prompt pasa a empezar con `(.venv)`. Luego:

```powershell
python -m pip install --upgrade google-cloud-aiplatform
```

## Valores del proyecto

Ya están cableados como default dentro de los scripts, no hay que pasarlos.

| | |
| --- | --- |
| Proyecto | `si7016-262-nlp` |
| Región | `us-central1` (cuota L4 aprobada; alterna `us-west1`) |
| Bucket | `gs://asilvaz1taller3` |
| Repo de imágenes | `si7016-taller3` en Artifact Registry |

## Paso 1: setup, una sola vez

```powershell
powershell -ExecutionPolicy Bypass -File .\si7016-asilvaz1-taller3-262\src\deploy\00-setup-gcp.ps1
```

Va a abrir **dos** ventanas del navegador seguidas: la primera autentica el CLI
y la segunda las credenciales que usan las librerías de Python. Son dos cosas
distintas, hay que completar ambas.

El script es idempotente. Si el bucket o el repositorio ya existen, lo dice en
amarillo y sigue. Puedes volver a correrlo sin romper nada.

Al final imprime la verificación: APIs habilitadas, ubicación del bucket,
repositorio de imágenes y endpoints desplegados.

## Paso 2: descubrir el model card de Gemma

```powershell
python .\si7016-asilvaz1-taller3-262\src\deploy\01-list-gemma-model-garden.py
```

No despliega nada y no cuesta nada. Imprime el id exacto del modelo, si tu
proyecto ya aceptó la licencia de Gemma, y las combinaciones de contenedor,
máquina y acelerador que Google tiene verificadas para ese modelo.

**Copia el id del modelo que imprima.** Lo necesitas en el paso siguiente.

Si quieres filtrar solo las opciones con L4:

```powershell
python .\si7016-asilvaz1-taller3-262\src\deploy\01-list-gemma-model-garden.py --accelerator L4
```

## Paso 3: desplegar el endpoint base

Primero el ensayo en seco, que solo imprime lo que haría:

```powershell
python .\si7016-asilvaz1-taller3-262\src\deploy\02-deploy-model-garden-base.py --dry_run
```

Si los valores se ven bien, el despliegue real, con el id del paso 2:

```powershell
python .\si7016-asilvaz1-taller3-262\src\deploy\02-deploy-model-garden-base.py --model "google/gemma@gemma-7b-it"
```

Tarda entre 15 y 30 minutos y **no hay que cerrar la terminal**: el SDK se queda
esperando a que el endpoint responda. Cuando termina deja un
`.endpoint_base.json` al lado del script, que leen los pasos 4 y 5.

Si el paso 2 mostró un contenedor distinto al default, pásalo:

```powershell
python ...\02-deploy-model-garden-base.py --model "<id>" --container "<uri>" --machine_type g2-standard-12 --accelerator NVIDIA_L4
```

## Mientras el paso 3 corre: ver el avance

**Abre una segunda ventana de PowerShell.** No toques la que está desplegando:
si la interrumpes pierdes la referencia al endpoint y toca buscarlo a mano.

En la consola web:

- Endpoints: `https://console.cloud.google.com/vertex-ai/online-prediction/endpoints?project=si7016-262-nlp`
- Modelos: `https://console.cloud.google.com/vertex-ai/models?project=si7016-262-nlp`

Desde la terminal, con un vigilante que refresca solo:

```powershell
powershell -ExecutionPolicy Bypass -File .\si7016-asilvaz1-taller3-262\src\deploy\05-ver-despliegue.ps1
```

O a mano, cuando quieras (`gcloud` usa HTTPS, así que funciona aunque gRPC
esté bloqueado):

```powershell
gcloud ai endpoints list --region us-central1 --project si7016-262-nlp
gcloud ai endpoints describe <ENDPOINT_ID> --region us-central1 --project si7016-262-nlp
```

Cómo se lee el avance:

| Lo que ves | Qué significa |
| --- | --- |
| `endpoints list` vacío | Apenas arrancando, el endpoint todavía no se registra |
| El endpoint aparece pero `deployedModels` está vacío | El contenedor se está montando y el modelo se está cargando. Es la parte larga |
| `deployedModels` ya trae un id | Listo, y desde ese momento **está cobrando** |

Un modelo de 7B tarda entre 15 y 30 minutos, casi todo en el segundo estado.

## Paso 4: probar y generar las respuestas del modelo base

Este paso produce la línea base contra la que después se comparan el
fine-tuning y el RAG. El taller la exige explícitamente: primero se evalúa el
modelo base puro, después el afinado.

### 4.1 Prueba de humo, con `--debug`

Antes de gastar 60 llamadas, confirma que el endpoint responde y que el script
entiende el formato de su respuesta. Cada contenedor de serving devuelve la
predicción con nombres de campo distintos (`generated_text`, `text`,
`content`, ...), así que la primera vez conviene verla cruda:

```powershell
python .\si7016-asilvaz1-taller3-262\src\deploy\03-predict-endpoint.py `
    --prompt "What is the scope of ISO 1996-1?" --debug
```

Si en vez de texto sale un volcado JSON con campos raros, cópiamelo y ajusto el
parser antes de seguir.

### 4.2 Las tres técnicas sobre el split de evaluación

```powershell
cd .\si7016-asilvaz1-taller3-262

python .\src\deploy\03-predict-endpoint.py --dataset .\data\qa\normas_ruido_eval.jsonl `
    --out .\results\respuestas-base-zero-shot.jsonl --technique zero-shot

python .\src\deploy\03-predict-endpoint.py --dataset .\data\qa\normas_ruido_eval.jsonl `
    --out .\results\respuestas-base-few-shot.jsonl --technique few-shot

python .\src\deploy\03-predict-endpoint.py --dataset .\data\qa\normas_ruido_eval.jsonl `
    --out .\results\respuestas-base-cot.jsonl --technique chain-of-thought

cd ..
```

Son 20 llamadas por corrida, unos pocos minutos cada una. Cada línea del jsonl
de salida trae `question`, `reference` (la respuesta del dataset),
`prediction`, `technique`, `system`, `source_file` y `section_label`, que es
todo lo que necesita el cálculo de ROUGE.

Al final de cada corrida el script informa cuántas respuestas salieron con
error. Si son más de dos o tres, para y revisa antes de seguir.

### 4.3 Documentar los prompts

El taller pide el prompt exacto de cada técnica. Ya están en `prompts/`,
generados desde el propio script para que no se desincronicen:

```powershell
python .\si7016-asilvaz1-taller3-262\src\deploy\export_prompts.py
```

### 4.4 Apagar antes de cerrar

No dejes el endpoint prendido entre sesiones. Ver el paso 5.

## Paso 5: apagar

Esto es lo que cuida el presupuesto. Un endpoint desplegado cobra por hora
aunque no reciba ni una petición.

```powershell
python .\si7016-asilvaz1-taller3-262\src\deploy\04-undeploy-cleanup.py --list
```

Lo que aparezca marcado `<-- COBRANDO` está gastando plata. Para apagarlo:

```powershell
python .\si7016-asilvaz1-taller3-262\src\deploy\04-undeploy-cleanup.py
```

Eso hace *undeploy* del modelo, que detiene el cobro y deja el endpoint vacío
listo para reusar. Para borrarlo todo, al final del taller:

```powershell
python .\si7016-asilvaz1-taller3-262\src\deploy\04-undeploy-cleanup.py --delete-endpoint --delete-model
```

## Por qué el paso 2 no se salta

`create-model-agent-platform.py` del repo del curso apunta a Qwen y trae un
`serving_container_image_uri` fijo. Esos URI cambian de versión con el tiempo y
el id del model card de Gemma no es el mismo que el de Qwen. El script 01 le
pregunta al SDK qué combinaciones tiene Google verificadas para ese modelo, en
vez de adivinar.

## Si algo falla

| Síntoma | Qué pasa |
| --- | --- |
| `la ejecución de scripts está deshabilitada` | Falta el `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, o usa la forma `powershell -ExecutionPolicy Bypass -File ...` |
| `gcloud no esta en el PATH` | Abre una terminal **nueva** después de instalar gcloud; el PATH no se refresca en las ya abiertas |
| `ModuleNotFoundError: vertexai` | El entorno no está activado. El prompt debe empezar con `(.venv)` |
| `403 PERMISSION_DENIED` al desplegar | Falta `gcloud auth application-default login`, o el proyecto activo no es el correcto: `gcloud config list` |
| `Quota exceeded ... nvidia_l4_gpus` | La cuota está en otra región. Prueba `--region us-west1` en los scripts 01 a 04 |
| `UNAVAILABLE ... WSA Error ... 11001` o `503` | La red bloquea gRPC. Los scripts ya usan REST por defecto; si aun así falla, corre `00b-diagnostico-red.ps1` (abajo). Nada se desplegó ni se cobró: la llamada no salió del computador |
| El despliegue se queda colgado más de 40 min | Cancela con Ctrl+C y revisa el estado real con `gcloud ai endpoints list --region us-central1`. El endpoint puede haber quedado a medias y estar cobrando |

## Si la red bloquea la conexión: `WSA Error 11001`, `UNAVAILABLE`, `503`

Ese error no viene de GCP sino de la red. El SDK de Vertex AI puede hablar con
la API de dos formas, y no son equivalentes cuando hay firewall o proxy:

- **gRPC** abre su propio canal TCP al 443 y trae su propio resolvedor DNS
  (c-ares). **No respeta el proxy del sistema Windows.** Es lo que los
  firewalls institucionales bloquean con más frecuencia.
- **REST** es HTTPS normal, igual que `gcloud` y que el navegador, y sí respeta
  las variables de proxy.

**Por eso los scripts 01 a 04 usan REST por defecto** (`--transport rest`). Si
alguna vez necesitas el otro, pasa `--transport grpc`.

Si sigue fallando, corre el diagnóstico:

```powershell
powershell -ExecutionPolicy Bypass -File .\si7016-asilvaz1-taller3-262\src\deploy\00b-diagnostico-red.ps1
```

Revisa siete cosas y termina con una conclusión: resolución DNS, conexión TCP
al 443, servidores DNS del equipo, adaptadores activos (una VPN aparece ahí),
proxy configurado, y por último una llamada real a la API **por REST y por
gRPC por separado**, que es lo que distingue los casos.

| Lo que sale | Qué significa | Qué hacer |
| --- | --- | --- |
| REST OK, gRPC falla | El firewall bloquea gRPC, que es lo normal en red institucional | Nada, los scripts ya usan REST |
| Los dos fallan y el punto 5 muestra un proxy | El proxy corporativo necesita declararse | Exporta `HTTPS_PROXY` y `HTTP_PROXY` (abajo) |
| Los dos fallan y no hay proxy | La red bloquea del todo la salida a la API | Prueba desde otra red, o usa Cloud Shell |
| Falla el punto 1 (DNS) | Es la VPN o el DNS del equipo | Desconecta la VPN, o pon el DNS en `8.8.8.8` |

Para declarar un proxy corporativo en la terminal actual:

```powershell
$env:HTTPS_PROXY = "http://usuario:clave@servidor:puerto"
$env:HTTP_PROXY  = $env:HTTPS_PROXY
```

**La salida que no depende de tu red** es Cloud Shell, que corre dentro de
Google y trae `gcloud` y Python ya autenticados:

`https://console.cloud.google.com/?cloudshell=true&project=si7016-262-nlp`

Ahí clonas el repo, corres el mismo script 02 y el endpoint queda desplegado
igual. Una vez desplegado, para consultarlo desde tu máquina basta `gcloud ai
endpoints predict`, que es HTTPS normal.

Para confirmar que no quedó nada desplegado cobrando, sin depender de gRPC:

```powershell
gcloud ai endpoints list --region us-central1 --project si7016-262-nlp
```

## Comandos sueltos de verificación

```powershell
gcloud config list                                   # proyecto y región activos
gcloud services list --enabled --project si7016-262-nlp
gcloud storage buckets describe gs://asilvaz1taller3 --format="value(location,locationType)"
gcloud artifacts repositories list --location us-central1
gcloud ai endpoints list --region us-central1        # lo que está desplegado
gcloud compute instances list                        # VM encendidas
gcloud storage du -s gs://asilvaz1taller3            # espacio usado
```

La cuota de GPU L4 no se ve bien por `gcloud`; revísala en la consola:
`https://console.cloud.google.com/iam-admin/quotas?project=si7016-262-nlp`,
servicio **Vertex AI API**, métrica `custom_model_training_nvidia_l4_gpus`.

## Archivos de estado

`02-deploy-model-garden-base.py` deja un `.endpoint_base.json` al lado, que
leen los scripts 03 y 04 para no tener que copiar ids a mano. Está en
`.gitignore` porque contiene identificadores del proyecto.
