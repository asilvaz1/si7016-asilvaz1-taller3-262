# Fase 3 — Despliegue en VM con vLLM, app de consulta y RAG en Vertex AI RAG Engine

Entrega: **miércoles 23 de septiembre** (plazo extendido). Este documento es el
runbook de la fase y, sobre todo, el criterio de qué conviene hacer a mano y qué
tiene que quedar en un script.

## 0. Qué falta y qué ya está

Cerrado en las fases anteriores: corpus, dataset de 100 pares, línea base del
modelo puro, fine-tuning con QLoRA, fusión de adaptadores, las 60 respuestas del
modelo afinado y el RAG local con FAISS. La tabla comparativa de los tres
sistemas ya existe en `results/metricas-comparativas.csv`.

Esta fase agrega tres cosas:

| Entregable | Archivo nuevo |
| --- | --- |
| El modelo afinado servido con vLLM en una VM | `src/deploy/07-vm-vllm.sh`, `src/deploy/08-predict-vllm.py` |
| Una app de consulta tipo Streamlit | `app/streamlit_app.py` |
| El corpus migrado a Vertex AI RAG Engine | `src/rag/vertex_rag_engine.py` |

## 1. El criterio: qué a mano y qué por script

La pregunta no es "cuál es más cómodo", es **qué tiene que ser reproducible**.
Todo lo que produce un número que va al informe tiene que estar en un script,
porque alguien debería poder correrlo otra vez y obtener lo mismo. Todo lo que
se hace una sola vez y se verifica mirando, va a mano: un script alrededor de un
`gcloud` que se ejecuta una vez esconde el error en vez de mostrarlo.

**A mano, en la consola o pegando comandos** — una sola vez, se ve al instante si
salió bien:

| Paso | Por qué a mano |
| --- | --- |
| Arrancar la VM y comprobar que tiene la L4 | Un vistazo a `nvidia-smi` lo dice todo |
| Crear el bucket `us-west1` del corpus RAG | Una línea, y el nombre tiene que ser único globalmente |
| Dar el permiso IAM al agente de servicio de RAG Engine | Requiere el número del proyecto; es más claro copiarlo que parametrizarlo |
| Abrir el túnel SSH | Es una sesión interactiva, no un paso de pipeline |
| **Apagar la VM al terminar** | Es la decisión de costo del taller y no debe estar escondida en ningún script |
| Las capturas de pantalla de la consola | Son la evidencia; se toman mirando |

**Por script, versionado en el repo** — produce entregables o se repite:

| Paso | Script |
| --- | --- |
| Instalar vLLM y bajar el modelo fusionado | `07-vm-vllm.sh install` / `sync` |
| Arrancar vLLM con los flags de memoria correctos | `07-vm-vllm.sh serve` |
| Generar las 60 respuestas por el endpoint | `08-predict-vllm.py --technique all` |
| Crear el corpus, importar y consultar RAG Engine | `vertex_rag_engine.py` |
| Las 20 preguntas contra el RAG Engine | `vertex_rag_engine.py run` |
| La tabla comparativa | `src/eval/run_eval.py` |

Los flags de vLLM son el caso donde esto más se nota. Son cinco valores
relacionados entre sí por una cuenta de memoria; escritos a mano en la terminal,
el intento tres ya no se parece al intento uno y no hay forma de saber cuál
produjo el resultado que se reportó.

## 2. Bloque A — vLLM en la VM

### A0. Comprobar la VM (a mano)

```powershell
gcloud compute instances list --project=si7016-262-nlp
gcloud compute instances describe NOMBRE_VM --zone=ZONA `
    --format="value(machineType,guestAccelerators,status)"
```

Lo que tiene que cumplirse: **una GPU de 24 GB o más** (L4, A10G, A100) y **al
menos 40 GB de disco libre**. El modelo fusionado en bf16 pesa 17.1 GB y vLLM con
su propio PyTorch pesa otros 10 a 15 GB.

Si la VM tiene una T4 (16 GB), este camino no sirve: gemma-7b-it en bf16 no cabe.
La salida en ese caso es servir el modelo base en 4 bits con los adaptadores
encima, que es lo que hace `infer_gemma.py` del repositorio del curso, y
documentar por qué.

Si la VM no tiene GPU asignada, se le agrega apagada:

```powershell
gcloud compute instances stop NOMBRE_VM --zone=ZONA
gcloud compute instances attach-disk ...   # solo si falta disco
```

### A1. Copiar el script y correrlo (script)

```powershell
gcloud compute scp D:\EAFIT\maestria_eafit\13_nlp\si7016-asilvaz1-taller3-262\src\deploy\07-vm-vllm.sh `
    si7016-262-asilvaz1:07-vm-vllm.sh --zone=us-west1-a --project=si7016-262-nlp
gcloud compute ssh si7016-262-asilvaz1 --zone=us-west1-a --project=si7016-262-nlp
```

**Nota de Windows.** `gcloud compute scp` y `gcloud compute ssh` usan `pscp.exe`
y `putty.exe` del SDK, no OpenSSH. Dos consecuencias que cuestan tiempo:

- `pscp` **no expande `~`**. Un destino `VM:~/` falla con `unable to open ~/:
  failure`. Hay que escribir la ruta relativa: `VM:07-vm-vllm.sh`.
- Si `pscp` falla igual (clave sin generar, passphrase, PuTTY mal configurado),
  no insistas. El script se sube por el **SSH del navegador** de la consola, con
  el boton *Subir archivo* de la rueda dentada, o por el bucket:

```powershell
gcloud storage cp D:\EAFIT\maestria_eafit\13_nlp\si7016-asilvaz1-taller3-262\src\deploy\07-vm-vllm.sh `
    gs://asilvaz1taller3/scripts/07-vm-vllm.sh
```

```bash
# ya dentro de la VM
gcloud storage cp gs://asilvaz1taller3/scripts/07-vm-vllm.sh .
```

Los pasos A1 a A3 de abajo funcionan igual desde el SSH del navegador. El unico
que no, es el tunel: ver la nota del paso A2.

Ya dentro de la VM:

```bash
chmod +x 07-vm-vllm.sh
./07-vm-vllm.sh check      # aborta temprano si falta driver, VRAM o disco
./07-vm-vllm.sh install    # 10 a 15 min, una sola vez
./07-vm-vllm.sh sync       # baja 17 GB de GCS, 5 a 10 min
./07-vm-vllm.sh serve      # arranca vLLM en background
./07-vm-vllm.sh status     # espera a que /health responda; si murió, muestra el log
./07-vm-vllm.sh test       # una pregunta de humo
```

**Si `install` falla con `ensurepip is not available`.** Las imagenes de
Debian y Ubuntu de Compute Engine traen `python3` pero no el modulo `venv`, asi
que `python3 -m venv` aborta y deja el directorio a medias. El script ya lo
detecta e instala el paquete solo, pero si vienes de un intento anterior:

```bash
sudo apt-get update && sudo apt-get install -y python3.10-venv python3-dev build-essential
rm -rf ~/vllm-venv
./07-vm-vllm.sh install
```

El numero de version tiene que coincidir con el `python3` de la VM
(`python3 --version`). En Ubuntu 22.04 es `python3.10-venv`; en Debian 12,
`python3.11-venv`.

Los flags no son arbitrarios. La cuenta de la L4:

| Concepto | GB |
| --- | --- |
| Memoria de la L4, medida con `nvidia-smi` | 22.5 |
| Presupuesto con `--gpu-memory-utilization=0.92` | 20.7 |
| Pesos de gemma-7b-it en bf16 (8.54 B parámetros) | 17.1 |
| Queda para caché KV, activaciones y grafos CUDA | 3.6 |

La primera fila es el dato medido en la VM, no el nominal: la L4 se vende como
24 GB y `nvidia-smi` reporta 23034 MiB, que son 22.5 GiB. La diferencia se la
come el ECC y la reserva del firmware, y son 1.5 GiB que sí cambian la cuenta.

`--enforce-eager` libera 1 a 2 GB que vLLM gastaría capturando grafos CUDA, y
`--max-model-len=1024` es holgado: el prompt más largo es el de few-shot, unos
250 tokens, más 256 de generación. Son las mismas correcciones que necesitaba el
endpoint de Vertex; la diferencia es que aquí el log se ve en vivo y un
reintento cuesta segundos.

Si aparece `CUDA out of memory` o `No available memory for the cache blocks`:

```bash
MAX_LEN=768 MAX_SEQS=2 ./07-vm-vllm.sh serve
```

### A2. El túnel SSH (a mano)

```powershell
gcloud compute ssh si7016-262-asilvaz1 --zone=us-west1-a --project=si7016-262-nlp `
    --ssh-flag="-N" --ssh-flag="-L" --ssh-flag="8000:localhost:8000"
```

Un flag por bandera y sin espacios adentro. La forma `-- -N -L 8000:...`, que es
la que documenta casi todo el mundo, **no funciona en PowerShell**: el shell se
queda con el `--` y gcloud recibe `-N` como argumento propio, con
`unrecognized arguments`. En bash o en Cloud Shell sí funciona.

Esa ventana se queda abierta y sin imprimir nada, que es lo esperado. Para
comprobar el túnel, desde otra ventana:

```powershell
curl.exe http://localhost:8000/v1/models
```

**No se abre el puerto 8000 en el firewall.** Un servidor de inferencia
expuesto a internet sin autenticación es un problema de seguridad real, no una
formalidad. El túnel además viaja por SSH sobre el puerto 22, que es lo que la
red del campus deja pasar; el DNS de `.goog` allá no resuelve.

**Si PuTTY no coopera o el 22 está filtrado**, el túnel se hace por IAP, que no
usa SSH: viaja por HTTPS contra la API de Google.

```powershell
gcloud compute firewall-rules create allow-iap-tunnel --network=default `
    --direction=INGRESS --action=allow --rules=tcp:22,tcp:8000 `
    --source-ranges=35.235.240.0/20 --project=si7016-262-nlp

gcloud compute start-iap-tunnel si7016-262-asilvaz1 8000 `
    --local-host-port=localhost:8000 --zone=us-west1-a --project=si7016-262-nlp
```

El rango `35.235.240.0/20` es el de los servidores de IAP, no internet, así que
el puerto 8000 sigue sin estar expuesto públicamente. Borra la regla al
terminar:

```powershell
gcloud compute firewall-rules delete allow-iap-tunnel --project=si7016-262-nlp
```

**Tercera opción**, si ya tienes llave propia y el 22 abierto desde tu IP, es
el `ssh` directo del repositorio del curso, cambiando el puerto de 8888 (que
allá es para Jupyter) a 8000:

```powershell
ssh -i ~\.ssh\gcp_key USUARIO@IP_PUBLICA -N -L 8000:localhost:8000
```

### A3. Las 60 respuestas por el endpoint (script)

```powershell
python .\si7016-asilvaz1-taller3-262\src\deploy\08-predict-vllm.py `
    --dataset data\qa\normas_ruido_eval.jsonl `
    --out results\respuestas-finetuning-vllm.jsonl --technique all
```

Este archivo **no reemplaza** a `results/respuestas-finetuning.jsonl`, que ya se
midió y está en la tabla. Sirve para confirmar que el endpoint devuelve lo mismo
que el job por lotes, que es la evidencia de que el despliegue funciona de
verdad y no solo arranca:

```powershell
python src\eval\comparar_finetuning_vllm.py
```

No tienen por qué coincidir al 100%: `transformers` y vLLM usan kernels
distintos y con `do_sample=False` igual hay diferencias de redondeo que cambian
algún token. Coincidencia alta y contenido equivalente es lo que se espera; si
sale muy baja, algo distinto se está cargando.

## 3. Bloque B — la app de Streamlit

```powershell
pip install streamlit
streamlit run .\si7016-asilvaz1-taller3-262\app\streamlit_app.py
```

Con el túnel abierto, la app encuentra el endpoint en `localhost:8000`. Tres
pestañas: modelo afinado, RAG Engine, y las dos respuestas lado a lado sobre la
misma pregunta. Esa tercera es la captura que mejor resume el taller, porque es
la versión visual de la conclusión que ya está escrita: el afinado sabe de qué
habla cada norma, el RAG copia las cifras del texto recuperado.

La app lee las plantillas de prompt del mismo sitio que los scripts
(`src/deploy/plantillas.py`, que a su vez lee el diccionario de
`03-predict-endpoint.py`), así que lo que se ve en pantalla es literalmente lo
que se midió.

Queda local a propósito. Contenerizarla y subirla a Cloud Run agrega build,
permisos y costo, y no aporta nada que el taller pida.

## 4. Bloque C — RAG Engine

### C0. Entorno aparte (a mano, una vez)

`google-cloud-agentplatform` sube `google-cloud-aiplatform` a la línea 2.x, y el
resto del repo está fijado en 1.75.0. Instalarlo encima rompería los scripts de
despliegue.

```powershell
python -m venv .venv-rag
.venv-rag\Scripts\activate
pip install -r requirements-fase3.txt
```

### C1. Bucket y permiso (a mano)

El corpus va en **us-west1**, no en us-central1: el modo *Spanner* de RAG
Engine, que es el que toma por defecto un proyecto nuevo, está restringido a
proyectos en lista blanca en us-central1, us-east1 y us-east4, y falla con
`INVALID_ARGUMENT ... restricted to only allowlisted projects`.

```powershell
# El bucket de us-west1 ya existe desde el job de inferencia. Se comprueba
# la region en vez de crearlo: el corpus tiene que estar en la misma.
gcloud storage buckets describe gs://asilvaz1taller3-west --format="value(location)"

$NUM = gcloud projects describe si7016-262-nlp --format="value(projectNumber)"
gcloud storage buckets add-iam-policy-binding gs://asilvaz1taller3-west `
    --member="serviceAccount:service-$NUM@gcp-sa-vertex-rag.iam.gserviceaccount.com" `
    --role="roles/storage.objectViewer"
```

Ese binding es el error silencioso típico: sin él, la importación no falla, solo
deja el corpus vacío y la primera consulta no devuelve nada.

### C2. Corpus y consultas (script)

```powershell
python src\rag\vertex_rag_engine.py upload    # sube los 33 .txt fuente
python src\rag\vertex_rag_engine.py create    # crea el corpus
python src\rag\vertex_rag_engine.py import    # chunk 512 / solape 100, asíncrono
python src\rag\vertex_rag_engine.py status    # tienen que aparecer 33 archivos
python src\rag\vertex_rag_engine.py ask --q "What does ISO 3382-1 specify?"
```

Se suben **33 de los 41** documentos: los 8 descartados en `corpus_map.csv` son
duplicados o ediciones superadas, y meterlos en el índice haría que el RAG
pudiera recuperar la versión equivocada de una misma norma. El corpus es el
mismo que usa el RAG con FAISS, que es lo que hace comparables los dos
recuperadores.

`ask` hace además la generación anclada por herramienta del notebook de la
clase, y muestra la respuesta con y sin RAG sobre la misma pregunta. Es la
captura de evidencia del bloque.

### C3. Las 20 preguntas medidas (script)

```powershell
python src\rag\vertex_rag_engine.py run --technique all
python src\eval\run_eval.py --detalle
```

`run_eval.py` ya reconoce el sistema `rag-vertex` y lo agrega a
`results/metricas-comparativas.csv` con sus propias filas de ROUGE y Recall@K.

Hay una decisión que conviene tomar consciente, porque cambia lo que la tabla
significa:

- `--generator gemini` (por defecto): recupera de RAG Engine y genera con
  Gemini. Es el pipeline administrado completo, y es lo que enseñó la clase.
  **Pero cambia dos cosas a la vez**, el recuperador y el generador, así que la
  diferencia contra el RAG de la Fase 1 no se le puede atribuir a ninguno de los
  dos por separado.
- `--generator vllm`: recupera de RAG Engine y genera con el **modelo afinado**
  servido en la VM. Cambia solo el recuperador, así que la diferencia sí mide la
  recuperación. Requiere el túnel abierto.

Correr las dos cuesta pocos centavos y la comparación entre ellas es, con
diferencia, el párrafo más interesante que puede tener el informe. Si solo da
tiempo para una, la de Gemini es la que responde al enunciado; la de vLLM es la
que responde a la pregunta científica.

Sobre Recall@K: RAG Engine no expone la posición del chunk dentro del documento,
solo un identificador opaco, así que el Recall a nivel de chunk **no** es
comparable con el del FAISS. El de documento sí, y es el que reporta la tabla.

## 5. Apagar (a mano, el mismo día)

```powershell
gcloud compute instances stop NOMBRE_VM --zone=ZONA
python src\deploy\04-undeploy-cleanup.py --list
python src\rag\vertex_rag_engine.py delete     # solo cuando ya no se consulte
```

La VM cobra mientras esté **encendida**, corra vLLM o no. `07-vm-vllm.sh stop`
solo mata el proceso; no deja de facturar. A US$0.56 la hora de L4 en Compute
Engine, una VM olvidada un fin de semana son unos US$27.

El corpus de RAG Engine cobra almacenamiento vectorial mientras exista. Es poco
(33 documentos de texto), pero conviene borrarlo después de tomar las capturas.

## 6. Costo estimado de esta fase

| Concepto | Estimado |
| --- | --- |
| VM con L4, 3 a 4 horas de trabajo | US$2.00 a 2.50 |
| Embeddings de los 33 documentos, una vez | centavos |
| Almacenamiento vectorial del corpus, unos días | centavos |
| Generación con Gemini, 80 llamadas | centavos |
| **Total de la fase** | **US$2 a 3** |

Suma sobre los US$4 a 7 ya gastados, así que el taller cierra alrededor de
US$7 a 10, dentro del presupuesto de la hoja de ruta.

## 7. Evidencia que hay que capturar

1. `nvidia-smi` dentro de la VM con el modelo cargado, que muestra la VRAM ocupada.
2. El log de arranque de vLLM con la línea de `KV cache size`.
3. La app de Streamlit respondiendo, en la pestaña lado a lado.
4. La consola en *Agent Platform → RAG Engine* con el corpus y sus 33 archivos.
5. La salida de `ask`, con y sin RAG sobre la misma pregunta.
6. La instancia **detenida** al final, que es la prueba de que se controló el costo.
