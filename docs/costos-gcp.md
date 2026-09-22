# Costos en GCP

Proyecto `si7016-262-nlp`. El enunciado pide uso prudente de los créditos, así
que este documento registra qué se encendió, por cuánto tiempo y qué sigue
generando costo.

## Principio que guió todas las decisiones

**Nada que cobre por hora se deja encendido.** En concreto:

- El entrenamiento, la fusión y la inferencia se hicieron con **Custom Jobs de
  Vertex**, que se apagan solos al terminar, en vez de una VM de Compute Engine
  que alguien puede olvidar prendida. Esa fue una decisión explícita de diseño,
  no una casualidad.
- La fusión de adaptadores corrió en **CPU**, porque es aritmética de pesos y no
  necesita GPU. Así no consumió la única L4 del proyecto ni pagó por ella.
- La Fase 1 completa (RAG local con FAISS) corrió en CPU local, con costo cero
  en GCP.
- El endpoint del modelo base se apagó con `04-undeploy-cleanup.py` en cuanto
  terminó de generar las 60 respuestas de la línea base.

## Qué se encendió

| Recurso | Máquina | Duración | Fuente del dato |
| --- | --- | --- | --- |
| Endpoint del modelo base (Model Garden) | 1 x L4 | horas de prueba | estimado |
| Prueba de humo del fine-tuning | `g2-standard-12` + 1 x L4 | ~45 min | estimado, incluye una descarga lenta de shards |
| Entrenamiento completo | `g2-standard-12` + 1 x L4 | **847 s** | medido, log del job |
| Fusión de adaptadores | `n1-highmem-8`, sin GPU | ~40 min | estimado |
| Inferencia del modelo afinado | `g2-standard-12` + 1 x L4 (`us-west1`) | ~25 min | estimado |
| Cloud Build (imagen de entrenamiento) | compartido | varios builds de 5 a 12 min | dentro de la capa gratuita diaria |

Las duraciones exactas de los tres jobs se obtienen así:

```powershell
gcloud ai custom-jobs list --project=si7016-262-nlp --region=us-central1 `
    --format="table(displayName,state,createTime,startTime,endTime)"
gcloud ai custom-jobs list --project=si7016-262-nlp --region=us-west1 `
    --format="table(displayName,state,createTime,startTime,endTime)"
```

## Precios de referencia

On-demand, `us-central1`, septiembre de 2026. El precio de la GPU es el término
dominante; el tipo de máquina (vCPU y RAM) se suma aparte.

| Recurso | Precio aprox. |
| --- | --- |
| GPU NVIDIA L4, Vertex Custom Training | US$0.64 / hora |
| GPU NVIDIA L4, Compute Engine | US$0.56 / hora |
| GPU NVIDIA T4, Compute Engine | US$0.35 / hora |
| Almacenamiento estándar regional | ~US$0.020 / GB / mes |
| Transferencia entre regiones dentro de EE. UU. | ~US$0.02 / GB |

## Estimado del gasto

| Concepto | Estimado |
| --- | --- |
| Endpoint del modelo base, sesión de la línea base | US$2 a 4 |
| Prueba de humo del fine-tuning | US$0.50 a 0.80 |
| Entrenamiento completo (847 s) | ~US$0.20 |
| Fusión de adaptadores (CPU) | US$0.30 a 0.50 |
| Inferencia del modelo afinado | ~US$0.35 |
| Transferencia del modelo fusionado a `us-west1` (17 GB) | ~US$0.35 |
| Almacenamiento del mes (adaptadores + fusionado + artefactos, ~18 GB) | ~US$0.36 |
| **Total estimado** | **US$4 a 7** |

Queda por debajo del presupuesto de US$7 a 16 que estimaba la hoja de ruta,
sobre todo porque el endpoint del modelo afinado nunca llegó a desplegarse: los
intentos fallidos no alcanzaron a facturar tiempo de GPU, ya que el contenedor
moría antes de quedar en servicio o el job se quedaba en `PENDING`, estado que no
cobra.

El gasto real se consulta en la consola, en **Facturación → Informes**,
filtrando por el proyecto `si7016-262-nlp` y agrupando por SKU.

## Fase 3: la VM con vLLM y el RAG Engine

| Recurso | Máquina | Duración | Fuente del dato |
| --- | --- | --- | --- |
| VM de Compute Engine con vLLM | `g2` + 1 x L4, `us-west1-a` | una sesión de trabajo | estimado |
| Embeddings de los 33 documentos | administrado | una vez | dentro de lo mínimo facturable |
| Almacenamiento vectorial del corpus | administrado | mientras exista | centavos al mes |
| Generación con Gemini (`ask`, la app y las 80 respuestas del eval) | administrado | ~100 llamadas | centavos |

Una nota de la corrida de las 80 respuestas: una pregunta falló con `429
RESOURCE_EXHAUSTED`, que es cuota por minuto y no presupuesto agotado. El
script reintenta ahora ese error con espera creciente, y tiene un modo
`run --reparar` que reintenta solo las filas sin respuesta sin repetir las
demás, que además de costar menos evita que una segunda corrida cambie las 79
respuestas que ya estaban medidas.

| Concepto | Estimado |
| --- | --- |
| VM con L4, 3 a 4 horas | US$2.00 a 2.50 |
| Embeddings, almacenamiento vectorial y generación | < US$0.50 |
| **Total de la fase** | **US$2 a 3** |

Con esto el taller cierra alrededor de **US$7 a 10**, dentro del presupuesto de
US$7 a 16 de la hoja de ruta.

Dos diferencias de la VM frente a los Custom Jobs, que son las que hay que tener
presentes:

- **La VM cobra mientras esté encendida, corra o no vLLM.** `07-vm-vllm.sh stop`
  solo mata el proceso. Lo que deja de facturar es
  `gcloud compute instances stop`. A US$0.56 la hora de L4 en Compute Engine,
  una VM olvidada un fin de semana son unos US$27.
- **Se hace `stop` y no `delete`.** El disco conserva vLLM instalado y los 17 GB
  del modelo, así que una demostración posterior arranca en dos minutos en vez
  de repetir `install` y `sync`. Un disco parado cuesta centavos al mes.

El corpus de RAG Engine cobra almacenamiento vectorial mientras exista. Es poco,
pero `src/deploy/09-apagar-todo.ps1 borrar-corpus` lo elimina cuando ya no se
vaya a consultar.

## Apagar al cerrar cada sesión

Hay un script que hace las dos cosas y distingue entre ellas: apaga lo que cobra
por tiempo encendido y **reporta** lo que cobra por existir, sin borrarlo.

```powershell
powershell -File src\deploy\09-apagar-todo.ps1 estado    # no toca nada
powershell -File src\deploy\09-apagar-todo.ps1 apagar    # VM, endpoints, regla de firewall
```

## Qué sigue costando después de la entrega

Lo único vivo es almacenamiento. El modelo fusionado pesa unos 17 GB y es de
lejos lo más grande.

```powershell
gcloud storage du -s gs://asilvaz1taller3 --readable-sizes
```

Para borrarlo cuando ya no se necesite:

```powershell
gcloud storage rm -r gs://asilvaz1taller3/normas-ruido-merged
gcloud storage rm -r gs://asilvaz1taller3-west/normas-ruido
```

Ojo con `gs://asilvaz1taller3-west`: además de los artefactos del job de
inferencia, ahora guarda los 33 documentos que alimentan el corpus de RAG
Engine. Borrar el bucket entero deja el corpus sin fuente. Si el corpus ya se
borró con `09-apagar-todo.ps1 borrar-corpus`, entonces sí se puede eliminar
completo.

Conviene **conservar** `gs://asilvaz1taller3/normas-ruido-lora/`: son unos 200 MB
y contienen los adaptadores entrenados, que es el resultado real del
fine-tuning. Con ellos y el modelo base se puede reconstruir todo.

## Antes de cerrar cada sesión de trabajo

```powershell
python .\si7016-asilvaz1-taller3-262\src\deploy\04-undeploy-cleanup.py --list
powershell -File .\si7016-asilvaz1-taller3-262\src\deploy\06-deploy-finetuned.ps1 status
```

El primero lista lo que está desplegado y cobrando. El segundo muestra el estado
del endpoint del modelo afinado. Un endpoint olvidado encendido es el único
riesgo real de costo de este taller: una L4 a US$0.64 la hora son unos US$15 al
día y US$460 al mes.

## Cuotas

La cuota aprobada es `custom_model_training_nvidia_l4_gpus`, con límite de **1
GPU por región** en `us-central1` y `us-west1`. Es importante no confundirla con
`custom_model_serving_nvidia_l4_gpus`, que es la de los endpoints y es una cuota
distinta.

Tener la cuota aprobada no garantiza que haya GPU disponible: el job de
inferencia falló tres veces en `us-central1` con `Resources are insufficient in
region`, con la cuota al 0% de uso. Es capacidad física de la región, no cuota, y
se resolvió moviendo el job a `us-west1`.
