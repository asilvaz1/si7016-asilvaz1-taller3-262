# 06-deploy-finetuned.ps1 - Despliegue del modelo AFINADO (fusionado) en Vertex.
#
# Port de ft-gemma-vertex/c-deploy-gcloud.sh del repositorio del curso, con tres
# diferencias: esta en PowerShell, trae los valores del proyecto cableados, y al
# terminar escribe .endpoint_finetuned.json con el mismo esquema que
# .endpoint_base.json, para que 03-predict-endpoint.py lo lea con
# --endpoint_file .endpoint_finetuned.json
#
# Usa gcloud, que habla HTTPS, asi que funciona aunque la red del campus
# bloquee gRPC.
#
# REQUISITO: el modelo fusionado tiene que estar ya en GCS. Lo deja ahi
# src/finetuning/submit_merge_job.py
#
# COMO EJECUTARLO (desde la carpeta 13_nlp), en este orden:
#   powershell -ExecutionPolicy Bypass -File .\...\06-deploy-finetuned.ps1 upload
#   powershell -ExecutionPolicy Bypass -File .\...\06-deploy-finetuned.ps1 create-endpoint
#   powershell -ExecutionPolicy Bypass -File .\...\06-deploy-finetuned.ps1 deploy
#   powershell -ExecutionPolicy Bypass -File .\...\06-deploy-finetuned.ps1 predict
#   powershell -ExecutionPolicy Bypass -File .\...\06-deploy-finetuned.ps1 status
#   powershell -ExecutionPolicy Bypass -File .\...\06-deploy-finetuned.ps1 cleanup
#
# ATENCION AL COSTO: 'deploy' enciende una L4 que cobra por hora hasta que
# corras 'cleanup'. No lo dejes prendido de un dia para otro.

param(
    [Parameter(Position = 0)]
    [ValidateSet("upload", "create-endpoint", "deploy", "predict", "status", "logs", "cleanup")]
    [string]$Comando = "status",

    # Sobrescriben los defaults de abajo sin editar el script. Utiles si
    # 01-list-gemma-model-garden.py reporta otra combinacion verificada.
    [string]$MachineType = "",
    [string]$VllmImage = ""
)

$PSNativeCommandUseErrorActionPreference = $false
$ErrorActionPreference = "Continue"

# --- Parametros del proyecto -------------------------------------------------
$PROJECT     = "si7016-262-nlp"
$REGION      = "us-central1"
$BUCKET      = "asilvaz1taller3"
$MERGED_DIR  = "gs://$BUCKET/normas-ruido-merged"

$MODEL_NAME    = "gemma-7b-it-normas-ruido-merged"
$ENDPOINT_NAME = "$MODEL_NAME-endpoint"

# g2-standard-4 tiene 1 x L4 pero solo 16 GB de RAM de host, y el modelo
# fusionado en bf16 pesa ~17 GB: el contenedor de vLLM se queda sin memoria
# al cargarlo y muere con "Model server exited unexpectedly".
# g2-standard-12 lleva la misma L4 con 48 GB de RAM.
$MACHINE_TYPE      = "g2-standard-12"
$ACCELERATOR_TYPE  = "nvidia-l4"
$ACCELERATOR_COUNT = 1

# Misma imagen de serving que usa el repo del curso. Si Model Garden publica
# una version mas nueva, se puede actualizar aqui.
$VLLM_IMAGE = "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20241210_0916_RC00"

if ($MachineType) { $MACHINE_TYPE = $MachineType }
if ($VllmImage)   { $VLLM_IMAGE   = $VllmImage }

$AQUI    = Split-Path -Parent $MyInvocation.MyCommand.Path
$ESTADO  = Join-Path $AQUI ".endpoint_finetuned.json"

function Paso($t)  { Write-Host "`n== $t ==" -ForegroundColor Cyan }
function Ok($t)    { Write-Host "   $t" -ForegroundColor Green }
function Aviso($t) { Write-Host "   $t" -ForegroundColor Yellow }
function Falla($t) { Write-Host "   $t" -ForegroundColor Red }

function Leer-Estado {
    if (Test-Path $ESTADO) { return Get-Content $ESTADO -Raw | ConvertFrom-Json }
    return $null
}

function Guardar-Estado($obj) {
    $obj | ConvertTo-Json -Depth 5 | Set-Content $ESTADO -Encoding UTF8
}

# --- upload -------------------------------------------------------------------
function Cmd-Upload {
    Paso "Comprobando que el modelo fusionado exista"
    $archivos = gcloud storage ls "$MERGED_DIR/" 2>$null
    if (-not $archivos) {
        Falla "No hay nada en $MERGED_DIR"
        Falla "Corre primero: python .\si7016-asilvaz1-taller3-262\src\finetuning\submit_merge_job.py"
        exit 1
    }
    if (-not ($archivos -match "config\.json")) {
        Falla "En $MERGED_DIR no aparece config.json. El modelo esta incompleto."
        exit 1
    }
    if (-not ($archivos -match "\.safetensors")) {
        Falla "En $MERGED_DIR no hay shards .safetensors. El modelo esta incompleto."
        exit 1
    }
    Ok "modelo fusionado presente"

    # La especificacion del contenedor de un modelo del Model Registry es
    # INMUTABLE: no se pueden cambiar los --container-args de un modelo ya
    # subido. Cada vez que se tocan los argumentos de vLLM hay que correr
    # "upload" otra vez, que crea un modelo nuevo, y desplegar ese. El modelo
    # viejo se borra con: gcloud ai models delete <id> --region=us-central1
    Paso "Subiendo al Model Registry"
    # Argumentos ajustados despues de que el primer despliegue muriera con
    # "Model server exited unexpectedly". Las cuentas de la L4:
    #   memoria de la tarjeta                                    24.0 GB
    #   presupuesto con --gpu-memory-utilization=0.85           ~20.4 GB
    #   pesos de gemma-7b-it en bf16 (8.54 B parametros)        ~17.1 GB
    #   sobra para cache KV + activaciones + grafos CUDA         ~3.3 GB
    # En esos 3.3 GB no caben la captura de grafos CUDA (1 a 2 GB) mas una cache
    # KV para 2048 tokens (~0.44 MB por token en Gemma 7B). vLLM aborta al
    # arrancar y Vertex lo reporta como muerte del contenedor, sin decir que fue
    # memoria. Los cuatro cambios, en orden de cuanto liberan:
    #   --enforce-eager           quita los grafos CUDA: 1 a 2 GB
    #   --max-model-len=1024      el prompt mas largo (few-shot) son ~250 tokens
    #                             mas 256 de generacion; 2048 reservaba el doble
    #   --gpu-memory-utilization  0.92 sube el presupuesto a ~22 GB
    #   --swap-space=4            16 GB de RAM de host para intercambio de
    #                             bloques que con 4 secuencias no se usan
    $vllmArgs = "--model=$MERGED_DIR," +
                "--tensor-parallel-size=$ACCELERATOR_COUNT," +
                "--swap-space=4," +
                "--gpu-memory-utilization=0.92," +
                "--max-model-len=1024," +
                "--max-num-seqs=4," +
                "--enforce-eager," +
                "--dtype=bfloat16," +
                "--disable-log-stats"

    # El otro sintoma del fallo era el timeout. Sin
    # --container-deployment-timeout-seconds, Vertex usa 1800 s por defecto, y
    # el contenedor tiene que traer 17 GB desde GCS y arrancar vLLM antes de
    # contestar /ping. Cuando la copia va lenta, el reloj se acaba y el
    # despliegue se declara fallido aunque el contenedor estuviera sano. Los
    # notebooks de Model Garden ponen 7200 s por esta misma razon.
    gcloud ai models upload `
        --project=$PROJECT --region=$REGION `
        --display-name=$MODEL_NAME `
        --container-image-uri=$VLLM_IMAGE `
        --container-command="python,-m,vllm.entrypoints.api_server,--host=0.0.0.0,--port=8080" `
        --container-args=$vllmArgs `
        --container-ports=8080 `
        --container-predict-route=/generate `
        --container-health-route=/ping `
        --container-deployment-timeout-seconds=7200 `
        --container-shared-memory-size-mb=16384 `
        --container-startup-probe-period-seconds=60 `
        --container-startup-probe-timeout-seconds=60
    if ($LASTEXITCODE -ne 0) { Falla "Fallo la subida del modelo."; exit 1 }

    $modelId = gcloud ai models list --project=$PROJECT --region=$REGION `
        --filter="displayName=$MODEL_NAME" --sort-by="~createTime" --limit=1 `
        --format="value(name)"
    $est = Leer-Estado
    if (-not $est) { $est = [pscustomobject]@{} }
    $est | Add-Member -NotePropertyName project -NotePropertyValue $PROJECT -Force
    $est | Add-Member -NotePropertyName region -NotePropertyValue $REGION -Force
    $est | Add-Member -NotePropertyName merged_dir -NotePropertyValue $MERGED_DIR -Force
    $est | Add-Member -NotePropertyName model_id -NotePropertyValue $modelId -Force
    $est | Add-Member -NotePropertyName model_display_name -NotePropertyValue $MODEL_NAME -Force
    $est | Add-Member -NotePropertyName dedicated -NotePropertyValue $false -Force
    Guardar-Estado $est
    Ok "modelo registrado: $modelId"
}

# --- create-endpoint ----------------------------------------------------------
function Cmd-CreateEndpoint {
    # Idempotente a proposito. Un endpoint duplicado no cobra por si solo, pero
    # es la forma tipica de terminar con una L4 desplegada en el endpoint que
    # nadie esta mirando.
    Paso "Buscando un endpoint que ya se llame asi"
    $existente = gcloud ai endpoints list --project=$PROJECT --region=$REGION `
        --filter="displayName=$ENDPOINT_NAME" --sort-by="~createTime" --limit=1 `
        --format="value(name.basename())"
    if ($existente) {
        Ok "ya existe: $existente (se reusa, no se crea otro)"
    }
    else {
        Paso "Creando el endpoint"
        gcloud ai endpoints create --project=$PROJECT --region=$REGION `
            --display-name=$ENDPOINT_NAME
        if ($LASTEXITCODE -ne 0) { Falla "Fallo la creacion del endpoint."; exit 1 }
    }

    $endpointId = gcloud ai endpoints list --project=$PROJECT --region=$REGION `
        --filter="displayName=$ENDPOINT_NAME" --sort-by="~createTime" --limit=1 `
        --format="value(name.basename())"
    $est = Leer-Estado
    if (-not $est) { Falla "Corre primero: upload"; exit 1 }
    $est | Add-Member -NotePropertyName endpoint_id -NotePropertyValue $endpointId -Force
    $est | Add-Member -NotePropertyName endpoint_display_name -NotePropertyValue $ENDPOINT_NAME -Force
    Guardar-Estado $est
    Ok "endpoint creado: $endpointId"
}

# --- deploy -------------------------------------------------------------------
function Cmd-Deploy {
    $est = Leer-Estado
    if (-not $est -or -not $est.model_id)    { Falla "Corre primero: upload"; exit 1 }
    if (-not $est.endpoint_id)               { Falla "Corre primero: create-endpoint"; exit 1 }

    # OJO: hay que sacar las propiedades a variables sueltas. Dentro de un
    # argumento tipo --model=$est.model_id, PowerShell expande $est (el objeto
    # completo) y deja ".model_id" como texto literal. Como argumento suelto si
    # funciona, pero pegado a un --flag= no.
    $modelId    = $est.model_id
    $endpointId = $est.endpoint_id

    Paso "Desplegando (tarda entre 15 y 30 minutos; no cierres la terminal)"
    Write-Host "   modelo   : $modelId"
    Write-Host "   endpoint : $endpointId"
    if ($modelId -notmatch '^\d+$') {
        Falla "El model_id no es numerico: '$modelId'. Revisa .endpoint_finetuned.json"
        exit 1
    }
    Aviso "Desde aqui la L4 cobra por hora hasta que corras: cleanup"
    gcloud ai endpoints deploy-model $endpointId `
        --project=$PROJECT --region=$REGION `
        --model=$modelId `
        --display-name=$MODEL_NAME `
        --machine-type=$MACHINE_TYPE `
        --accelerator="type=$ACCELERATOR_TYPE,count=$ACCELERATOR_COUNT" `
        --min-replica-count=1 --max-replica-count=1
    # Nota: el registro del contenedor viene ENCENDIDO por defecto en gcloud.
    # La bandera que existe es --disable-container-logging, para apagarlo. No
    # hay --enable-container-logging.
    if ($LASTEXITCODE -ne 0) {
        Falla "Fallo el despliegue."
        Falla "Para ver por que murio el contenedor de vLLM, corre 06-...ps1 logs"
        exit 1
    }

    $deployedId = gcloud ai endpoints describe $endpointId `
        --project=$PROJECT --region=$REGION --format="value(deployedModels[0].id)"
    $est | Add-Member -NotePropertyName deployed_model_id -NotePropertyValue $deployedId -Force
    Guardar-Estado $est
    Ok "desplegado. deployed_model_id=$deployedId"

    Write-Host "`nSiguiente paso, generar las respuestas del modelo afinado:" -ForegroundColor Cyan
    Write-Host "  python .\si7016-asilvaz1-taller3-262\src\deploy\03-predict-endpoint.py ``"
    Write-Host "      --dataset data\qa\normas_ruido_eval.jsonl ``"
    Write-Host "      --system finetuning --endpoint_file .endpoint_finetuned.json ``"
    Write-Host "      --technique zero-shot"
}

# --- predict (prueba rapida) --------------------------------------------------
function Cmd-Predict {
    $est = Leer-Estado
    if (-not $est -or -not $est.endpoint_id) { Falla "Corre primero: deploy"; exit 1 }

    Paso "Prediccion de prueba"
    $prompt = "<start_of_turn>user`nYou are an expert on ISO, UNE and BS acoustics standards for noise measurement. Answer the question precisely and cite the clause when you know it.`n`nQuestion: What is the scope of ISO 1996-1?<end_of_turn>`n<start_of_turn>model`n"
    $cuerpo = @{ instances = @(@{
        prompt      = $prompt
        max_tokens  = 128
        temperature = 0.0
        top_p       = 1.0
        top_k       = -1
    }) } | ConvertTo-Json -Depth 5

    $tmp = [System.IO.Path]::GetTempFileName()
    Set-Content -Path $tmp -Value $cuerpo -Encoding UTF8
    gcloud ai endpoints predict $est.endpoint_id `
        --project=$PROJECT --region=$REGION --json-request=$tmp
    Remove-Item $tmp -Force
}

# --- status -------------------------------------------------------------------
function Cmd-Status {
    Paso "Estado"
    $est = Leer-Estado
    if ($est) { $est | ConvertTo-Json -Depth 5 | Write-Host }
    else { Aviso "todavia no hay .endpoint_finetuned.json" }

    Paso "Lo que esta cobrando ahora mismo en este proyecto"
    gcloud ai endpoints list --project=$PROJECT --region=$REGION `
        --format="table(displayName,name.basename(),deployedModels[0].id)"
}

# --- logs ---------------------------------------------------------------------
# El filtro lleva comillas dobles dentro de comillas simples, que en PowerShell
# a mano es una fuente segura de errores. Aqui se arma una sola vez y bien.
function Cmd-Logs {
    $est = Leer-Estado
    if (-not $est -or -not $est.endpoint_id) { Falla "No hay endpoint todavia."; exit 1 }
    $endpointId = $est.endpoint_id

    $filtro = 'resource.type="aiplatform.googleapis.com/Endpoint" AND resource.labels.endpoint_id="' + $endpointId + '"'
    Paso "Logs del contenedor del endpoint $endpointId"
    gcloud logging read $filtro --project=$PROJECT --freshness=2h --limit=100 `
        --order=asc --format="value(textPayload)"
}

# --- cleanup ------------------------------------------------------------------
function Cmd-Cleanup {
    $est = Leer-Estado
    if (-not $est -or -not $est.endpoint_id) { Aviso "No hay endpoint que limpiar."; return }

    # Mismas variables sueltas que en Cmd-Deploy, por la misma razon.
    $endpointId = $est.endpoint_id
    $modelId    = $est.model_id
    $deployedId = $est.deployed_model_id

    if ($deployedId) {
        Paso "Retirando el modelo del endpoint"
        gcloud ai endpoints undeploy-model $endpointId `
            --project=$PROJECT --region=$REGION `
            --deployed-model-id=$deployedId --quiet
    }
    Paso "Borrando el endpoint"
    gcloud ai endpoints delete $endpointId --project=$PROJECT --region=$REGION --quiet
    if ($modelId) {
        Paso "Borrando el modelo del registro"
        gcloud ai models delete $modelId --project=$PROJECT --region=$REGION --quiet
    }
    Remove-Item $ESTADO -Force -ErrorAction SilentlyContinue
    Ok "limpieza completa"
    Aviso "El modelo fusionado sigue en $MERGED_DIR y ocupa ~17 GB de almacenamiento."
    Aviso "Borralo con: gcloud storage rm -r $MERGED_DIR   (solo cuando ya no lo necesites)"
}

switch ($Comando) {
    "upload"          { Cmd-Upload }
    "create-endpoint" { Cmd-CreateEndpoint }
    "deploy"          { Cmd-Deploy }
    "predict"         { Cmd-Predict }
    "status"          { Cmd-Status }
    "logs"            { Cmd-Logs }
    "cleanup"         { Cmd-Cleanup }
}
