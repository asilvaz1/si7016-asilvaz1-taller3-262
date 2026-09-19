# 00-setup-gcp.ps1 - Preparacion unica del proyecto GCP para el taller 3.
#
# COMO EJECUTARLO (desde la carpeta 13_nlp):
#   powershell -ExecutionPolicy Bypass -File .\si7016-asilvaz1-taller3-262\src\deploy\00-setup-gcp.ps1
#
# Es idempotente: si el bucket o el repositorio ya existen, lo dice y sigue.
# Se puede volver a correr cuantas veces haga falta.

# Los comandos de gcloud escriben avisos en stderr de forma rutinaria.
# En PowerShell 7.3+ eso aborta el script si no se desactiva esto primero.
$PSNativeCommandUseErrorActionPreference = $false
$ErrorActionPreference = "Continue"

# --- Parametros del proyecto -------------------------------------------------
$PROJECT  = "si7016-262-nlp"
$REGION   = "us-central1"        # cuota L4 aprobada; alterna: us-west1
$BUCKET   = "asilvaz1taller3"    # sin gs://, debe ser REGIONAL y en $REGION
$AR_REPO  = "si7016-taller3"     # repositorio de imagenes Docker

function Paso($texto) { Write-Host "`n== $texto ==" -ForegroundColor Cyan }
function Ok($texto)   { Write-Host "   $texto" -ForegroundColor Green }
function Aviso($texto){ Write-Host "   $texto" -ForegroundColor Yellow }

# --- 0. gcloud esta instalado? -----------------------------------------------
Paso "Comprobando gcloud"
$gcloud = Get-Command gcloud -ErrorAction SilentlyContinue
if (-not $gcloud) {
    Write-Host "gcloud no esta en el PATH de esta terminal." -ForegroundColor Red
    Write-Host "Abre una terminal nueva despues de instalarlo, o usa la ruta completa." -ForegroundColor Red
    exit 1
}
gcloud version | Select-Object -First 1

# --- 1. Autenticacion --------------------------------------------------------
# gcloud auth login                     -> credenciales para el CLI
# gcloud auth application-default login -> credenciales para las librerias Python
Paso "Autenticacion (se abriran dos ventanas del navegador)"
gcloud auth login
gcloud auth application-default login
gcloud config set project $PROJECT
gcloud config set ai/region $REGION
Ok "Proyecto activo: $PROJECT / region: $REGION"

# --- 2. APIs -----------------------------------------------------------------
Paso "Habilitando APIs (puede tardar un par de minutos)"
gcloud services enable `
    aiplatform.googleapis.com `
    compute.googleapis.com `
    cloudbuild.googleapis.com `
    artifactregistry.googleapis.com `
    storage.googleapis.com `
    iam.googleapis.com `
    --project=$PROJECT
if ($LASTEXITCODE -eq 0) { Ok "APIs habilitadas." } else { Aviso "Revisa el error de arriba." }

# --- 3. Bucket de GCS --------------------------------------------------------
# Regional y en la misma region del entrenamiento: si el bucket es multi-region
# o esta en otra region, el Custom Training Job falla o cobra egress.
Paso "Bucket gs://$BUCKET"
gcloud storage buckets describe "gs://$BUCKET" --project=$PROJECT --format="value(name)" *> $null
if ($LASTEXITCODE -eq 0) {
    Aviso "Ya existe. Ubicacion actual:"
    gcloud storage buckets describe "gs://$BUCKET" --project=$PROJECT --format="value(location,locationType)"
} else {
    gcloud storage buckets create "gs://$BUCKET" `
        --project=$PROJECT `
        --location=$REGION `
        --uniform-bucket-level-access
    if ($LASTEXITCODE -eq 0) { Ok "Bucket creado." } else { Aviso "No se pudo crear el bucket." }
}

# Carpetas logicas del pipeline. GCS no tiene carpetas reales: se crean al
# escribir el primer objeto, esto solo deja el arbol visible en la consola.
Paso "Creando el arbol de carpetas en el bucket"
$keep = Join-Path $env:TEMP "keep.txt"
"placeholder" | Out-File -Encoding ascii -FilePath $keep
foreach ($carpeta in @("gemma-7b-it-normas-lora", "gemma-7b-it-normas-merged", "datasets")) {
    gcloud storage cp $keep "gs://$BUCKET/$carpeta/.keep" --project=$PROJECT *> $null
    if ($LASTEXITCODE -eq 0) { Ok "gs://$BUCKET/$carpeta/" } else { Aviso "fallo: $carpeta" }
}
Remove-Item $keep -ErrorAction SilentlyContinue

# --- 4. Artifact Registry ----------------------------------------------------
Paso "Artifact Registry: $AR_REPO"
gcloud artifacts repositories describe $AR_REPO --location=$REGION --project=$PROJECT --format="value(name)" *> $null
if ($LASTEXITCODE -eq 0) {
    Aviso "Ya existe."
} else {
    gcloud artifacts repositories create $AR_REPO `
        --repository-format=docker `
        --location=$REGION `
        --project=$PROJECT `
        --description="Imagenes de fine-tuning taller 3 SI7016"
    if ($LASTEXITCODE -eq 0) { Ok "Repositorio creado." } else { Aviso "No se pudo crear." }
}

# --- 5. Verificacion ---------------------------------------------------------
Paso "Verificacion"
Write-Host "APIs habilitadas:"
gcloud services list --enabled --project=$PROJECT `
    --filter="config.name:(aiplatform OR compute OR cloudbuild OR artifactregistry OR storage)" `
    --format="table(config.name)"

Write-Host "`nBucket:"
gcloud storage buckets describe "gs://$BUCKET" --project=$PROJECT `
    --format="table(name,location,locationType,storageClass)"

Write-Host "`nRepositorio de imagenes:"
gcloud artifacts repositories list --location=$REGION --project=$PROJECT `
    --format="table(name,format,createTime)"

Write-Host "`nEndpoints desplegados ahora mismo (si hay alguno, esta cobrando):"
gcloud ai endpoints list --region=$REGION --project=$PROJECT --format="table(name,displayName)"

Write-Host "`n== Listo ==" -ForegroundColor Green
Write-Host "Siguiente paso:" -ForegroundColor Cyan
Write-Host "  python .\si7016-asilvaz1-taller3-262\src\deploy\01-list-gemma-model-garden.py"
Write-Host "`nLa cuota de GPU L4 no se ve bien por gcloud; revisala en la consola:" -ForegroundColor Yellow
Write-Host "  https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT"
Write-Host "  Servicio: Vertex AI API | Metrica: custom_model_training_nvidia_l4_gpus | Region: $REGION"
