# 00-build-image.ps1 - Construye y sube la imagen de entrenamiento del taller 3.
#
# COMO EJECUTARLO (desde la carpeta 13_nlp):
#   powershell -ExecutionPolicy Bypass -File .\si7016-asilvaz1-taller3-262\src\finetuning\00-build-image.ps1
#
# Que hace, en orden:
#   1. Copia el dataset desde data/qa/ a esta carpeta, porque Docker solo puede
#      COPY archivos que esten dentro del contexto de build.
#   2. Lanza gcloud builds submit, que sube el contexto a Cloud Build, arma la
#      imagen alla y la deja en Artifact Registry.
#   3. Borra las copias del dataset para que no queden sueltas en el repo.
#
# Hay que volver a correrlo cada vez que se cambie train_gemma.py o el dataset.
# Tarda entre 5 y 12 minutos la primera vez (la imagen base pesa varios GB);
# las siguientes son mas rapidas por la cache de Cloud Build.
#
# Cloud Build habla HTTPS, asi que esto funciona aunque la red del campus
# bloquee gRPC.

$PSNativeCommandUseErrorActionPreference = $false
$ErrorActionPreference = "Continue"

# --- Parametros del proyecto (los mismos de src/deploy) ----------------------
$PROJECT = "si7016-262-nlp"
$REGION  = "us-central1"
$AR_REPO = "si7016-taller3"
$TAG     = "gemma-7b-it-normas-ruido:latest"
$IMAGE   = "$REGION-docker.pkg.dev/$PROJECT/$AR_REPO/$TAG"

function Paso($texto) { Write-Host "`n== $texto ==" -ForegroundColor Cyan }
function Ok($texto)   { Write-Host "   $texto" -ForegroundColor Green }
function Error2($t)   { Write-Host "   $t" -ForegroundColor Red }

$AQUI = Split-Path -Parent $MyInvocation.MyCommand.Path
$REPO = Split-Path -Parent (Split-Path -Parent $AQUI)
$QA   = Join-Path $REPO "data\qa"

# --- 1. Preparar el contexto de build ----------------------------------------
Paso "Copiando el dataset al contexto de build"
$archivos = @("normas_ruido_train.jsonl", "normas_ruido_eval.jsonl")
foreach ($a in $archivos) {
    $origen = Join-Path $QA $a
    if (-not (Test-Path $origen)) {
        Error2 "No encuentro $origen. Revisa que data/qa/ este completo."
        exit 1
    }
    Copy-Item $origen (Join-Path $AQUI $a) -Force
    $n = (Get-Content $origen | Measure-Object -Line).Lines
    Ok "$a  ($n lineas)"
}

# --- 1b. Salvaguarda: el .gcloudignore tiene que existir ---------------------
# Sin el, gcloud arma un .gcloudignore a partir del .gitignore de esta carpeta,
# que excluye los .jsonl, y el COPY del Dockerfile falla con
# "file not found in build context". Ya paso una vez.
Paso "Comprobando el .gcloudignore"
$gi = Join-Path $AQUI ".gcloudignore"
if (-not (Test-Path $gi)) {
    Error2 "Falta .gcloudignore en src/finetuning."
    Error2 "Sin el, gcloud usa el .gitignore y deja el dataset fuera del build."
    foreach ($a in $archivos) {
        $copia = Join-Path $AQUI $a
        if (Test-Path $copia) { Remove-Item $copia -Force }
    }
    exit 1
}
# Solo las reglas reales: las lineas en blanco y los comentarios (#) no
# excluyen nada, y este archivo menciona "jsonl" en sus comentarios.
$reglas = Get-Content $gi | Where-Object {
    $_.Trim() -ne "" -and -not $_.TrimStart().StartsWith("#")
}
if ($reglas -match "jsonl") {
    Error2 "El .gcloudignore tiene una REGLA que excluye archivos .jsonl:"
    ($reglas -match "jsonl") | ForEach-Object { Error2 "    $_" }
    Error2 "Quitala: el dataset tiene que viajar en el contexto de build."
    foreach ($a in $archivos) {
        $copia = Join-Path $AQUI $a
        if (Test-Path $copia) { Remove-Item $copia -Force }
    }
    exit 1
}
Ok "presente y sin excluir el dataset"

# --- 2. Construir y subir -----------------------------------------------------
Paso "Cloud Build -> $IMAGE"
Write-Host "   (esto sube el contexto y construye en GCP, no en tu maquina)"
gcloud builds submit $AQUI --tag $IMAGE --project $PROJECT
$codigo = $LASTEXITCODE

# --- 3. Limpiar el contexto ---------------------------------------------------
Paso "Limpiando las copias del dataset"
foreach ($a in $archivos) {
    $copia = Join-Path $AQUI $a
    if (Test-Path $copia) { Remove-Item $copia -Force; Ok "borrado $a" }
}

if ($codigo -ne 0) {
    Error2 "Cloud Build fallo (codigo $codigo). Revisa el log que imprimio arriba."
    exit $codigo
}

Paso "Listo"
Ok "Imagen: $IMAGE"
Write-Host "`nSiguiente paso, el smoke test de 20 pasos:" -ForegroundColor Cyan
Write-Host "  python .\si7016-asilvaz1-taller3-262\src\finetuning\submit_vertex_job.py --smoke_test"
