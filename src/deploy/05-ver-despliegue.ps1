# 05-ver-despliegue.ps1 - Sigue el avance del despliegue sin tocar la terminal
# donde esta corriendo el script 02.
#
# ABRE UNA SEGUNDA VENTANA de PowerShell para esto. No uses la misma donde
# esta el despliegue: si la interrumpes, pierdes la referencia al endpoint.
#
# Ejecutar:
#   powershell -ExecutionPolicy Bypass -File .\si7016-asilvaz1-taller3-262\src\deploy\05-ver-despliegue.ps1
#   ...\05-ver-despliegue.ps1 -Segundos 30      # revisar mas seguido
#   ...\05-ver-despliegue.ps1 -UnaVez           # una sola foto y salir

param(
    [int]$Segundos = 60,
    [switch]$UnaVez
)

$PSNativeCommandUseErrorActionPreference = $false
$ErrorActionPreference = "Continue"

$PROJECT = "si7016-262-nlp"
$REGION  = "us-central1"

# gcloud usa HTTPS normal, asi que esto funciona aunque gRPC este bloqueado.
function Foto {
    $hora = Get-Date -Format "HH:mm:ss"
    Write-Host "`n[$hora] ===============================================" -ForegroundColor Cyan

    Write-Host "Endpoints:" -ForegroundColor Cyan
    $endpoints = gcloud ai endpoints list --region=$REGION --project=$PROJECT `
        --format="value(name.basename(),displayName)" 2>$null
    if (-not $endpoints) {
        Write-Host "   Todavia no aparece ninguno. El endpoint se crea en los" -ForegroundColor Yellow
        Write-Host "   primeros minutos; el modelo tarda bastante mas en montarse."
        return
    }

    foreach ($linea in $endpoints) {
        $campos = $linea -split "`t"
        $id = $campos[0]
        $nombre = if ($campos.Count -gt 1) { $campos[1] } else { "" }
        Write-Host "   $id  $nombre" -ForegroundColor Green

        # deployedModels vacio = el endpoint existe pero el modelo sigue montandose
        $modelos = gcloud ai endpoints describe $id --region=$REGION --project=$PROJECT `
            --format="value(deployedModels[].id)" 2>$null
        if ($modelos) {
            Write-Host "      LISTO y COBRANDO. deployed_model_id: $modelos" -ForegroundColor Magenta
        } else {
            Write-Host "      Desplegando todavia (deployedModels aun vacio)..." -ForegroundColor Yellow
        }
    }

    Write-Host "Modelos en el registro:" -ForegroundColor Cyan
    gcloud ai models list --region=$REGION --project=$PROJECT `
        --format="table[no-heading](name.basename(),displayName)" 2>$null
}

Write-Host "Consola web, por si prefieres mirarlo ahi:" -ForegroundColor Cyan
Write-Host "  Endpoints: https://console.cloud.google.com/vertex-ai/online-prediction/endpoints?project=$PROJECT"
Write-Host "  Modelos:   https://console.cloud.google.com/vertex-ai/models?project=$PROJECT"

if ($UnaVez) { Foto; exit 0 }

Write-Host "`nRevisando cada $Segundos segundos. Ctrl+C para salir (no afecta el despliegue)." -ForegroundColor Yellow
while ($true) {
    Foto
    Start-Sleep -Seconds $Segundos
}
