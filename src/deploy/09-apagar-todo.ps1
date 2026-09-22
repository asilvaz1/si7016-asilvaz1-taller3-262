# 09-apagar-todo.ps1 - Apaga todo lo que cobra por hora y reporta lo que queda.
#
# Se corre al terminar cada sesion de trabajo y, sobre todo, al cerrar el taller.
# Hace dos cosas distintas y conviene no confundirlas:
#
#   - APAGA lo que cobra por tiempo encendido: la VM con la L4, los endpoints
#     de Vertex desplegados y la regla de firewall de IAP.
#   - REPORTA lo que cobra por existir (almacenamiento, corpus vectorial) sin
#     borrarlo, porque borrarlo es una decision de la entrega y no de fin de dia.
#
# USO
#   powershell -ExecutionPolicy Bypass -File .\09-apagar-todo.ps1 estado    # solo mira
#   powershell -ExecutionPolicy Bypass -File .\09-apagar-todo.ps1 apagar    # apaga
#   powershell -ExecutionPolicy Bypass -File .\09-apagar-todo.ps1 borrar-corpus
#
# 'estado' no toca nada: uselo primero.

param(
    [Parameter(Position = 0)]
    [ValidateSet("estado", "apagar", "borrar-corpus")]
    [string]$Comando = "estado",

    [string]$Vm     = "si7016-262-asilvaz1",
    [string]$Zona   = "us-west1-a",
    [string]$Proyecto = "si7016-262-nlp"
)

$ErrorActionPreference = "Continue"
$AQUI = Split-Path -Parent $MyInvocation.MyCommand.Path

function Paso($t)  { Write-Host "`n== $t ==" -ForegroundColor Cyan }
function Ok($t)    { Write-Host "   $t" -ForegroundColor Green }
function Aviso($t) { Write-Host "   $t" -ForegroundColor Yellow }

# --- 1. La VM: el unico riesgo real de costo -------------------------------
function Vm-Estado {
    Paso "VM $Vm ($Zona)"
    $estado = gcloud compute instances describe $Vm --zone=$Zona --project=$Proyecto `
        --format="value(status)" 2>$null
    if (-not $estado) { Aviso "No existe o no hay acceso"; return $null }
    if ($estado -eq "RUNNING") {
        Write-Host "   ENCENDIDA - esta cobrando" -ForegroundColor Red
    } else {
        Ok "$estado - no cobra computo"
    }
    return $estado
}

function Vm-Apagar {
    if ((Vm-Estado) -eq "RUNNING") {
        Paso "Apagando la VM"
        # 'stop' y no 'delete': el disco con vLLM instalado y el modelo de 17 GB
        # se conserva, asi que una demostracion posterior arranca en 2 minutos
        # en vez de repetir install y sync. El disco parado cuesta centavos.
        gcloud compute instances stop $Vm --zone=$Zona --project=$Proyecto
        Ok "Detenida. El disco se conserva; volver a encenderla no repite install ni sync"
    }
}

# --- 2. Endpoints de Vertex -------------------------------------------------
function Endpoints-Estado {
    Paso "Endpoints de Vertex con modelos desplegados"
    $hay = $false
    foreach ($region in @("us-central1", "us-west1")) {
        $lista = gcloud ai endpoints list --project=$Proyecto --region=$region `
            --format="value(name.basename(),displayName)" 2>$null
        foreach ($linea in $lista) {
            if (-not $linea) { continue }
            $id = ($linea -split "\s+")[0]
            $desplegados = gcloud ai endpoints describe $id --project=$Proyecto `
                --region=$region --format="value(deployedModels[].id)" 2>$null
            if ($desplegados) {
                Write-Host "   $region/$id tiene modelos desplegados - COBRANDO" -ForegroundColor Red
                $hay = $true
            } else {
                Ok "$region/$id existe pero sin modelo desplegado (no cobra)"
            }
        }
    }
    if (-not $hay) { Ok "Ningun modelo desplegado" }
    return $hay
}

function Endpoints-Apagar {
    if (Endpoints-Estado) {
        Paso "Retirando modelos de los endpoints"
        # El script de limpieza de la fase 0 ya sabe recorrer las regiones y
        # hacer undeploy; aqui solo se invoca para no duplicar esa logica.
        python (Join-Path $AQUI "04-undeploy-cleanup.py") --all
    }
}

# --- 3. Regla de firewall del tunel IAP -------------------------------------
function Firewall-Apagar {
    Paso "Regla de firewall del tunel IAP"
    $regla = gcloud compute firewall-rules describe allow-iap-tunnel `
        --project=$Proyecto --format="value(name)" 2>$null
    if ($regla) {
        gcloud compute firewall-rules delete allow-iap-tunnel --project=$Proyecto --quiet
        Ok "Regla borrada"
    } else {
        Ok "No existe (nada que borrar)"
    }
}

# --- 4. Lo que sigue costando por existir -----------------------------------
function Reportar-Persistente {
    Paso "Lo que sigue costando (almacenamiento, no se apaga)"
    gcloud storage du -s gs://asilvaz1taller3 --readable-sizes 2>$null
    gcloud storage du -s gs://asilvaz1taller3-west --readable-sizes 2>$null
    Aviso "El modelo fusionado son ~17 GB, que es casi todo el gasto de almacenamiento."
    Aviso "CONSERVAR gs://asilvaz1taller3/normas-ruido-lora/ (~200 MB): son los"
    Aviso "adaptadores entrenados, el resultado real del fine-tuning."

    $estadoRag = Join-Path (Split-Path -Parent $AQUI) "rag\.rag_corpus.json"
    if (Test-Path $estadoRag) {
        $rag = Get-Content $estadoRag -Raw | ConvertFrom-Json
        Aviso "Corpus de RAG Engine vivo: $($rag.display_name) en $($rag.location)"
        Aviso "Cobra almacenamiento vectorial mientras exista. Para borrarlo:"
        Aviso "  .\09-apagar-todo.ps1 borrar-corpus"
    }
}

function Borrar-Corpus {
    Paso "Borrando el corpus de RAG Engine"
    Aviso "Esto es IRREVERSIBLE y deja de funcionar la pestana de RAG de la app."
    Aviso "Hazlo solo cuando ya tengas las capturas y el jsonl de resultados."
    $r = Read-Host "Escribe BORRAR para confirmar"
    if ($r -ne "BORRAR") { Aviso "Cancelado"; return }
    python (Join-Path (Split-Path -Parent $AQUI) "rag\vertex_rag_engine.py") delete
}

# --- Despacho ---------------------------------------------------------------
switch ($Comando) {
    "estado" {
        Vm-Estado | Out-Null
        Endpoints-Estado | Out-Null
        Reportar-Persistente
        Write-Host "`nNada se modifico. Para apagar: .\09-apagar-todo.ps1 apagar" -ForegroundColor Cyan
    }
    "apagar" {
        Vm-Apagar
        Endpoints-Apagar
        Firewall-Apagar
        Reportar-Persistente
        Write-Host "`nTodo lo que cobra por hora esta apagado." -ForegroundColor Green
    }
    "borrar-corpus" { Borrar-Corpus }
}
