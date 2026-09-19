# 00b-diagnostico-red.ps1 - Diagnostica el error
#   "UNAVAILABLE ... getaddrinfo: WSA Error ... 11001"
# que lanza gRPC cuando no logra resolver el host de Vertex AI.
#
# Ejecutar:
#   powershell -ExecutionPolicy Bypass -File .\si7016-asilvaz1-taller3-262\src\deploy\00b-diagnostico-red.ps1

$PSNativeCommandUseErrorActionPreference = $false
$ErrorActionPreference = "Continue"

$REGION = "us-central1"
$HOSTS = @(
    "$REGION-aiplatform.googleapis.com",
    "aiplatform.googleapis.com",
    "storage.googleapis.com",
    "oauth2.googleapis.com"
)

function Titulo($t) { Write-Host "`n== $t ==" -ForegroundColor Cyan }

Titulo "1. Resolucion DNS"
foreach ($h in $HOSTS) {
    $r = Resolve-DnsName -Name $h -Type A -ErrorAction SilentlyContinue
    if ($r) {
        $ips = ($r | Where-Object { $_.IPAddress } | Select-Object -ExpandProperty IPAddress) -join ", "
        Write-Host ("   OK     {0,-42} -> {1}" -f $h, $ips) -ForegroundColor Green
    } else {
        Write-Host ("   FALLA  {0,-42} -> no resuelve" -f $h) -ForegroundColor Red
    }
}

Titulo "2. Conexion TCP al puerto 443"
foreach ($h in $HOSTS) {
    $t = Test-NetConnection -ComputerName $h -Port 443 -WarningAction SilentlyContinue
    if ($t.TcpTestSucceeded) {
        Write-Host ("   OK     {0,-42} -> {1}" -f $h, $t.RemoteAddress) -ForegroundColor Green
    } else {
        Write-Host ("   FALLA  {0,-42} -> sin conexion" -f $h) -ForegroundColor Red
    }
}

Titulo "3. Servidores DNS que esta usando este equipo"
Get-DnsClientServerAddress -AddressFamily IPv4 |
    Where-Object { $_.ServerAddresses } |
    Format-Table InterfaceAlias, ServerAddresses -AutoSize

Titulo "4. Adaptadores de red activos (una VPN aparece aqui)"
Get-NetAdapter | Where-Object Status -eq "Up" |
    Format-Table Name, InterfaceDescription, LinkSpeed -AutoSize

Titulo "5. Proxy configurado en el sistema"
$proxy = Get-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings" -ErrorAction SilentlyContinue
if ($proxy.ProxyEnable -eq 1) {
    Write-Host "   Proxy ACTIVO: $($proxy.ProxyServer)" -ForegroundColor Yellow
    Write-Host "   Si es un proxy corporativo, gRPC necesita las variables"
    Write-Host "   HTTPS_PROXY y NO_PROXY para funcionar."
} else {
    Write-Host "   Sin proxy configurado." -ForegroundColor Green
}
Write-Host "   Variables de entorno de proxy en esta sesion:"
foreach ($v in @("HTTP_PROXY","HTTPS_PROXY","NO_PROXY","http_proxy","https_proxy","no_proxy")) {
    $val = [Environment]::GetEnvironmentVariable($v)
    if ($val) { Write-Host "     $v = $val" -ForegroundColor Yellow }
}

Titulo "6. Prueba real contra la API por REST (HTTPS normal, como gcloud)"
python -c @"
import os
os.environ.setdefault('GRPC_DNS_RESOLVER','native')
import vertexai
from vertexai import model_garden
vertexai.init(project='si7016-262-nlp', location='$REGION', api_transport='rest')
m = model_garden.list_deployable_models(model_filter='gemma')
print('   OK REST: la API respondio. Modelos encontrados:', len(m))
"@
$restOk = ($LASTEXITCODE -eq 0)

Titulo "7. Prueba real contra la API por gRPC"
$env:GRPC_DNS_RESOLVER = "native"
python -c @"
import os
os.environ.setdefault('GRPC_DNS_RESOLVER','native')
import vertexai
from vertexai import model_garden
vertexai.init(project='si7016-262-nlp', location='$REGION', api_transport='grpc')
m = model_garden.list_deployable_models(model_filter='gemma')
print('   OK gRPC: la API respondio. Modelos encontrados:', len(m))
"@
$grpcOk = ($LASTEXITCODE -eq 0)

Titulo "Conclusion"
if ($restOk) {
    Write-Host "   REST funciona. Los scripts 01 a 04 ya usan REST por defecto." -ForegroundColor Green
    if (-not $grpcOk) {
        Write-Host "   gRPC esta bloqueado, pero no importa: no lo vas a usar." -ForegroundColor Yellow
    }
    Write-Host "`n   Vuelve a correr el paso 3 tal cual." -ForegroundColor Green
} elseif ($grpcOk) {
    Write-Host "   Caso raro: gRPC si y REST no. Pasa --transport grpc a los scripts." -ForegroundColor Yellow
} else {
    Write-Host "   Ninguno de los dos transportes llega a la API." -ForegroundColor Red
    Write-Host "   - Si el punto 5 mostro un proxy, exporta las variables y reintenta:"
    Write-Host '       $env:HTTPS_PROXY = "http://usuario:clave@servidor:puerto"'
    Write-Host '       $env:HTTP_PROXY  = $env:HTTPS_PROXY'
    Write-Host "   - Si estas en una red institucional o con VPN, prueba desde otra red"
    Write-Host "     (por ejemplo el celular en modo modem) para confirmar que es la red."
    Write-Host "   - Alternativa que no depende de tu red: Cloud Shell en la consola de GCP."
    Write-Host "       https://console.cloud.google.com/?cloudshell=true&project=si7016-262-nlp"
}
