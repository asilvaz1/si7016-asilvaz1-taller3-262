#!/usr/bin/env bash
# 07-vm-vllm.sh - Servir el modelo AFINADO con vLLM en una VM de Compute Engine.
#
# Este script NO se corre desde Windows: se copia a la VM y se corre alli.
# Cubre el requisito del enunciado "correr el modelo en la VM usando vLLM",
# que es la ruta alterna al endpoint de Vertex (ver 06-deploy-finetuned.ps1).
#
# Por que una VM y no el endpoint de Vertex: en la VM se ven los logs de vLLM
# en vivo, se pueden cambiar los flags y reintentar en segundos, y no hay
# health check ni reloj de despliegue que declare fallido un arranque sano.
# La VM usa la cuota NVIDIA_L4_GPUS de Compute Engine, distinta de la de
# Vertex serving.
#
# COMO USARLO
#   # desde tu maquina, una sola vez:
#   gcloud compute scp src/deploy/07-vm-vllm.sh VM:07-vm-vllm.sh --zone=ZONA
#   # (pscp en Windows no expande '~': un destino VM:~/ falla)
#   gcloud compute ssh VM --zone=ZONA
#
#   # ya dentro de la VM:
#   chmod +x 07-vm-vllm.sh
#   ./07-vm-vllm.sh check      # driver, GPU, disco: aborta temprano si algo falta
#   ./07-vm-vllm.sh install    # venv + vLLM (10-15 min, una sola vez)
#   ./07-vm-vllm.sh sync       # baja el modelo fusionado de GCS (~17 GB, 5-10 min)
#   ./07-vm-vllm.sh serve      # arranca vLLM en background, log en ~/vllm.log
#   ./07-vm-vllm.sh status     # espera a que /health responda
#   ./07-vm-vllm.sh test       # una pregunta de humo
#   ./07-vm-vllm.sh stop       # baja el server (NO apaga la VM)
#
# ATENCION AL COSTO: la VM con L4 cobra por hora mientras este ENCENDIDA,
# corra o no vLLM. 'stop' solo mata el proceso. Para dejar de pagar:
#   gcloud compute instances stop VM --zone=ZONA
set -euo pipefail

# --- Parametros -------------------------------------------------------------
BUCKET="${BUCKET:-asilvaz1taller3}"
MERGED_URI="${MERGED_URI:-gs://$BUCKET/normas-ruido-merged}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/normas-ruido-merged}"
VENV="${VENV:-$HOME/vllm-venv}"
PUERTO="${PUERTO:-8000}"
SERVED_NAME="${SERVED_NAME:-gemma-7b-it-normas-ruido}"
LOG="$HOME/vllm.log"
PIDFILE="$HOME/vllm.pid"

# Flags de memoria. Son los mismos que corrigen el despliegue de Vertex, y la
# razon es aritmetica, no capricho:
#   L4, memoria real que reporta nvidia-smi   22.5 GiB (23034 MiB, no 24576)
#   presupuesto con --gpu-memory-utilization  20.7 GiB (0.92)
#   pesos de gemma-7b-it en bf16              17.1 GiB (8.54 B parametros)
#   queda para cache KV + activaciones         3.6 GiB
# Con --enforce-eager (sin grafos CUDA) y --max-model-len=1024 la cache KV de 4
# secuencias pesa ~1.8 GiB (0.44 MB por token en Gemma 7B), asi que entra, pero
# sin holgura. Si aborta por memoria, los dos diales son GPU_UTIL=0.95 primero y
# MAX_LEN=768 MAX_SEQS=2 despues.
# --enforce-eager evita que vLLM capture grafos CUDA al arrancar, que por si
# solos pesan 1 a 2 GB. --max-model-len 1024 sobra: el prompt mas largo es el
# de few-shot (~250 tokens) mas 256 de generacion.
GPU_UTIL="${GPU_UTIL:-0.92}"
MAX_LEN="${MAX_LEN:-1024}"
MAX_SEQS="${MAX_SEQS:-4}"

azul()  { printf '\n\033[36m== %s ==\033[0m\n' "$1"; }
ok()    { printf '   \033[32m%s\033[0m\n' "$1"; }
aviso() { printf '   \033[33m%s\033[0m\n' "$1"; }
falla() { printf '   \033[31m%s\033[0m\n' "$1"; }

cmd_check() {
  azul "Driver y GPU"
  if ! command -v nvidia-smi >/dev/null; then
    falla "No hay nvidia-smi. La VM no tiene driver de NVIDIA instalado."
    aviso "En una imagen 'Deep Learning VM' el driver ya viene. En una imagen"
    aviso "limpia de Debian/Ubuntu hay que instalarlo antes de seguir."
    exit 1
  fi
  # nvidia-smi puede existir y aun asi fallar. El caso tipico es que el paquete
  # del driver se actualizo (unattended-upgrades) y las bibliotecas de espacio
  # de usuario ya son la version nueva, mientras que el modulo del kernel
  # cargado sigue siendo el viejo. No es un driver roto: es uno a medio
  # actualizar, y se arregla reiniciando para que cargue el modulo que
  # corresponde.
  if ! salida=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1); then
    falla "nvidia-smi existe pero fallo:"
    echo "$salida" | sed 's/^/     /'
    if echo "$salida" | grep -qi "version mismatch"; then
      aviso "Driver a medio actualizar. Modulo cargado contra bibliotecas nuevas."
      aviso "  modulo del kernel : $(cat /proc/driver/nvidia/version 2>/dev/null | head -1 || echo 'no cargado')"
      aviso "  paquete instalado : $(modinfo nvidia 2>/dev/null | awk '/^version:/{print $2}' || echo '?')"
      aviso "Arreglo, en este orden:"
      aviso "  1. sudo reboot                     # resuelve casi siempre"
      aviso "  2. sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia"
      aviso "  3. sudo /opt/deeplearning/install-driver.sh   # solo en imagenes Deep Learning VM"
    fi
    exit 1
  fi
  echo "$salida"
  local vram
  vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  if [ "$vram" -lt 22000 ]; then
    falla "La GPU tiene ${vram} MiB. gemma-7b-it fusionado en bf16 pesa ~17.1 GB"
    falla "y no cabe con margen para la cache KV. Necesitas 24 GB (L4, A10G) o mas."
    exit 1
  fi
  ok "VRAM suficiente: ${vram} MiB"

  # El torch que trae vLLM viene compilado contra CUDA 12, que exige driver
  # 525 o mas nuevo. Con uno viejo, 'install' termina bien y es 'serve' el que
  # falla, media hora despues, con "CUDA driver version is insufficient".
  local driver mayor
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  mayor=${driver%%.*}
  if [ "${mayor:-0}" -lt 525 ]; then
    falla "Driver $driver. vLLM necesita 525 o superior (CUDA 12)."
    falla "Actualizalo antes de seguir:"
    falla "  sudo /opt/deeplearning/install-driver.sh     # imagenes Deep Learning"
    falla "  curl -O https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py \\"
    falla "    && sudo python3 install_gpu_driver.py      # imagenes limpias"
    exit 1
  fi
  ok "Driver $driver"

  azul "Disco"
  df -h "$HOME" | tail -1
  local libre_gb
  libre_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
  if [ "$libre_gb" -lt 40 ]; then
    falla "Quedan ${libre_gb} GB libres. El modelo pesa 17 GB y vLLM con sus"
    falla "dependencias (torch + CUDA) otros 10 a 15 GB. Amplia el disco:"
    falla "  gcloud compute disks resize DISCO --size=100GB --zone=ZONA"
    exit 1
  fi
  ok "Disco suficiente: ${libre_gb} GB libres"

  azul "Python y gcloud"
  python3 --version
  command -v gcloud >/dev/null && ok "gcloud presente" || aviso "sin gcloud: 'sync' no va a funcionar"
}

cmd_install() {
  azul "Requisitos de Python"
  # Las imagenes de Debian/Ubuntu de GCE traen python3 pero NO el modulo venv:
  # 'python3 -m venv' falla con "ensurepip is not available" y deja un
  # directorio a medias que hay que borrar antes de reintentar.
  if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
    ver=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    aviso "Falta el modulo venv. Instalando python${ver}-venv (pide sudo)"
    sudo apt-get update -qq
    sudo apt-get install -y "python${ver}-venv" python3-dev build-essential \
      || sudo apt-get install -y python3-venv python3-dev build-essential
    python3 -c "import ensurepip" >/dev/null 2>&1 || {
      falla "Sigue faltando venv. Instalalo a mano:"
      falla "  sudo apt-get install -y python${ver}-venv"
      exit 1
    }
  fi
  ok "python3 $(python3 -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])') con venv"

  azul "Entorno virtual en $VENV"
  # Un intento anterior fallido deja el directorio sin bin/pip: se descarta.
  if [ -d "$VENV" ] && [ ! -x "$VENV/bin/pip" ]; then
    aviso "Habia un entorno incompleto en $VENV; se rehace"
    rm -rf "$VENV"
  fi
  python3 -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  pip install --upgrade pip wheel setuptools
  azul "Instalando vLLM (trae su propio torch con CUDA; 10 a 15 min)"
  pip install vllm
  ok "vLLM $(python -c 'import vllm; print(vllm.__version__)')"
}

cmd_sync() {
  azul "Bajando el modelo fusionado de GCS"
  mkdir -p "$MODEL_DIR"
  # rsync y no cp: si se corta la descarga, el segundo intento no repite los
  # shards que ya llegaron completos.
  gcloud storage rsync -r "$MERGED_URI" "$MODEL_DIR"
  azul "Verificacion del contenido"
  ls -lh "$MODEL_DIR"
  for f in config.json; do
    [ -f "$MODEL_DIR/$f" ] || { falla "falta $f: la fusion quedo incompleta"; exit 1; }
  done
  ls "$MODEL_DIR" | grep -q '^tokenizer' || { falla "faltan los archivos del tokenizer"; exit 1; }
  ls "$MODEL_DIR" | grep -qE '\.safetensors$|\.bin$' || { falla "faltan los pesos"; exit 1; }
  ok "Modelo completo en $MODEL_DIR"
}

cmd_serve() {
  [ -d "$MODEL_DIR" ] || { falla "No existe $MODEL_DIR. Corre 'sync' primero."; exit 1; }
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    aviso "Ya hay un vLLM corriendo (pid $(cat "$PIDFILE")). Corre 'stop' si quieres reiniciarlo."
    exit 0
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"

  # Los flags de vLLM cambian entre versiones y 'vllm serve' aborta si recibe
  # uno que no conoce, sin arrancar. Ejemplos reales: el motor V1 elimino
  # --swap-space (ya no intercambia bloques a RAM del host) y reemplazo
  # --disable-log-requests por --enable-log-requests, que viene apagado.
  # En vez de fijar una lista y que se rompa en la proxima version, se
  # consulta la ayuda y solo se pasan los opcionales que existan.
  local ayuda
  ayuda=$(vllm serve --help 2>&1 || true)
  soporta() { grep -q -- "$1" <<< "$ayuda"; }

  local args=(
    "$MODEL_DIR"
    --served-model-name "$SERVED_NAME"
    --host 0.0.0.0 --port "$PUERTO"
    --dtype bfloat16
    --max-model-len "$MAX_LEN"
    --gpu-memory-utilization "$GPU_UTIL"
    --max-num-seqs "$MAX_SEQS"
  )
  # --enforce-eager es el unico opcional que de verdad importa: libera 1 a 2 GB
  # de grafos CUDA, y sin el la cuenta de memoria de la L4 no cierra.
  if soporta "--enforce-eager"; then
    args+=(--enforce-eager)
  else
    aviso "Esta version de vLLM no acepta --enforce-eager. Si falta memoria,"
    aviso "baja MAX_LEN y MAX_SEQS."
  fi
  soporta "--swap-space" && args+=(--swap-space 2)
  soporta "--disable-log-requests" && args+=(--disable-log-requests)

  azul "Arrancando vLLM $(vllm --version 2>/dev/null | tail -1)"
  echo "   modelo      : $MODEL_DIR"
  echo "   puerto      : $PUERTO"
  echo "   max-model-len: $MAX_LEN   gpu-util: $GPU_UTIL   max-num-seqs: $MAX_SEQS"
  echo "   flags       : ${args[*]:1}"
  nohup vllm serve "${args[@]}" > "$LOG" 2>&1 &
  echo $! > "$PIDFILE"
  ok "pid $(cat "$PIDFILE"), log en $LOG"
  aviso "El arranque tarda 2 a 5 min (carga 17 GB a la GPU). Sigue el log con:"
  aviso "  tail -f $LOG"
}

cmd_status() {
  azul "Esperando a que el server responda en :$PUERTO"
  for i in $(seq 1 60); do
    if curl -sf "http://localhost:$PUERTO/health" >/dev/null 2>&1; then
      ok "vLLM arriba despues de ~$((i * 10)) s"
      curl -s "http://localhost:$PUERTO/v1/models" | python3 -m json.tool
      return 0
    fi
    if [ -f "$PIDFILE" ] && ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      falla "El proceso murio. Ultimas lineas del log:"
      tail -30 "$LOG"
      # Los dos errores que importan, con su remedio.
      if grep -qi "out of memory\|No available memory for the cache" "$LOG"; then
        falla "Es falta de VRAM. Reintenta con margen mas estrecho:"
        falla "  MAX_LEN=768 MAX_SEQS=2 ./07-vm-vllm.sh serve"
      fi
      return 1
    fi
    sleep 10
  done
  falla "No respondio en 10 min. Mira $LOG"
  return 1
}

cmd_test() {
  azul "Prueba de humo"
  curl -s "http://localhost:$PUERTO/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED_NAME\",
         \"prompt\":\"<start_of_turn>user\\nYou are an expert on ISO, UNE and BS acoustics standards for noise measurement. Answer the question precisely and cite the clause when you know it.\\n\\nQuestion: What does ISO 3382-1 specify?<end_of_turn>\\n<start_of_turn>model\\n\",
         \"max_tokens\":256, \"temperature\":0}" \
    | python3 -m json.tool
}

cmd_stop() {
  if [ -f "$PIDFILE" ]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null && ok "vLLM detenido" || aviso "ya no estaba corriendo"
    rm -f "$PIDFILE"
  else
    aviso "No hay pidfile"
  fi
  aviso "La VM sigue ENCENDIDA y cobrando. Para dejar de pagar:"
  aviso "  gcloud compute instances stop NOMBRE_VM --zone=ZONA"
}

case "${1:-check}" in
  check)   cmd_check ;;
  install) cmd_install ;;
  sync)    cmd_sync ;;
  serve)   cmd_serve ;;
  status)  cmd_status ;;
  test)    cmd_test ;;
  stop)    cmd_stop ;;
  *) echo "Uso: $0 {check|install|sync|serve|status|test|stop}"; exit 1 ;;
esac
