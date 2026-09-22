#!/usr/bin/env python
"""Interfaz de consulta del taller 3: modelo afinado (vLLM) y RAG Engine.

Corre en tu maquina y consulta el modelo AFINADO que sirve vLLM dentro de la
VM, a traves del tunel SSH, y el corpus de normas en Vertex AI RAG Engine. La
tercera pestana pone las dos respuestas lado a lado sobre la misma pregunta,
con los fragmentos que uso el RAG y el ROUGE de cada una contra la respuesta de
referencia: es la version visual de la conclusion del taller.

ANTES DE ABRIRLA
  1. En la VM:   ./07-vm-vllm.sh serve  &&  ./07-vm-vllm.sh status
  2. En Windows: gcloud compute ssh VM --zone=ZONA --ssh-flag="-N"
                 --ssh-flag="-L" --ssh-flag="8000:localhost:8000"
  3. En otra ventana, con .venv-rag activado (trae streamlit y el SDK de RAG):
        streamlit run app/streamlit_app.py

Cada parte degrada por separado: sin tunel funciona la pestana de RAG, y sin
el SDK de RAG funciona la del modelo afinado.
"""
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src" / "deploy"))
sys.path.insert(0, str(RAIZ / "src" / "rag"))
sys.path.insert(0, str(RAIZ / "src" / "eval"))

from plantillas import PLANTILLAS, limpiar_eco  # noqa: E402

EVAL_SET = RAIZ / "data" / "qa" / "normas_ruido_eval.jsonl"
BITACORA = RAIZ / "evidencia" / "consultas-streamlit.md"

st.set_page_config(page_title="Taller 3 NLP - Normas de ruido", layout="wide")


# --- Carga perezosa de lo opcional -------------------------------------------
@st.cache_data
def cargar_eval():
    if not EVAL_SET.exists():
        return []
    return [json.loads(l) for l in EVAL_SET.read_text(encoding="utf-8").splitlines() if l.strip()]


@st.cache_resource
def cargar_rouge():
    """ROUGE es opcional: si falta rouge-score, la app funciona sin las metricas."""
    try:
        from metrics import limpiar_prediccion, rouge
        return rouge, limpiar_prediccion
    except Exception:  # noqa: BLE001
        return None, None


@st.cache_data(ttl=30)
def modelo_servido(base_url):
    """Cacheado 30 s: sin eso, cada rerun de Streamlit (y hay uno por cada
    interaccion con un widget) dispara una llamada al endpoint."""
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/v1/models", timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))["data"][0]["id"]


@st.cache_resource
def cliente_rag():
    """Devuelve (cliente, (estado, modulo)) o (None, mensaje de error)."""
    estado_path = RAIZ / "src" / "rag" / ".rag_corpus.json"
    if not estado_path.exists():
        return None, "Falta src/rag/.rag_corpus.json: corre antes `vertex_rag_engine.py create`."
    try:
        import agentplatform
        from google import genai  # noqa: F401
    except ImportError as e:
        return None, (f"Falta una dependencia ({e}). Activa .venv-rag o instala "
                      "google-cloud-agentplatform y google-genai.")
    import vertex_rag_engine as vre
    estado = json.loads(estado_path.read_text(encoding="utf-8"))
    proyecto = vre.numero_de_proyecto(estado) or estado["project"]
    return agentplatform.Client(project=proyecto, location=estado["location"]), (estado, vre)


def args_rag(estado, vre, base_url):
    """Los generadores de vertex_rag_engine.py esperan el namespace de argparse."""
    return SimpleNamespace(project=vre.numero_de_proyecto(estado) or estado["project"],
                           model=vre.MODELO_GEN, base_url=base_url)


# --- Llamadas ----------------------------------------------------------------
def generar_vllm(base_url, modelo, prompt, max_tokens, temperatura):
    cuerpo = json.dumps({
        "model": modelo, "prompt": prompt,
        "max_tokens": max_tokens, "temperature": temperatura,
        "stop": ["<end_of_turn>"],
    }).encode("utf-8")
    req = urllib.request.Request(f"{base_url.rstrip('/')}/v1/completions",
                                 data=cuerpo, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        datos = json.loads(r.read().decode("utf-8"))
    return limpiar_eco(datos["choices"][0]["text"]), datos.get("usage", {})


def consultar_rag(cliente, estado, vre, pregunta, top_k, tecnica, base_url):
    hits = vre.recuperar(cliente, estado["corpus"], pregunta, top_k, vre.mapa_fuentes())
    texto = vre.generar_gemini(pregunta, hits, tecnica, args_rag(estado, vre, base_url))
    return texto, hits


def rouge_contra_referencia(prediccion, referencia, tecnica):
    calc, limpiar = cargar_rouge()
    if not calc or not prediccion or not referencia:
        return None
    return calc(limpiar(prediccion, tecnica), referencia)


# --- Presentacion ------------------------------------------------------------
def mostrar_fragmentos(hits, titulo="Fragmentos recuperados"):
    if not hits:
        st.warning("El recuperador no devolvio fragmentos para esta pregunta.")
        return
    st.markdown(f"**{titulo}**")
    fuentes = list(dict.fromkeys(h["source_file"] for h in hits))
    st.caption("Normas consultadas, en orden de ranking: " + " · ".join(fuentes))
    for h in hits:
        etiqueta = f"[{h['rank']}] {h['source_file']}"
        if h.get("score") is not None:
            etiqueta += f"   (score {h['score']:.4f})"
        with st.expander(etiqueta):
            st.text(h["text"][:2000])


def mostrar_metricas(prediccion, referencia, tecnica, latencia, uso=None):
    cols = st.columns(4 if uso else 3)
    cols[0].metric("Latencia", f"{latencia:.1f} s")
    cols[1].metric("Caracteres", len(prediccion or ""))
    sc = rouge_contra_referencia(prediccion, referencia, tecnica)
    cols[2].metric("ROUGE-1", f"{sc['rouge1']:.4f}" if sc else "n/d",
                   help="F-measure contra la respuesta de referencia del dataset. "
                        "Es la misma metrica de la tabla comparativa, calculada "
                        "aqui sobre una sola pregunta.")
    if uso:
        cols[3].metric("Tokens generados", uso.get("completion_tokens", "?"))


def guardar_en_bitacora(bloque):
    BITACORA.parent.mkdir(parents=True, exist_ok=True)
    cabecera = "" if BITACORA.exists() else (
        "# Consultas hechas desde la app de Streamlit\n\n"
        "Cada entrada es una consulta real a los dos sistemas desplegados, guardada\n"
        "desde la interfaz. Sirve como evidencia escrita, que no se degrada como una\n"
        "captura de pantalla y se puede citar en el informe.\n")
    with BITACORA.open("a", encoding="utf-8") as f:
        f.write(cabecera + bloque)


# --- Barra lateral -----------------------------------------------------------
st.sidebar.title("Configuracion")
base_url = st.sidebar.text_input("Endpoint vLLM", "http://localhost:8000",
                                 help="Con el tunel SSH abierto, es localhost:8000")
tecnica = st.sidebar.selectbox("Tecnica de prompt (modelo afinado)", list(PLANTILLAS))
max_tokens = st.sidebar.slider("max_tokens", 64, 512, 256, 32)
temperatura = st.sidebar.slider("temperature", 0.0, 1.0, 0.0, 0.1,
                                help="0.0 = determinista, igual que en la evaluacion")

st.sidebar.divider()
try:
    modelo = modelo_servido(base_url)
    st.sidebar.success(f"Modelo afinado en linea\n\n`{modelo}`")
except (urllib.error.URLError, OSError, KeyError, IndexError):
    modelo = None
    st.sidebar.error("Sin conexion al endpoint vLLM.\n\n"
                     "Revisa el tunel SSH y que el server este arriba.")

_cliente, _extra = cliente_rag()
if _cliente is None:
    st.sidebar.error(_extra)
    estado_rag = vre = None
else:
    estado_rag, vre = _extra
    st.sidebar.success(f"RAG Engine en linea\n\n`{estado_rag['display_name']}`\n\n"
                       f"{estado_rag['location']}")

st.sidebar.divider()
top_k = st.sidebar.slider("Fragmentos a recuperar (top_k)", 1, 10, 5)
tecnica_rag = st.sidebar.selectbox(
    "Tecnica de prompt (RAG)", list(vre.PLANTILLAS_RAG) if vre else ["rag-anclado"])

st.sidebar.divider()
st.sidebar.caption(
    "**Afinado**: gemma-7b-it + QLoRA sobre 80 pares de normas, fusionado y "
    "servido con vLLM en una VM de GCE.\n\n"
    "**RAG**: las 33 normas fuente en Vertex AI RAG Engine, recuperacion con "
    "text-embedding-005 y generacion anclada.")

# --- Cabecera y banco de preguntas ------------------------------------------
st.title("Normas de medicion de ruido")
st.caption("Taller 3 - SI7016 NLP Aplicado - fine-tuning contra RAG sobre gemma-7b-it")

eval_set = cargar_eval()
banco = {f["question"]: f for f in eval_set}

elegida = st.selectbox(
    "Preguntas del conjunto de evaluacion (opcional)",
    ["-- escribir una propia --"] + list(banco),
    help="Estas 20 son las que se usaron para medir los tres sistemas. Al elegir "
         "una se muestra tambien su respuesta de referencia y el ROUGE de cada "
         "sistema contra ella.")
pregunta = st.text_area("Pregunta", height=90,
                        value="" if elegida.startswith("--") else elegida,
                        placeholder="What does ISO 3382-1 specify?")
referencia = (banco.get(pregunta) or {}).get("answer", "")


def mostrar_referencia():
    fila = banco.get(pregunta)
    if not fila:
        st.info("Pregunta fuera del conjunto de evaluacion: no hay respuesta de "
                "referencia, asi que no se puede calcular ROUGE.")
        return
    with st.expander("Respuesta de referencia del dataset", expanded=False):
        st.write(fila["answer"])
        st.caption(f"Fuente: {fila['source_file']}, seccion {fila['section_label']}")


pestanas = st.tabs(["Lado a lado", "Modelo afinado (vLLM)", "RAG Engine (Vertex)"])

# --- Pestana 1: lado a lado (la que vale como evidencia) ---------------------
with pestanas[0]:
    st.caption("La misma pregunta a los dos sistemas desplegados. El afinado "
               "responde de memoria; el RAG copia del texto recuperado. La "
               "diferencia entre las dos respuestas es la conclusion del taller.")
    if st.button("Consultar ambos", type="primary", disabled=not pregunta):
        izq, der = st.columns(2)
        resultado = {}

        with izq:
            st.markdown("### Modelo afinado")
            st.caption(f"vLLM en la VM · tecnica {tecnica}")
            if not modelo:
                st.error("Sin endpoint vLLM.")
            else:
                t0 = time.time()
                try:
                    texto, uso = generar_vllm(base_url, modelo,
                                              PLANTILLAS[tecnica].format(q=pregunta),
                                              max_tokens, temperatura)
                    st.write(texto or "_(vacia)_")
                    mostrar_metricas(texto, referencia, tecnica, time.time() - t0, uso)
                    resultado["ft"] = texto
                except Exception as e:  # noqa: BLE001
                    st.error(str(e))

        with der:
            st.markdown("### RAG Engine")
            st.caption(f"Vertex AI · tecnica {tecnica_rag} · top_k {top_k}")
            if _cliente is None:
                st.info(_extra)
            else:
                t0 = time.time()
                try:
                    texto, hits = consultar_rag(_cliente, estado_rag, vre, pregunta,
                                                top_k, tecnica_rag, base_url)
                    st.write(texto or "_(vacia)_")
                    mostrar_metricas(texto, referencia, tecnica_rag, time.time() - t0)
                    resultado["rag"], resultado["hits"] = texto, hits
                except Exception as e:  # noqa: BLE001
                    st.error(str(e))

        mostrar_referencia()
        if resultado.get("hits"):
            st.divider()
            mostrar_fragmentos(resultado["hits"],
                               "Lo que el RAG recupero para esta pregunta")
        st.session_state["ultima"] = {**resultado, "pregunta": pregunta,
                                      "referencia": referencia}

    if st.session_state.get("ultima", {}).get("pregunta"):
        if st.button("Guardar esta comparacion en evidencia/"):
            u = st.session_state["ultima"]
            fuentes = " · ".join(dict.fromkeys(h["source_file"] for h in u.get("hits", [])))
            bloque = (f"\n## {u['pregunta']}\n\n"
                      f"_{datetime.now():%Y-%m-%d %H:%M}_ · afinado: {tecnica} · "
                      f"RAG: {tecnica_rag}, top_k {top_k}\n\n"
                      f"**Modelo afinado**\n\n{u.get('ft', '(sin respuesta)')}\n\n"
                      f"**RAG Engine**\n\n{u.get('rag', '(sin respuesta)')}\n\n"
                      f"**Referencia**\n\n{u.get('referencia') or '(fuera del eval set)'}\n\n"
                      f"**Normas recuperadas**: {fuentes or '(ninguna)'}\n")
            guardar_en_bitacora(bloque)
            st.success(f"Guardado en {BITACORA.relative_to(RAIZ)}")

# --- Pestana 2: solo el modelo afinado ---------------------------------------
with pestanas[1]:
    if st.button("Consultar el modelo afinado", disabled=not pregunta):
        if not modelo:
            st.error("No hay endpoint. Revisa la barra lateral.")
        else:
            prompt = PLANTILLAS[tecnica].format(q=pregunta)
            t0 = time.time()
            with st.spinner("Generando..."):
                try:
                    texto, uso = generar_vllm(base_url, modelo, prompt, max_tokens, temperatura)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Fallo la consulta: {e}")
                    texto = None
            if texto is not None:
                st.markdown("### Respuesta")
                st.write(texto or "_(vacia)_")
                mostrar_metricas(texto, referencia, tecnica, time.time() - t0, uso)
                mostrar_referencia()
                with st.expander("Prompt exacto que se envio"):
                    st.code(prompt, language="text")

# --- Pestana 3: solo el RAG --------------------------------------------------
with pestanas[2]:
    if _cliente is None:
        st.info(_extra)
    else:
        st.caption(f"Corpus `{estado_rag['display_name']}` en {estado_rag['location']}, "
                   f"33 normas fuente, chunks de {vre.CHUNK_SIZE} con solape "
                   f"{vre.CHUNK_OVERLAP}, embeddings text-embedding-005.")
        solo_recuperar = st.checkbox(
            "Solo recuperar, sin generar", value=False,
            help="Muestra los fragmentos sin llamar al generador. Util para "
                 "verificar el recuperador por separado y mas rapido.")
        if st.button("Consultar el RAG", disabled=not pregunta):
            t0 = time.time()
            with st.spinner("Recuperando..."):
                try:
                    hits = vre.recuperar(_cliente, estado_rag["corpus"], pregunta,
                                         top_k, vre.mapa_fuentes())
                    texto = None if solo_recuperar else vre.generar_gemini(
                        pregunta, hits, tecnica_rag, args_rag(estado_rag, vre, base_url))
                except Exception as e:  # noqa: BLE001
                    st.error(f"Fallo la consulta: {e}")
                    hits, texto = None, None
            if hits is not None:
                if texto is not None:
                    st.markdown("### Respuesta anclada")
                    st.write(texto or "_(vacia)_")
                    mostrar_metricas(texto, referencia, tecnica_rag, time.time() - t0)
                    mostrar_referencia()
                    st.divider()
                mostrar_fragmentos(hits)
