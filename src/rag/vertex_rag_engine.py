#!/usr/bin/env python
"""Fase 3: el mismo corpus de normas, ahora en Vertex AI RAG Engine.

Que hace y por que. La Fase 1 monto un RAG local (chunking propio + embeddings
de sentence-transformers + FAISS). Este script reconstruye ese mismo pipeline
como servicio administrado en GCP: el corpus vive en RAG Engine (consola:
Agent Platform -> RAG Engine), el chunking y los embeddings los hace Google, y
la recuperacion es una llamada a la API. El resultado es una cuarta familia de
filas en la tabla comparativa, con el sistema 'rag-vertex'.

Lo que se mantiene igual a proposito, para que la comparacion mida el
recuperador y no otra cosa:
  - las mismas 20 preguntas del split de evaluacion,
  - las mismas plantillas de src/rag/prompts.py (PLANTILLAS_RAG),
  - el mismo esquema de salida del HANDOFF, seccion 4.2,
  - el mismo generador, si se usa --generator vllm.

REGION. El corpus va en us-west1 y no en us-central1 a proposito: el modo
'Spanner' de RAG Engine, que es el que usa por defecto un proyecto nuevo,
esta restringido a proyectos en lista blanca en us-central1, us-east1 y
us-east4, y falla con 'INVALID_ARGUMENT ... restricted to only allowlisted
projects'. us-west1 queda fuera de esa restriccion. El bucket del corpus tiene
que estar en la misma region; es independiente del bucket del fine-tuning.

DEPENDENCIAS. Instalar en un entorno aparte del principal: el SDK nuevo
(google-cloud-agentplatform) sube google-cloud-aiplatform a la linea 2.x, y el
resto del repo esta fijado en 1.75.0.

    python -m venv .venv-rag && .venv-rag\\Scripts\\activate
    pip install google-cloud-agentplatform google-genai google-cloud-storage pandas

USO, en orden:
    python src/rag/vertex_rag_engine.py upload     # sube los 33 .txt fuente a GCS
    python src/rag/vertex_rag_engine.py create     # crea el corpus
    python src/rag/vertex_rag_engine.py import     # importa y vectoriza (asincrono)
    python src/rag/vertex_rag_engine.py status     # cuantos archivos quedaron indexados
    python src/rag/vertex_rag_engine.py ask --q "What does ISO 3382-1 specify?"
    python src/rag/vertex_rag_engine.py run        # las 20 preguntas -> jsonl
    python src/rag/vertex_rag_engine.py delete     # borra el corpus (deja de cobrar)
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(RAIZ / "src" / "data"))

from prompts import PLANTILLAS_RAG, construir_prompt  # noqa: E402

ESTADO = Path(__file__).with_name(".rag_corpus.json")

PROJECT = "si7016-262-nlp"
LOCATION = "us-west1"
# Se reutiliza el bucket de us-west1 que ya existia (lo creo el job de
# inferencia que hubo que mover de region). El corpus tiene que estar en la
# misma region que el bucket, y los documentos van bajo su propio prefijo,
# asi que no se mezclan con los artefactos que ya viven ahi.
BUCKET = "asilvaz1taller3-west"
PREFIJO = "normas-ruido/"
CORPUS_NOMBRE = "normas-ruido-taller3"
EMBEDDING = "publishers/google/models/text-embedding-005"
MODELO_GEN = "gemini-2.5-flash"
GENAI_LOCATION = "global"

# Tope de la API de RAG Engine: 'GCS URIs cannot be specified more than 25
# times'. Con 33 documentos, la importacion va en dos lotes.
SISTEMAS_POR_GENERADOR = {"gemini": "rag-vertex", "vllm": "rag-vertex-ft"}

LOTE_URIS = 25

CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
TOP_K = 5
UMBRAL_DISTANCIA = 0.5


# --- Mapa .txt -> PDF original ----------------------------------------------
# El eval set identifica la fuente por el nombre del PDF ("UNE-EN ISO 3382-1.pdf"),
# pero lo que se sube al corpus es el .txt extraido, cuyo nombre es el slug del
# PDF. Sin este mapa, Recall@K compara manzanas con peras.
def mapa_fuentes():
    from extract_pdf import slug  # misma funcion que nombro los .txt
    mapa = {}
    with (RAIZ / "data" / "qa" / "corpus_map.csv").open(encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            if fila["estado"] != "fuente":
                continue
            mapa[f"{slug(fila['archivo_pdf'])}.txt"] = fila["archivo_pdf"]
    return mapa


def cargar_estado():
    if not ESTADO.exists():
        raise SystemExit(f"No existe {ESTADO.name}. Corre antes el subcomando 'create'.")
    return json.loads(ESTADO.read_text(encoding="utf-8"))


def numero_de_proyecto(est):
    """El numero de proyecto, sacado del nombre del corpus.

    Vertex devuelve los nombres de recurso como
    projects/<NUMERO>/locations/<region>/ragCorpora/<id>, asi que el numero ya
    esta ahi y no hay que preguntarselo a nadie.

    Importa porque el SDK, cuando le pasas un project ID, lo traduce a numero
    llamando a la Cloud Resource Manager API. En un proyecto donde esa API no
    esta habilitada, eso falla, y como toda la comprobacion vive dentro de un
    'except Exception: return False', el error que sale es
    'Bucket ... does not belong to project ...'. Pasando el numero desde el
    principio, el SDK ni siquiera intenta la traduccion: hay una rama
    'if expected_project.isdigit()' que corta antes.

    Devuelve None si el nombre no trae numero (por ejemplo si una version
    futura devolviera el project ID), y entonces se usa el ID de siempre.
    """
    partes = (est.get("corpus") or "").split("/")
    if len(partes) > 1 and partes[0] == "projects" and partes[1].isdigit():
        return partes[1]
    return est.get("project_number")


def cliente_rag(args, est=None):
    import agentplatform
    proyecto = (numero_de_proyecto(est) if est else None) or args.project
    return agentplatform.Client(project=proyecto, location=args.location)


# --- Subcomandos -------------------------------------------------------------
def cmd_upload(args):
    """Sube a GCS solo los .txt de las normas FUENTE, no los 41 extraidos.

    Los 8 descartados son duplicados o ediciones superadas (ver corpus_map.csv):
    indexarlos meteria en el corpus texto que el dataset QA decidio no usar, y
    el RAG podria recuperar la version equivocada de una misma norma."""
    from google.cloud import storage

    mapa = mapa_fuentes()
    origen = RAIZ / "data" / "processed"
    faltantes = [n for n in mapa if not (origen / n).exists()]
    if faltantes:
        raise SystemExit(
            f"Faltan {len(faltantes)} .txt en data/processed: {faltantes[:5]}\n"
            "Corre antes: python src/data/extract_pdf.py")

    cliente = storage.Client(project=args.project)
    bucket = cliente.bucket(args.bucket)
    if not bucket.exists():
        raise SystemExit(
            f"No existe gs://{args.bucket}. Creala en la region del corpus:\n"
            f"  gcloud storage buckets create gs://{args.bucket} --location={args.location}")

    for i, nombre in enumerate(sorted(mapa), 1):
        blob = bucket.blob(f"{PREFIJO}{nombre}")
        blob.upload_from_filename(str(origen / nombre), content_type="text/plain")
        print(f"  [{i}/{len(mapa)}] {nombre}")
    print(f"\n{len(mapa)} documentos en gs://{args.bucket}/{PREFIJO}")
    print("\nPaso manual que falta, una sola vez: darle al agente de servicio de")
    print("RAG Engine permiso de lectura sobre el bucket.")
    print(f"  gcloud storage buckets add-iam-policy-binding gs://{args.bucket} \\")
    print('    --member="serviceAccount:service-NUMERO_PROYECTO@gcp-sa-vertex-rag.iam.gserviceaccount.com" \\')
    print('    --role="roles/storage.objectViewer"')
    print("  (NUMERO_PROYECTO sale de: gcloud projects describe "
          f"{args.project} --format='value(projectNumber)')")


def cmd_create(args):
    from agentplatform import types
    cliente = cliente_rag(args)

    # Si ya existe uno con el mismo display_name, se reutiliza: crear dos
    # corpus iguales cobra almacenamiento vectorial dos veces.
    for c in (cliente.rag.list_corpora().rag_corpora or []):
        if c.display_name == args.corpus_name:
            print(f"Ya existia: {c.name}")
            ESTADO.write_text(json.dumps({
                "project": args.project, "location": args.location,
                "bucket": args.bucket, "corpus": c.name,
                "display_name": c.display_name}, indent=2), encoding="utf-8")
            return 0

    corpus = cliente.rag.create_corpus(
        rag_corpus=types.RagCorpus(
            display_name=args.corpus_name,
            description="41 normas ISO/UNE/BS de medicion de ruido; 33 fuente. Taller 3 SI7016.",
            rag_vector_db_config=types.RagVectorDbConfig(
                rag_embedding_model_config=types.RagEmbeddingModelConfig(
                    vertex_prediction_endpoint=types.RagEmbeddingModelConfigVertexPredictionEndpoint(
                        endpoint=EMBEDDING))),
        ))
    print("Corpus creado:", corpus.name)
    estado = {"project": args.project, "location": args.location,
              "bucket": args.bucket, "corpus": corpus.name,
              "display_name": corpus.display_name, "embedding": EMBEDDING}
    estado["project_number"] = numero_de_proyecto(estado)
    ESTADO.write_text(json.dumps(estado, indent=2), encoding="utf-8")
    if estado["project_number"]:
        print("Numero de proyecto:", estado["project_number"])
    return 0


def verificar_propiedad_bucket(project, bucket_name):
    """Reproduce la comprobacion que hace import_files, pero diciendo la verdad.

    El SDK verifica que el bucket pertenezca al proyecto (una defensa contra
    'bucket squatting'), y para eso traduce el project ID a numero de proyecto
    con google-cloud-resource-manager. Toda esa logica vive dentro de un
    'except Exception: return False', asi que un paquete que falta, una API sin
    habilitar o un permiso ausente salen todos como
    'Bucket ... does not belong to project ...', que manda a revisar el bucket
    cuando el bucket esta bien. Esta funcion separa los casos.
    """
    from google.cloud import storage
    try:
        bucket = storage.Client(project=project).bucket(bucket_name)
        bucket.reload()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"No se pudo leer gs://{bucket_name}: {e}\n"
            "Revisa el nombre del bucket y que tu cuenta tenga acceso.")

    if str(project).isdigit():
        esperado = str(project)
    else:
        try:
            from google.cloud import resourcemanager_v3
        except ImportError:
            raise SystemExit(
                "Falta el paquete google-cloud-resource-manager.\n\n"
                "agentplatform no lo declara como dependencia, pero import_files lo\n"
                "necesita para traducir el project ID a numero de proyecto. Sin el,\n"
                "el SDK falla con 'Bucket ... does not belong to project ...', que es\n"
                "un mensaje enganoso: el bucket esta bien.\n\n"
                "    pip install google-cloud-resource-manager")
        try:
            proyecto = resourcemanager_v3.ProjectsClient().get_project(
                name=f"projects/{project}")
            esperado = proyecto.name.split("/")[-1]
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"No se pudo resolver el numero del proyecto {project}: {e}\n\n"
                "Suele ser la API de Cloud Resource Manager sin habilitar, o falta\n"
                "el permiso resourcemanager.projects.get:\n"
                "    gcloud services enable cloudresourcemanager.googleapis.com")

    if str(bucket.project_number) != esperado:
        raise SystemExit(
            f"gs://{bucket_name} pertenece al proyecto {bucket.project_number} y "
            f"no a {project} ({esperado}). Esta vez el mensaje del SDK si seria "
            "literal: usa un bucket del proyecto correcto.")
    return esperado


def listar_objetos(project, bucket_name, prefijo):
    """Los objetos que hay de verdad bajo el prefijo, como URIs completas."""
    from google.cloud import storage
    bucket = storage.Client(project=project).bucket(bucket_name)
    return sorted(f"gs://{bucket_name}/{b.name}"
                  for b in bucket.list_blobs(prefix=prefijo)
                  if not b.name.endswith("/"))


def cmd_import(args):
    from agentplatform import types
    from google.genai import types as gt

    est = cargar_estado()
    # Con el numero de proyecto, ni esta verificacion ni la del SDK necesitan
    # la Cloud Resource Manager API.
    verificar_propiedad_bucket(numero_de_proyecto(est) or est["project"], est["bucket"])

    # Se importan las URIs de los 33 objetos, una por una, en vez de la forma
    # con comodin gs://bucket/prefijo/*. El comodin es lo que documenta el
    # quickstart y lo que usa el notebook de la clase, pero ahi mismo hay una
    # nota diciendo que al final los archivos se cargaron a mano por la
    # consola: el import con comodin devuelve 0 importados, 0 fallidos y 0
    # omitidos, sin error, y el corpus queda vacio. Con URIs explicitas no hay
    # ambiguedad, y ademas sabemos exactamente que subimos.
    uris = listar_objetos(est["project"], est["bucket"], PREFIJO)
    if not uris:
        raise SystemExit(
            f"No hay objetos bajo gs://{est['bucket']}/{PREFIJO}\n"
            "Corre antes el subcomando 'upload'.")
    print(f"{len(uris)} objetos a importar (chunk {CHUNK_SIZE}, solape {CHUNK_OVERLAP})")
    for u in uris[:3]:
        print(f"  {u}")
    if len(uris) > 3:
        print(f"  ... y {len(uris) - 3} mas")

    cliente = cliente_rag(args, est)

    # La API rechaza mas de 25 URIs por llamada ("GCS URIs cannot be specified
    # more than 25 times"), asi que 33 documentos van en dos lotes. Los
    # contadores se suman entre lotes para que el resumen final sea del
    # conjunto y no del ultimo.
    total = {"importados": 0, "fallidos": 0, "omitidos": 0}
    fallos_detallados = []
    for i in range(0, len(uris), LOTE_URIS):
        lote = uris[i:i + LOTE_URIS]
        n_lote = i // LOTE_URIS + 1
        print(f"\n  lote {n_lote}: {len(lote)} documentos")
        respuesta = cliente.rag.import_files(
            name=est["corpus"],
            import_config=types.ImportRagFilesConfig(
                gcs_source=gt.GcsSource(uris=lote),
                rag_file_transformation_config=types.RagFileTransformationConfig(
                    rag_file_chunking_config=types.RagFileChunkingConfig(
                        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)),
                max_embedding_requests_per_min=1000,
            ))
        total["importados"] += respuesta.imported_rag_files_count or 0
        total["fallidos"] += respuesta.failed_rag_files_count or 0
        total["omitidos"] += respuesta.skipped_rag_files_count or 0
        if respuesta.partial_failures_gcs_path:
            fallos_detallados.append(respuesta.partial_failures_gcs_path)
        print(f"    importados {respuesta.imported_rag_files_count or 0}, "
              f"fallidos {respuesta.failed_rag_files_count or 0}, "
              f"omitidos {respuesta.skipped_rag_files_count or 0}")

    print(f"\n  TOTAL importados : {total['importados']} de {len(uris)}")
    print(f"  TOTAL fallidos   : {total['fallidos']}")
    print(f"  TOTAL omitidos   : {total['omitidos']}")
    for ruta in fallos_detallados:
        print(f"  detalle de fallos: {ruta}")

    if total["importados"] == 0 and total["fallidos"] == 0:
        print("\nCero importados y cero fallidos es el sintoma de que RAG Engine no")
        print("pudo LEER los objetos. Casi siempre es el permiso del agente de")
        print("servicio sobre el bucket:")
        print(f"  gcloud storage buckets get-iam-policy gs://{est['bucket']} "
              "--format=json | Select-String vertex-rag")
        return 1
    print("\nConfirma con 'status' antes de consultar.")
    return 0


def listar_archivos_corpus(cliente, corpus):
    """Todos los archivos del corpus, recorriendo las paginas.

    list_files devuelve 25 por pagina. Leer solo la primera hace creer que el
    corpus tiene 25 documentos cuando tiene 33, que es exactamente el numero
    de lotes que uso la importacion: dos errores distintos que se disfrazan
    del mismo sintoma.
    """
    from agentplatform import types
    archivos, token = [], None
    while True:
        config = types.ListRagFilesConfig(page_size=100, page_token=token) if token \
            else types.ListRagFilesConfig(page_size=100)
        respuesta = cliente.rag.list_files(name=corpus, config=config)
        archivos += respuesta.rag_files or []
        token = respuesta.next_page_token
        if not token:
            return archivos


def cmd_status(args):
    est = cargar_estado()
    cliente = cliente_rag(args, est)
    archivos = listar_archivos_corpus(cliente, est["corpus"])
    print(f"Corpus : {est['corpus']}")
    print(f"Archivos indexados: {len(archivos)}")

    # Un archivo puede estar listado y aun asi haber fallado al vectorizarse.
    estados = {}
    for f in archivos:
        estado = str(getattr(f.file_status, "state", "") or "SIN_ESTADO")
        estados[estado] = estados.get(estado, 0) + 1
    for estado, n in sorted(estados.items()):
        print(f"  {estado}: {n}")

    esperados = len(mapa_fuentes())
    if len(archivos) < esperados:
        faltan = sorted({n for n in mapa_fuentes()} -
                        {f.display_name for f in archivos})
        print(f"\nAVISO: se esperaban {esperados} y hay {len(archivos)}.")
        print(f"Faltan {len(faltan)}: {', '.join(faltan[:8])}"
              + (" ..." if len(faltan) > 8 else ""))
        print("Si acabas de importar, espera unos minutos y repite. Si no, corre")
        print("'import' otra vez: los archivos que ya estan se omiten.")
        return 1
    print(f"\nLos {esperados} documentos fuente estan en el corpus.")
    return 0


def recuperar(cliente, corpus, pregunta, top_k, mapa):
    """Devuelve los hits en el formato que espera src/rag/prompts.py."""
    from agentplatform import types
    from google.genai import types as gt

    respuesta = cliente.rag.retrieve_contexts(
        vertex_rag_store=gt.VertexRagStore(
            rag_resources=[gt.VertexRagStoreRagResource(rag_corpus=corpus)]),
        query=types.RagQuery(
            text=pregunta,
            rag_retrieval_config=gt.RagRetrievalConfig(
                top_k=top_k,
                filter=gt.RagRetrievalConfigFilter(
                    vector_distance_threshold=UMBRAL_DISTANCIA))),
    )
    contextos = (respuesta.contexts.contexts if respuesta.contexts else []) or []
    hits = []
    for rango, c in enumerate(contextos, 1):
        archivo_txt = (c.source_uri or "").rsplit("/", 1)[-1]
        pdf = mapa.get(archivo_txt, archivo_txt)
        chunk = getattr(c, "chunk", None)
        hits.append({
            "rank": rango,
            "norma": Path(pdf).stem,
            "text": (c.text or (chunk.text if chunk else "") or "").strip(),
            "source_file": pdf,
            # RAG Engine no expone la posicion del chunk dentro del documento,
            # solo un id opaco. Se conserva tal cual para trazabilidad, pero
            # NO es el indice de 4 digitos del HANDOFF 4.3: el Recall@K a nivel
            # de chunk no es comparable con el del FAISS, el de documento si.
            "chunk_id": f"{pdf}#{getattr(chunk, 'chunk_id', None) or f'rank{rango:04d}'}",
            "score": c.score,
            "distance": c.distance,
        })
    return hits


def generar_gemini(pregunta, hits, tecnica, args):
    """Genera con Gemini pasandole el contexto en el prompt.

    Se recupera por separado y se inyecta el contexto a mano, en vez de darle a
    Gemini la herramienta de recuperacion, por dos razones: asi se usan las
    MISMAS plantillas que el RAG de la Fase 1, y asi 'retrieved_sources' es
    exactamente lo que vio el generador. La version con herramienta, que es la
    del notebook de la clase, esta en el subcomando 'ask'."""
    from google import genai

    prompt = construir_prompt(tecnica, pregunta, hits)
    # Gemini no usa el chat template de Gemma: las etiquetas de turno se quitan
    # para que no lleguen como texto literal dentro del prompt.
    for token in ("<start_of_turn>user\n", "<end_of_turn>", "<start_of_turn>model\n"):
        prompt = prompt.replace(token, "")
    cliente = genai.Client(enterprise=True, project=args.project, location=GENAI_LOCATION)
    respuesta = cliente.models.generate_content(model=args.model, contents=prompt.strip())
    return (respuesta.text or "").strip()


_VLLM = {}


def _cliente_vllm(args):
    """Carga una sola vez el cliente de vLLM y el nombre del modelo servido.

    Sin la cache, cada una de las 80 generaciones volvia a ejecutar el modulo
    08-predict-vllm.py (que a su vez reparsea 03-predict-endpoint.py con ast) y
    ademas preguntaba a /v1/models. Funcionaba, pero eran 160 operaciones
    inutiles por corrida."""
    if not _VLLM:
        import importlib.util
        sys.path.insert(0, str(RAIZ / "src" / "deploy"))
        ruta = RAIZ / "src" / "deploy" / "08-predict-vllm.py"
        spec = importlib.util.spec_from_file_location("predict_vllm", ruta)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # args.model trae el default de Gemini cuando no se pidio otro: en ese
        # caso el nombre correcto se lo preguntamos al propio endpoint.
        modelo = args.model if args.model != MODELO_GEN else mod.modelo_servido(args.base_url)
        _VLLM["mod"], _VLLM["modelo"] = mod, modelo
        print(f"  generador: vLLM en {args.base_url}, modelo {modelo}")
    return _VLLM["mod"], _VLLM["modelo"]


def generar_vllm(pregunta, hits, tecnica, args):
    """Genera con el modelo AFINADO servido por vLLM en la VM.

    Esta es la variante interesante para el informe: cambia el recuperador
    (FAISS local -> RAG Engine) dejando fijo el generador, asi que la
    diferencia en ROUGE se le puede atribuir a la recuperacion."""
    from plantillas import limpiar_eco
    mod, modelo = _cliente_vllm(args)
    prompt = construir_prompt(tecnica, pregunta, hits)
    return limpiar_eco(mod.pedir(args.base_url, modelo, prompt, 256, 0.0, 180.0))


def cmd_ask(args):
    """Consulta suelta, con la generacion anclada por herramienta del notebook
    de la clase. Sirve para la captura de evidencia."""
    from google import genai
    from google.genai import types as gt

    est = cargar_estado()
    cliente = cliente_rag(args, est)
    mapa = mapa_fuentes()

    hits = recuperar(cliente, est["corpus"], args.q, args.top_k, mapa)
    print(f"\n--- {len(hits)} fragmentos recuperados ---")
    for h in hits:
        print(f"  [{h['rank']}] {h['source_file']}  score={h['score']}")
        print(f"      {h['text'][:160]}...")

    herramienta = gt.Tool(retrieval=gt.Retrieval(
        vertex_rag_store=gt.VertexRagStore(
            rag_resources=[gt.VertexRagStoreRagResource(rag_corpus=est["corpus"])],
            rag_retrieval_config=gt.RagRetrievalConfig(top_k=args.top_k))))
    genai_cliente = genai.Client(enterprise=True, project=args.project, location=GENAI_LOCATION)
    con_rag = genai_cliente.models.generate_content(
        model=args.model, contents=args.q,
        config=gt.GenerateContentConfig(tools=[herramienta]))
    sin_rag = genai_cliente.models.generate_content(model=args.model, contents=args.q)

    print("\n--- CON RAG (anclado en las normas) ---")
    print(con_rag.text)
    print("\n--- SIN RAG (solo el conocimiento del modelo) ---")
    print(sin_rag.text)
    return 0


def generar_con_reintentos(generar, pregunta, hits, tecnica, args, intentos=4):
    """Genera, reintentando los 429 con espera creciente.

    La cuota de Gemini es por minuto, asi que un RESOURCE_EXHAUSTED no
    significa que la pregunta sea imposible: significa que hay que esperar. Sin
    reintento, una sola pregunta de 80 se queda sin respuesta y contamina una
    fila entera de la tabla, que es justo lo que paso en la primera corrida.

    Solo reintenta el 429. Un error de permisos o de argumento no mejora
    esperando, y disfrazarlo de lentitud alarga la corrida sin arreglar nada.
    """
    espera = 20
    for intento in range(1, intentos + 1):
        try:
            return generar(pregunta, hits, tecnica, args), None
        except Exception as e:  # noqa: BLE001
            texto = f"{type(e).__name__}: {e}"
            recuperable = "429" in texto or "RESOURCE_EXHAUSTED" in texto
            if not recuperable or intento == intentos:
                return "", texto
            print(f"      cuota agotada; reintento {intento}/{intentos - 1} "
                  f"en {espera} s")
            time.sleep(espera)
            espera *= 2


def cmd_run(args):
    est = cargar_estado()
    cliente = cliente_rag(args, est)
    mapa = mapa_fuentes()

    eval_set = [json.loads(l) for l in
                (RAIZ / "data" / "qa" / "normas_ruido_eval.jsonl")
                .read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        eval_set = eval_set[:args.limit]
    tecnicas = list(PLANTILLAS_RAG) if args.technique == "all" else [args.technique]
    generar = generar_vllm if args.generator == "vllm" else generar_gemini

    # Los dos generadores son SISTEMAS DISTINTOS y no pueden compartir ni el
    # archivo ni la etiqueta: 'rag-vertex' recupera de RAG Engine y genera con
    # Gemini, 'rag-vertex-ft' recupera de lo mismo y genera con el modelo
    # afinado. Mezclarlos promediaria dos cosas distintas en la misma fila, y
    # escribir en el mismo archivo borraria mediciones ya hechas.
    sistema = SISTEMAS_POR_GENERADOR[args.generator]
    destino = RAIZ / (args.out or f"results/respuestas-{sistema}.jsonl")

    # --- Modo reparar: solo las filas que quedaron sin respuesta -------------
    # Repetir las 80 para arreglar una es gastar 79 llamadas y, peor, cambiar
    # las otras 79 respuestas: con un modelo generativo, una segunda corrida no
    # devuelve lo mismo, asi que la tabla ya medida dejaria de corresponder a
    # este archivo.
    if args.reparar:
        if not destino.exists():
            raise SystemExit(f"No existe {destino}. Corre 'run' sin --reparar primero.")
        filas = [json.loads(l) for l in
                 destino.read_text(encoding="utf-8").splitlines() if l.strip()]
        pendientes = [i for i, f in enumerate(filas)
                      if not (f.get("prediction") or "").strip()]
        if not pendientes:
            print(f"Las {len(filas)} filas de {destino.name} tienen respuesta. "
                  "Nada que reparar.")
            return 0
        print(f"{len(pendientes)} de {len(filas)} filas sin respuesta. Reintentando solo esas.\n")
        arregladas = 0
        for i in pendientes:
            fila = filas[i]
            print(f"  [{fila['technique']}] {fila['question'][:70]}")
            # Se vuelve a recuperar: los fragmentos guardados son los mismos,
            # pero asi la fila queda coherente si el corpus cambio.
            hits = recuperar(cliente, est["corpus"], fila["question"], args.top_k, mapa)
            texto, error = generar_con_reintentos(
                generar, fila["question"], hits, fila["technique"], args)
            if error:
                print(f"      sigue fallando: {error}")
                continue
            fila["prediction"] = fila["prediction_raw"] = texto
            fila["retrieved_sources"] = [h["source_file"] for h in hits]
            fila["retrieved_chunk_ids"] = [h["chunk_id"] for h in hits]
            arregladas += 1
            print(f"      {len(texto)} caracteres, {len(hits)} fragmentos")
            time.sleep(args.pausa)

        with destino.open("w", encoding="utf-8") as f:
            for r in filas:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        quedan = len(pendientes) - arregladas
        print(f"\n{arregladas} reparadas, {quedan} sin respuesta todavia.")
        if quedan:
            print("Si el error sigue siendo 429, espera unos minutos y repite: la")
            print("cuota de Gemini es por minuto.")
        return 1 if quedan else 0

    # --- Corrida completa ----------------------------------------------------
    # La recuperacion no depende de la tecnica: se hace una vez por pregunta y
    # se reusa en las cuatro. Ahorra 3 de cada 4 llamadas al recuperador.
    cache = {}
    salida, fallos = [], 0
    for tecnica in tecnicas:
        for i, fila in enumerate(eval_set, 1):
            pregunta = fila["question"]
            if pregunta not in cache:
                cache[pregunta] = recuperar(cliente, est["corpus"], pregunta, args.top_k, mapa)
            hits = cache[pregunta]
            texto, error = generar_con_reintentos(generar, pregunta, hits, tecnica, args)
            if error:
                print(f"  [{tecnica} {i}/{len(eval_set)}] ERROR: {error}")
                fallos += 1
            salida.append({
                "question": pregunta,
                "reference": fila["answer"],
                "prediction": texto,
                "prediction_raw": texto,
                "technique": tecnica,
                "system": sistema,
                "source_file": fila["source_file"],
                "section_label": fila["section_label"],
                "retrieved_sources": [h["source_file"] for h in hits],
                "retrieved_chunk_ids": [h["chunk_id"] for h in hits],
            })
            if not error:
                print(f"  [{tecnica} {i}/{len(eval_set)}] {len(texto)} caracteres, "
                      f"{len(hits)} fragmentos")
            time.sleep(args.pausa)

    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as f:
        for r in salida:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{len(salida)} filas en {destino.relative_to(RAIZ)}"
          + (f"  ({fallos} sin respuesta)" if fallos else ""))
    if fallos:
        print("Para reintentar SOLO esas, sin tocar las demas:")
        print(f"  python src/rag/vertex_rag_engine.py run --reparar "
              f"--generator {args.generator}")
    return 1 if fallos else 0


def cmd_delete(args):
    est = cargar_estado()
    cliente = cliente_rag(args, est)
    cliente.rag.delete_corpus(name=est["corpus"])
    print("Corpus borrado:", est["corpus"])
    ESTADO.unlink(missing_ok=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("comando", choices=["upload", "create", "import", "status",
                                        "ask", "run", "delete"])
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--location", default=LOCATION)
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--corpus_name", default=CORPUS_NOMBRE)
    ap.add_argument("--model", default=MODELO_GEN,
                    help="Modelo de Gemini para la generacion. Si no esta habilitado "
                         "en el proyecto, revisa Model Garden y cambialo.")
    ap.add_argument("--generator", default="gemini", choices=["gemini", "vllm"],
                    help="gemini: generacion administrada. vllm: el modelo AFINADO "
                         "en la VM, para aislar el efecto del recuperador.")
    ap.add_argument("--base_url", default="http://localhost:8000",
                    help="Solo con --generator vllm")
    ap.add_argument("--technique", default="rag-anclado",
                    choices=[*PLANTILLAS_RAG, "all"])
    ap.add_argument("--top_k", type=int, default=TOP_K)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pausa", type=float, default=0.0,
                    help="Segundos entre preguntas, si aparece rate limiting")
    ap.add_argument("--out", default=None,
                    help="Por defecto se deduce del generador: "
                         "results/respuestas-rag-vertex.jsonl con gemini y "
                         "respuestas-rag-vertex-ft.jsonl con vllm. Asi una "
                         "corrida no puede pisar la otra.")
    ap.add_argument("--reparar", action="store_true",
                    help="Con 'run': en vez de generar las 80 filas, lee el archivo "
                         "de --out y reintenta solo las que quedaron sin respuesta. "
                         "Las demas no se tocan.")
    ap.add_argument("--q", default=None, help="Pregunta para 'ask'")
    args = ap.parse_args()

    if args.comando == "ask" and not args.q:
        raise SystemExit("'ask' necesita --q \"tu pregunta\"")

    return {
        "upload": cmd_upload, "create": cmd_create, "import": cmd_import,
        "status": cmd_status, "ask": cmd_ask, "run": cmd_run, "delete": cmd_delete,
    }[args.comando](args)


if __name__ == "__main__":
    sys.exit(main())
