#!/usr/bin/env python
"""Corre el RAG sobre el eval set y escribe results/respuestas-rag.jsonl.

Por cada pregunta recupera k chunks, arma el prompt de cada tecnica y llama al
endpoint del modelo base (el mismo gemma-7b-it de los otros sistemas). Escribe
el esquema del HANDOFF (seccion 4.2) con system='rag' mas prediction_raw (sin
limpiar, como en el base), retrieved_sources y retrieved_chunk_ids, ambos en
orden de ranking y alineados uno a uno.

Uso:
    # solo recuperacion + prompts, sin GPU ni GCP (prediction queda vacia)
    python src/rag/run_rag.py --generator none

    # generacion real contra el endpoint que dejo src/deploy/02
    python src/rag/run_rag.py --generator endpoint
"""
import argparse
import json
import os
import sys
from pathlib import Path

from chunking import RAIZ
from prompts import PLANTILLAS_RAG, construir_prompt
from retriever import MODOS, RERANKER_DEFAULT, Retriever

# El contenedor de Model Garden devuelve el prompt pegado antes de la respuesta;
# la limpieza es la misma que se uso con el modelo base.
sys.path.insert(0, str(RAIZ / "src" / "eval"))
from clean_predictions import limpiar  # noqa: E402

ESTADO = RAIZ / "src" / "deploy" / ".endpoint_base.json"


def crear_endpoint(transport: str):
    """Misma conexion que src/deploy/03-predict-endpoint.py. Devuelve
    (endpoint, estado)."""
    if not ESTADO.exists():
        raise SystemExit(f"No existe {ESTADO.name}: corre antes src/deploy/02-deploy-model-garden-base.py")
    est = json.loads(ESTADO.read_text(encoding="utf-8"))
    os.environ.setdefault("GRPC_DNS_RESOLVER", "native")  # ver nota en 03-predict-endpoint.py
    from google.cloud import aiplatform
    aiplatform.init(project=est["project"], location=est["region"], api_transport=transport)
    endpoint = aiplatform.Endpoint(
        endpoint_name=f"projects/{est['project']}/locations/{est['region']}"
                      f"/endpoints/{est['endpoint_id']}")
    return endpoint, est


def es_error_de_dns(e) -> bool:
    texto = f"{type(e).__name__}: {e}"
    return any(s in texto for s in ("getaddrinfo", "NameResolution", "Failed to resolve",
                                    "11001", "Name or service not known"))


class Generador:
    """Consulta el endpoint. Si el DNS dedicado no resuelve (redes del campus),
    cae a la URL regional compartida, igual que 03-predict-endpoint.py."""

    def __init__(self, transport: str, timeout: float):
        self.endpoint, est = crear_endpoint(transport)
        self.timeout = timeout
        self.url_compartida = (
            f"https://{est['region']}-aiplatform.googleapis.com/v1"
            f"/projects/{est['project']}/locations/{est['region']}"
            f"/endpoints/{est['endpoint_id']}:predict")
        self.compartida = False
        self._sesion = None

    def _via_compartida(self, instancia):
        if self._sesion is None:
            import google.auth
            from google.auth.transport.requests import AuthorizedSession
            cred, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            self._sesion = AuthorizedSession(cred)
        r = self._sesion.post(self.url_compartida, json={"instances": [instancia]},
                              timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        return (r.json().get("predictions") or [None])[0]

    def __call__(self, prompt: str, max_tokens: int, temperature: float) -> str:
        instancia = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature,
                     "top_p": 1.0, "top_k": -1}
        if self.compartida:
            pred = self._via_compartida(instancia)
        else:
            try:
                pred = self.endpoint.predict(instances=[instancia], timeout=self.timeout).predictions[0]
            except Exception as e:  # noqa: BLE001
                if not es_error_de_dns(e):
                    raise
                print("DNS dedicado sin resolver: cambiando a la ruta compartida.")
                self.compartida = True
                pred = self._via_compartida(instancia)
        if isinstance(pred, dict):
            for clave in ("generated_text", "text", "content", "output"):
                if clave in pred:
                    return str(pred[clave])
            return json.dumps(pred, ensure_ascii=False)
        return str(pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(RAIZ / "data" / "qa" / "normas_ruido_eval.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--technique", nargs="+", default=list(PLANTILLAS_RAG),
                    choices=list(PLANTILLAS_RAG))
    ap.add_argument("--generator", choices=["endpoint", "none"], default="none")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--mode", choices=MODOS, default="dense")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--transport", choices=["rest", "grpc"], default="rest")
    ap.add_argument("--timeout", type=float, default=300.0, help="Segundos de espera por respuesta")
    ap.add_argument("--limit", type=int, default=0, help="0 = todas")
    args = ap.parse_args()

    # Sin generador el archivo no es una entrega valida: va a otro nombre para
    # que run_eval.py no lo lea por accidente como respuestas reales.
    nombre = "respuestas-rag.jsonl" if args.generator == "endpoint" else "rag-solo-recuperacion.jsonl"
    destino = Path(args.out) if args.out else RAIZ / "results" / nombre

    filas = [json.loads(l) for l in Path(args.dataset).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        filas = filas[:args.limit]

    ret = Retriever(RERANKER_DEFAULT if args.rerank else None)
    generador = Generador(args.transport, args.timeout) if args.generator == "endpoint" else None

    salida = []
    for i, d in enumerate(filas, 1):
        print(f"[{i}/{len(filas)}] {d['question'][:70]}...")
        hits = ret.search(d["question"], args.k, args.mode)
        for tecnica in args.technique:
            bruto = ""
            if generador:
                try:
                    bruto = generador(construir_prompt(tecnica, d["question"], hits),
                                      args.max_tokens, args.temperature)
                except Exception as e:  # noqa: BLE001
                    bruto = f"__ERROR__: {e}"
            pred = bruto if bruto.startswith("__ERROR__") else limpiar(bruto)
            salida.append({
                "question": d["question"],
                "reference": d.get("answer", ""),
                "prediction": pred,
                "prediction_raw": bruto,
                "technique": tecnica,
                "system": "rag",
                "source_file": d.get("source_file", ""),
                "section_label": d.get("section_label", ""),
                "retrieved_sources": [h["source_file"] for h in hits],
                "retrieved_chunk_ids": [h["chunk_id"] for h in hits],
            })

    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in salida) + "\n",
                       encoding="utf-8")
    errores = sum(1 for x in salida if x["prediction"].startswith("__ERROR__"))
    print(f"\n{len(salida)} filas -> {destino}  ({errores} con error)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
