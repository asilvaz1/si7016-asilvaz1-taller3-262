#!/usr/bin/env python
"""Carga el corpus procesado y lo parte en chunks con id estable.

Solo entran las normas con estado 'fuente' en data/qa/corpus_map.csv y cuyo
.txt exista en data/processed/. Las descartadas (duplicados, ediciones
superadas, erratas) se dejan fuera para no contaminar la recuperacion.

Cada chunk lleva el id `<source_file>#<indice de 4 digitos>` que exige el
contrato del HANDOFF (seccion 4.3), donde source_file es el nombre del PDF tal
como aparece en el dataset QA.
"""
import csv
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

RAIZ = Path(__file__).resolve().parents[2]
PROCESSED = RAIZ / "data" / "processed"
CORPUS_MAP = RAIZ / "data" / "qa" / "corpus_map.csv"
INVENTARIO = RAIZ / "data" / "qa" / "inventario_corpus.csv"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def limpiar(texto: str) -> str:
    """pdftotext -layout deja columnas de espacios y saltos de pagina; se
    compactan para que el chunk no gaste caracteres en relleno."""
    texto = texto.replace("\f", "\n").replace("\r", "")
    texto = re.sub(r"[ \t]{2,}", " ", texto)
    texto = re.sub(r" +\n", "\n", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def documentos_fuente():
    """Devuelve [(source_file, norma, ruta_txt)] de las normas fuente que
    tienen texto disponible, y la lista de PDF fuente sin texto."""
    txt_de = {r["archivo_pdf"]: r["archivo_txt"]
              for r in csv.DictReader(open(INVENTARIO, encoding="utf-8"))}
    docs, faltan = [], []
    for r in csv.DictReader(open(CORPUS_MAP, encoding="utf-8")):
        if r["estado"] != "fuente":
            continue
        ruta = PROCESSED / txt_de[r["archivo_pdf"]]
        if ruta.exists():
            docs.append((r["archivo_pdf"], r["norma"], ruta))
        else:
            faltan.append(r["archivo_pdf"])
    return docs, faltan


def hacer_chunks(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""])
    docs, faltan = documentos_fuente()
    chunks = []
    for source_file, norma, ruta in docs:
        texto = limpiar(ruta.read_text(encoding="utf-8", errors="replace"))
        for i, parte in enumerate(splitter.split_text(texto)):
            chunks.append({
                "chunk_id": f"{source_file}#{i:04d}",
                "source_file": source_file,
                "norma": norma,
                "text": parte,
            })
    return chunks, faltan
