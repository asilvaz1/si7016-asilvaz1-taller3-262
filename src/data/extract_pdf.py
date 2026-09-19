#!/usr/bin/env python3
"""Extrae el texto de las normas ISO/UNE del taller 3 y genera un inventario.

Uso:
    python src/data/extract_pdf.py \
        --src "../talleres/taller3/NORMAS ESTANDARES" \
        --out data/processed \
        --inventory data/qa/inventario_corpus.csv

Estrategia: pdftotext -layout (poppler) como motor principal, pypdf como
respaldo. Las normas con muy pocos caracteres por pagina se marcan como
candidatas a OCR (PDF escaneado o protegido).
"""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

UMBRAL_CHARS_POR_PAGINA = 200  # por debajo de esto, se sospecha PDF escaneado


def slug(nombre: str) -> str:
    s = re.sub(r"\.pdf$", "", nombre, flags=re.I)
    s = re.sub(r"[^\w\s.-]", "", s)
    s = re.sub(r"[\s_]+", "_", s.strip())
    return s.lower()


def extraer_pdftotext(pdf: Path) -> str | None:
    try:
        r = subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf), "-"],
            capture_output=True, timeout=180,
        )
        if r.returncode == 0:
            return r.stdout.decode("utf-8", errors="replace")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def extraer_pypdf(pdf: Path) -> tuple[str | None, int]:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf))
        paginas = [(p.extract_text() or "") for p in reader.pages]
        return "\n".join(paginas), len(reader.pages)
    except Exception as e:  # noqa: BLE001
        print(f"  ! pypdf fallo en {pdf.name}: {e}", file=sys.stderr)
        return None, 0


def contar_paginas(pdf: Path) -> int:
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(pdf)).pages)
    except Exception:  # noqa: BLE001
        return 0


def limpiar(texto: str) -> str:
    texto = texto.replace("\x0c", "\n")
    texto = re.sub(r"[ \t]+\n", "\n", texto)
    texto = re.sub(r"\n{4,}", "\n\n\n", texto)
    return texto.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--inventory", default="data/qa/inventario_corpus.csv")
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Path(args.inventory).parent.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(src.glob("*.pdf"))
    if not pdfs:
        print(f"No se encontraron PDF en {src}", file=sys.stderr)
        return 1

    filas = []
    for pdf in pdfs:
        print(f"-> {pdf.name}")
        texto = extraer_pdftotext(pdf)
        motor = "pdftotext"
        paginas = contar_paginas(pdf)
        if not texto or len(texto.strip()) < 100:
            texto, paginas_pypdf = extraer_pypdf(pdf)
            motor = "pypdf"
            paginas = paginas or paginas_pypdf
        texto = limpiar(texto or "")

        destino = out / f"{slug(pdf.name)}.txt"
        destino.write_text(texto, encoding="utf-8")

        chars = len(texto)
        cpp = chars / paginas if paginas else 0
        filas.append({
            "archivo_pdf": pdf.name,
            "archivo_txt": destino.name,
            "paginas": paginas,
            "caracteres": chars,
            "chars_por_pagina": round(cpp, 1),
            "motor": motor,
            "requiere_ocr": "SI" if cpp < UMBRAL_CHARS_POR_PAGINA else "no",
        })

    with open(args.inventory, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    ocr = [f["archivo_pdf"] for f in filas if f["requiere_ocr"] == "SI"]
    print(f"\n{len(filas)} normas procesadas -> {out}")
    print(f"Inventario -> {args.inventory}")
    if ocr:
        print(f"\nATENCION: {len(ocr)} PDF con poco texto extraible (posible escaneo):")
        for n in ocr:
            print(f"  - {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
