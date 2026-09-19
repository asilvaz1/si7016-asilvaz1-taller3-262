# Dataset QA: Normas estándares de medición de ruido

Dataset instructivo (pregunta-respuesta) para el fine-tuning y la evaluación
del taller 3. 100 pares QA anclados al texto real de las normas.

## Archivos

| Archivo | Contenido |
| --- | --- |
| `normas_ruido_all.jsonl` | Los 100 pares QA completos |
| `normas_ruido_train.jsonl` | 80 pares para entrenamiento |
| `normas_ruido_eval.jsonl` | 20 pares para evaluación (split estratificado por categoría, seed 42) |
| `inventario_corpus.csv` | Páginas, caracteres y motor de extracción de los 41 PDF |
| `corpus_map.csv` | Norma → archivo fuente elegido, categoría, estado y preguntas asignadas |
| `cobertura.csv` | Preguntas efectivas por categoría y por norma |
| `parts/cat0*.jsonl` | Los pares QA agrupados por categoría (fuente editable) |

## Esquema

Mismo formato del ejemplo del repo del curso
(`apoyo-al-taller3/pensamiento_computacional_train.jsonl`):

```json
{
  "question": "What instrumentation does ISO 1996-2 require for environmental noise measurements?",
  "answer": "The instrumentation system ... class 1 or class 2 as specified in IEC 61672-1:2002 ...",
  "source_file": "UNE ISO 1996 - 2 .pdf",
  "source_doc_title": "ISO 1996-2:2007 - Acoustics: Description, measurement and assessment of environmental noise. Part 2: ...",
  "section_kind": "requirement",
  "section_label": "5.1"
}
```

`section_kind` toma uno de: `scope`, `requirement`, `definition`, `procedure`,
`formula`, `table`. `section_label` es el número de capítulo, apartado, tabla o
anexo de donde sale la respuesta, para que cada par sea trazable al PDF.

## Idioma

Todo el dataset está en inglés, incluidas las preguntas sobre las normas UNE en
español. Razones: `gemma-7b-it` rinde mejor en inglés, el dataset queda
homogéneo, y ROUGE no mezcla idiomas en la evaluación. Los símbolos y términos
técnicos se conservan en su forma original (LAeq, Lden, LW, Rw, Ln,w, STI).

## Reparto de las 100 preguntas

| Cat. | Categoría | Normas fuente | Preguntas |
| --- | --- | --- | --- |
| cat01 | Ruido ambiental: medición, descriptores y límites | ISO 1996-1, -2, -3 | 15 |
| cat02 | Propagación del sonido al aire libre | ISO 9613-1, -2 | 10 |
| cat03 | Potencia sonora de fuentes | ISO 3744, 3745, 3746, ISO 8297 | 15 |
| cat04 | Ruido industrial y evaluación de molestia | BS 4142 | 5 |
| cat05 | Aislamiento acústico en edificaciones | ISO 10140-1..5, 140-18, 16283-1..3, 717-1/-2, 10534-1/-2, 354, 16032 | 25 |
| cat06 | Acústica de salas y reverberación | ISO 3382-1, -2, -3 | 10 |
| cat07 | Exposición ocupacional y protección auditiva | ISO 9612, 9921, 4869-3 | 10 |
| cat08 | Modelos de cálculo: eventos impulsivos y ruido de tránsito | ISO 13474, CNOSSOS-EU | 10 |

Nota sobre cat08: la hoja de ruta la llamaba "modelos de ruido de tránsito",
pero ISO 13474 trata de la distribución estadística de niveles de exposición
sonora de eventos impulsivos (voladuras, artillería), no de tránsito. La
categoría se renombró para que el título describa su contenido real.

## Criterio de duplicados

La carpeta `NORMAS ESTANDARES` tiene 41 PDF, no 43. De esos 41, **33 se usan
como fuente** y **8 se descartan** para no generar preguntas contradictorias
sobre el mismo contenido. `corpus_map.csv` documenta cada decisión:

- **ISO 1996-1**: se usa la UNE-ISO 1996-1:2005 (= ISO 1996-1:2003, publicada).
  El archivo BSOL es un *Draft for Public Comment* de 2013, no una norma vigente.
- **ISO 1996-2**: se usa la UNE-EN ISO 1996-2:2009 (= ISO 1996-2:2007). El BSOL
  es BS 7445-2:1991 (= ISO 1996-2:**1987**), una edición anterior.
- **ISO 3744**: se usa la tercera edición de 2010. El otro archivo es la segunda
  edición de 1994, superada.
- **ISO 3746**: se usa la versión inglesa BS EN ISO 3746:2010; la UNE-EN es la
  misma edición en español.
- **Modificaciones y erratas** (10140-1 A1, 10140-3 A1, 3382-2 V2, 9612 ERRATUM):
  se descartan como fuente independiente; su contenido pertenece a la norma base.

## Reproducir

```bash
python src/data/extract_pdf.py \
    --src "../talleres/taller3/NORMAS ESTANDARES" \
    --out data/processed \
    --inventory data/qa/inventario_corpus.csv

python src/data/build_qa_dataset.py --seed 42
```

`build_qa_dataset.py` valida el esquema, verifica que cada `source_file` tenga
su texto extraído en `data/processed/`, detecta preguntas duplicadas y
respuestas demasiado cortas, y falla con código 2 si algo no cuadra.
