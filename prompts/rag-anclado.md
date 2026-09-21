# Tecnica: RAG anclado

Prompt de generacion del sistema RAG. Antepone a la pregunta los k fragmentos recuperados (numerados, con el nombre de la norma) y le exige al modelo responder **solo** con ellos, citar el fragmento por su numero y declarar 'I do not have enough information.' cuando la respuesta no este en el contexto.

Ademas de esta variante estricta, las tres tecnicas del taller (zero-shot, few-shot y chain-of-thought) se aplican a la etapa de generacion del RAG con el mismo texto que en el modelo base mas el bloque de contexto. Se distinguen en `results/respuestas-rag.jsonl` por el campo `technique`; en los cuatro casos `system` es `rag`.

Los marcadores `{context}` y `{q}` se reemplazan por los fragmentos y la pregunta. Las etiquetas `<start_of_turn>` y `<end_of_turn>` son el chat template de Gemma. El texto operativo vive en `src/rag/prompts.py`; este archivo se regenera con `python src/rag/prompts.py --export`.

## Formato del contexto

```text
[1] (ISO 3746) <texto del chunk>

[2] (BS 4142) <texto del chunk>
```

## Prompt exacto: Variante estricta (technique = rag-anclado)

```text
<start_of_turn>user
You are an expert on ISO, UNE and BS acoustics standards for noise measurement. Answer the question using ONLY the excerpts below. Quote the exact passage you used and cite it by its number, e.g. [2]. If the answer is not in the excerpts, reply exactly "I do not have enough information."

Context:
{context}

Question: {q}<end_of_turn>
<start_of_turn>model

```

## Prompt exacto: RAG + zero-shot

```text
<start_of_turn>user
You are an expert on ISO, UNE and BS acoustics standards for noise measurement. Answer the question precisely and cite the clause when you know it.

Use ONLY the numbered excerpts below, taken from the standards. If the answer is not in the excerpts, reply exactly "I do not have enough information."

Context:
{context}

Question: {q}<end_of_turn>
<start_of_turn>model

```

## Prompt exacto: RAG + few-shot

```text
<start_of_turn>user
You are an expert on ISO, UNE and BS acoustics standards for noise measurement.

Example 1
Question: What is the reference sound pressure used in ISO 1996-1?
Answer: The reference sound pressure is 20 microPa, and sound pressure is expressed in pascals.

Example 2
Question: What instrumentation class does ISO 3746 require?
Answer: The instrumentation system shall meet IEC 61672-1:2002 class 2, although class 1 is recommended.

Use ONLY the numbered excerpts below, taken from the standards. If the answer is not in the excerpts, reply exactly "I do not have enough information."

Context:
{context}

Now answer in the same style.
Question: {q}<end_of_turn>
<start_of_turn>model

```

## Prompt exacto: RAG + chain-of-thought

```text
<start_of_turn>user
You are an expert on ISO, UNE and BS acoustics standards for noise measurement.
Think step by step: first identify which excerpt and which clause the question refers to, then recall what that clause requires, then state the final answer.
Finish with a line starting with 'Answer:' that contains only the final answer.

Use ONLY the numbered excerpts below, taken from the standards. If the answer is not in the excerpts, reply exactly "I do not have enough information."

Context:
{context}

Question: {q}<end_of_turn>
<start_of_turn>model

```
