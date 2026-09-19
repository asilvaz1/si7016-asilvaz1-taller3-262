# Tecnica: Few-shot

Se anteponen dos pares pregunta-respuesta resueltos, tomados del dataset de entrenamiento, para fijar el formato y el nivel de detalle esperado. Mide cuanto mejora la respuesta solo por ver ejemplos del estilo, sin que el modelo aprenda contenido nuevo.

Se aplica igual al modelo base, al modelo afinado y a la etapa de
generacion del RAG, para que la comparacion sea sobre lo mismo.

El marcador `{q}` se reemplaza por la pregunta. Las etiquetas
`<start_of_turn>` y `<end_of_turn>` son el chat template de Gemma.

## Prompt exacto

```text
<start_of_turn>user
You are an expert on ISO, UNE and BS acoustics standards for noise measurement.

Example 1
Question: What is the reference sound pressure used in ISO 1996-1?
Answer: The reference sound pressure is 20 microPa, and sound pressure is expressed in pascals.

Example 2
Question: What instrumentation class does ISO 3746 require?
Answer: The instrumentation system shall meet IEC 61672-1:2002 class 2, although class 1 is recommended.

Now answer in the same style.
Question: {q}<end_of_turn>
<start_of_turn>model

```
