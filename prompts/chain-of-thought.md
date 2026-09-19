# Tecnica: Chain-of-thought

Se le pide al modelo razonar por pasos antes de responder: primero identificar la norma y el apartado, luego recordar que exige ese apartado, y al final dar la respuesta en una linea que empieza con 'Answer:'. Mide si el razonamiento explicito ayuda en un dominio donde la respuesta correcta depende de ubicar la clausula adecuada.

Se aplica igual al modelo base, al modelo afinado y a la etapa de
generacion del RAG, para que la comparacion sea sobre lo mismo.

El marcador `{q}` se reemplaza por la pregunta. Las etiquetas
`<start_of_turn>` y `<end_of_turn>` son el chat template de Gemma.

## Prompt exacto

```text
<start_of_turn>user
You are an expert on ISO, UNE and BS acoustics standards for noise measurement.
Think step by step: first identify which standard and which clause the question refers to, then recall what that clause requires, then state the final answer.
Finish with a line starting with 'Answer:' that contains only the final answer.

Question: {q}<end_of_turn>
<start_of_turn>model

```
