# Tecnica: Zero-shot

Se le da al modelo solo el rol de experto y la pregunta, sin ningun ejemplo resuelto. Es la linea base mas pura: mide lo que el modelo sabe del dominio sin ayuda de contexto.

Se aplica igual al modelo base, al modelo afinado y a la etapa de
generacion del RAG, para que la comparacion sea sobre lo mismo.

El marcador `{q}` se reemplaza por la pregunta. Las etiquetas
`<start_of_turn>` y `<end_of_turn>` son el chat template de Gemma.

## Prompt exacto

```text
<start_of_turn>user
You are an expert on ISO, UNE and BS acoustics standards for noise measurement. Answer the question precisely and cite the clause when you know it.

Question: {q}<end_of_turn>
<start_of_turn>model

```
