# Curva de entrenamiento - fine-tuning de gemma-7b-it

Transcrita de Cloud Logging del Custom Training Job de Vertex AI del
2026-09-20 (`gemma-7b-it-normas-ruido-lora`). Se guarda aqui porque los
checkpoints intermedios vivian en el disco efimero del contenedor y los logs
de GCP no se conservan indefinidamente.

- Modelo base: `google/gemma-7b-it`, QLoRA 4-bit NF4
- Dataset: 80 pares pregunta-respuesta de normas ISO/UNE/BS de medicion de ruido
- LoRA r=16, alpha=32, dropout=0.05
- Lote efectivo 8, 10 epocas, 100 pasos, learning rate 2e-4 con scheduler cosine
- Duracion: 847 s (~14 min) en 1 x NVIDIA L4
- `train_loss` promedio reportado por el Trainer: 1.1671

| Época | Loss | grad_norm |
| --- | --- | --- |
| 0.5 | 7.3347 | 13.31 |
| 1.0 | 3.3561 | 21.18 |
| 1.5 | 2.6242 | 15.85 |
| 2.0 | 1.9162 | 18.25 |
| 2.5 | 1.6011 | 18.58 |
| 3.0 | 1.3425 | 25.69 |
| 3.5 | 1.0168 | 19.73 |
| 4.0 | 0.8420 | 19.85 |
| 4.5 | 0.5321 | 17.12 |
| 5.0 | 0.5087 | 16.85 |
| 5.5 | 0.3564 | 11.49 |
| 6.0 | 0.3537 | 10.49 |
| 6.5 | 0.2476 | 3.32 |
| 7.0 | 0.2415 | 4.01 |
| 7.5 | 0.1958 | 3.20 |
| 8.0 | 0.1998 | 7.43 |
| 8.5 | 0.1975 | 7.70 |
| 9.0 | 0.1563 | 3.56 |
| 9.5 | 0.1614 | 2.94 |
| 10.0 | 0.1585 | 0.00 |

## Lectura de la curva

El descenso es limpio y monotono hasta la epoca 6.5, donde el loss cruza 0.25 y
la norma del gradiente cae de ~10 a ~3. De ahi en adelante la curva se aplana:
entre las epocas 7 y 10 el loss solo baja de 0.24 a 0.16.

Esa parte plana es memorizacion, no aprendizaje. Con 80 ejemplos, un loss por
debajo de 0.2 significa que el modelo practicamente reproduce de memoria las
respuestas de entrenamiento. El aprendizaje util ocurrio entre las epocas 1 y 6.

Consecuencia para la evaluacion: las 20 preguntas del split de evaluacion nunca
se usaron en el entrenamiento, asi que ROUGE sobre ese conjunto mide
generalizacion real y no memorizacion. Si el modelo afinado superara al base en
las preguntas de entrenamiento pero no en las de evaluacion, la causa seria este
sobreajuste, y la correccion es entrenar 5 o 6 epocas en vez de 10.

El `grad_norm` de 0.00 en el ultimo paso es normal: el scheduler cosine deja el
learning rate en 2.4e-09, practicamente cero.
