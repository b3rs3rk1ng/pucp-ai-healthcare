# EEG Motor Imagery: Electrode Reduction Experiment

Clasificación de motor imagery (imaginar mover mano izquierda vs derecha) a partir de señales EEG, con reducción sistemática de electrodos de 64 a 2 canales.

**Curso**: AI in Healthcare — PUCP
**Dataset**: [PhysioNet EEG Motor Movement/Imagery](https://physionet.org/content/eegmmidb/1.0.0/) (109 sujetos, 64 canales, 160 Hz)

## Notebook principal

[`eeg_electrode_reduction.ipynb`](eeg_electrode_reduction.ipynb) — contiene el experimento completo: preprocesamiento, arquitecturas, resultados, permutation test y discusión con 10 referencias.

## Modelos comparados

| Modelo | Tipo | Parámetros | Referencia |
|--------|------|------------|------------|
| CSP+LDA | Clásico | N/A | Blankertz et al. |
| EEGNet | CNN compacta | ~2K | Lawhern et al., 2018 |
| ShallowConvNet | CNN shallow | ~100K | Schirrmeister et al., 2017 |
| ATCNet | Atención + TCN | ~113K | Altaheri et al., 2022 |
| EEG Conformer | Transformer | ~100-200K | Song et al., 2022 |

## Resultados

```
Canales   ATCNet   EEGNet   ShallowConv   Conformer   CSP+LDA
   64      0.742    0.733      0.725        0.611       0.590
   32      0.651    0.652      0.658        0.655       0.589
   16      0.639    0.640      0.647        0.584       0.581
    8      0.628    0.572      0.635        0.595       0.587
    4      0.584    0.583      0.599        0.589       0.551
    2      0.548    0.550      0.563        0.548       0.508
```

- ATCNet domina con 64 canales (74.2%)
- ShallowConvNet domina con pocos canales (56.3% con solo C3+C4)
- Solo ShallowConvNet y Conformer pasan el permutation test a 2 canales
- El límite práctico validado estadísticamente es 4 canales (C3, Cz, C4, CPz)

## Validación

Todos los resultados verificados con **permutation test** (labels reales vs barajados). Evaluación cross-subject con **GroupKFold** por sujeto (sin data leakage).

## Estructura

```
src/
  preprocess.py    # Preprocesamiento: filtrado, epoching, normalización
  model.py         # Arquitectura EEGNet
  train.py         # Entrenamiento y evaluación
  predict.py       # Inferencia CLI
  channels.py      # Configuraciones de canales
models/            # Pesos entrenados (.pt)
modal_deploy.py    # Entrenamiento remoto (Modal GPU)
```

## Infraestructura

Entrenamiento en GPUs remotas via [Modal](https://modal.com) con checkpointing por configuración de canales. Los 5 modelos corren en paralelo (4 GPU + 1 CPU para CSP+LDA).
