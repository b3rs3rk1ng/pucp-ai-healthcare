# EEG Motor Imagery: Reducción de Electrodos

**Kevin Antonio Navarro Carrera** — Código 20193292 — AI in Healthcare, PUCP 2026-I

---

## 1. Problema

Clasificar motor imagery (imaginar mover mano izquierda vs derecha) a partir de señales EEG, reduciendo sistemáticamente de 64 a 2 electrodos. El objetivo es determinar el mínimo número de electrodos viable para un dispositivo BCI wearable, aplicable a pacientes con parálisis o pérdida de movilidad.

## 2. Dataset

PhysioNet EEG Motor Movement/Imagery (eegmmidb): 104 sujetos, 64 canales EEG, 160 Hz. Se usaron los runs 4, 8 y 12 (motor imagery izq/der), totalizando 4,470 trials balanceados (50.4% izq / 49.6% der).

## 3. Preprocesamiento

Bandpass 8-30 Hz (mu + beta), epoching 0-4s, rechazo de artefactos >500µV, z-score por época y canal.

## 4. Modelos

Se compararon 5 modelos representando paradigmas diferentes (implementados via braindecode y MNE+sklearn):

| Modelo | Tipo | Parámetros |
|--------|------|------------|
| CSP+LDA | Clásico | N/A |
| EEGNet | CNN compacta | ~2K |
| ShallowConvNet | CNN shallow | ~100K |
| ATCNet | Atención + TCN | ~113K |
| EEG Conformer | Transformer | ~100-200K |

## 5. Evaluación

**Cross-patient**: 5-fold GroupKFold por sujeto — ningún sujeto aparece en train y test simultáneamente. Métricas: accuracy, Cohen's Kappa, ROC-AUC. Validación con **permutation test** (20 permutaciones, p < 0.05).

## 6. Resultados

Reducción de electrodos (accuracy por modelo y configuración):

| Modelo | 64ch | 32ch | 16ch | 8ch | 4ch | 2ch |
|--------|------|------|------|-----|-----|-----|
| ATCNet | **0.751** | 0.647 | 0.616 | 0.635 | 0.547 | 0.532 |
| EEGNet | 0.748 | 0.622 | 0.621 | 0.602 | 0.594 | 0.478 |
| ShallowConvNet | 0.734 | 0.661 | 0.659 | **0.636** | **0.586** | **0.574** |
| Conformer | 0.642 | 0.622 | 0.589 | 0.528 | 0.571 | 0.543 |
| CSP+LDA | 0.587 | 0.585 | 0.570 | 0.566 | 0.553 | 0.534 |

Permutation test (p < 0.05): todos pasan excepto EEGNet a 2ch (p=0.857).

## 7. SHAP — Channel Importance

SHAP (GradientExplainer) sobre 4 modelos reveló que **F7 y F8** (frontales laterales) dominan sobre C3/C4 (motor cortex) en 3 de 4 modelos:

| Modelo | Top 5 SHAP | C3 rank | C4 rank |
|--------|-----------|---------|---------|
| EEGNet | F7, F8, T9, T10, F4 | #19/64 | #33/64 |
| ShallowConvNet | F7, F8, T10, Fp2, Fpz | #16/64 | #25/64 |
| ATCNet | F8, F7, P4, P5, T10 | #12/64 | #40/64 |
| Conformer | PO3, TP8, FC6, Fp1, C5 | #49/64 | #9/64 |

**Verificación con ICA**: se aplicó ICA para remover componentes oculares y se re-corrió SHAP:

| | Sin ICA (top 5) | Con ICA (top 5) |
|--|-----------------|-----------------|
| EEGNet | F7, F8, T9, T10, F4 | FC2, **F8**, **Cz**, F2, **CP4** |

- **F7 bajó de #1 a #61** — era artefacto ocular
- **F8 se mantuvo en #2** — señal frontal legítima
- **Cz y CP4 subieron al top 5** — canales motores emergen tras limpiar artefactos

La selección anatómica tradicional (C3/C4) no es óptima. La selección data-driven post-ICA sugiere FC2, F8, Cz como los canales más informativos.

## 8. Conclusiones

- **ATCNet** domina con 64 canales (75.1%); **ShallowConvNet** domina con pocos canales (57.4% a 2ch)
- El modelo óptimo depende del número de electrodos — no existe un modelo universalmente superior
- SHAP + ICA revelan que F7 era artefacto ocular, pero F8 y canales motores (Cz, CP4, FC2) son señal legítima
- Límite práctico validado: **4 canales** (todos los modelos pasan permutation test, p < 0.05)
- Evaluación **cross-patient** (GroupKFold por sujeto): ningún sujeto en train y test simultáneamente

Código y modelos entrenados: https://github.com/b3rs3rk1ng/pucp-ai-healthcare
