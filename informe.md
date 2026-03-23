# EEG Motor Imagery: Electrode Reduction Experiment

**Kevin Antonio Navarro Carrera** — Código 20193292 — AI in Healthcare, PUCP 2026-I

---

## 1. Problem Definition

**What**: Classify left vs right hand motor imagery from EEG signals, systematically reducing from 64 to 2 electrodes to determine the minimum viable channel set for a wearable BCI device.

**Why**: A 64-electrode laboratory cap is impractical for real-world BCI applications. Patients with paralysis or motor disabilities need portable devices with minimal electrodes. Determining how few electrodes are needed while maintaining classification accuracy has direct implications for wearable BCI design.

## 2. Dataset Description

PhysioNet EEG Motor Movement/Imagery (eegmmidb): 104 subjects, 64 EEG channels, 160 Hz. Runs 4, 8, and 12 (motor imagery left/right hand), totaling 4,470 balanced trials (50.4% left / 49.6% right). Cross-patient evaluation: 5-fold GroupKFold by subject — no subject appears in both train and test simultaneously.

## 3. Preprocessing Pipeline

Bandpass filter 8–30 Hz (mu + beta motor rhythms), epoching 0–4s from stimulus onset, artifact rejection >500µV peak-to-peak, z-score normalization per epoch and channel. ICA was applied separately to verify channel importance (SHAP analysis, Section 5).

## 4. Model Selection & Justification

Five models representing different algorithmic paradigms were compared (DL models via braindecode, CSP+LDA via MNE+sklearn):

| Model | Type | Params |
|-------|------|--------|
| CSP+LDA | Classical | N/A |
| EEGNet | Compact CNN | ~2K |
| ShallowConvNet | Shallow CNN | ~100K |
| ATCNet | Attention+TCN | ~113K |
| EEG Conformer | Transformer | ~100-200K |

CSP+LDA as classical baseline. EEGNet was designed for EEG-BCI (Lawhern et al., 2018). ShallowConvNet resists overfitting with limited data. ATCNet uses temporal attention to select informative moments. Conformer applies self-attention across timepoints. Multiple models were chosen to test whether the optimal architecture depends on electrode count.

## 5. Evaluation Metrics

**Accuracy** (primary), **Cohen's Kappa** (chance-corrected), **ROC-AUC** (threshold-independent). All results validated with **permutation test** (20 permutations, p < 0.05) — the gold standard for verifying that a model learned real patterns rather than noise.

**Results — Electrode reduction** (accuracy per model and configuration):

| Model | 64ch | 32ch | 16ch | 8ch | 4ch | 2ch |
|-------|------|------|------|-----|-----|-----|
| ATCNet | **0.751** | 0.647 | 0.616 | 0.635 | 0.547 | 0.532 |
| EEGNet | 0.748 | 0.622 | 0.621 | 0.602 | 0.594 | 0.478 |
| ShallowConvNet | 0.734 | 0.661 | 0.659 | **0.636** | **0.586** | **0.574** |
| Conformer | 0.642 | 0.622 | 0.589 | 0.528 | 0.571 | 0.543 |
| CSP+LDA | 0.587 | 0.585 | 0.570 | 0.566 | 0.553 | 0.534 |

Permutation test (p < 0.05): all pass except EEGNet at 2ch (p=0.857).

**SHAP channel importance**: SHAP (GradientExplainer) across 4 models revealed F7/F8 (frontal lateral) as most important — not C3/C4 (motor cortex). ICA verification confirmed F7 was an ocular artifact (#1 → #61 after ICA removal), while F8 remained legitimate (#2). Post-ICA top channels: FC2, F8, Cz — a mix of frontal and motor cortex signals.

**Motor execution vs imagery**: Execution (runs 3,7,11) did not outperform imagery (runs 4,8,12) in cross-patient evaluation (0.744 vs 0.755 at 64ch). Imagery patterns are more standardized across subjects than real movement patterns.

**Key findings**:

- ATCNet dominates at 64 channels (75.1%); ShallowConvNet dominates at few channels (57.4% at 2ch)
- The optimal model depends on electrode count — no single model is universally superior
- Practical validated limit: **4 channels** (all models pass permutation test)
- Anatomical channel selection (C3/C4) is not optimal — data-driven selection via SHAP+ICA yields better candidates

## 6. Code

**Model selection** (all 5 models, swap in 1 line via braindecode):

```python
from braindecode.models import (EEGNetv4,
  ShallowFBCSPNet, ATCNet, EEGConformer)
from mne.decoding import CSP  # classical
from sklearn.discriminant_analysis import (
  LinearDiscriminantAnalysis as LDA)

models = {
  'EEGNet':    EEGNetv4(n_chans=nc, n_outputs=2, n_times=nt),
  'Shallow':   ShallowFBCSPNet(n_chans=nc, n_outputs=2, n_times=nt),
  'ATCNet':    ATCNet(n_chans=nc, n_outputs=2, n_times=nt),
  'Conformer': EEGConformer(n_chans=nc, n_outputs=2, n_times=nt),
  'CSP+LDA':   Pipeline([('csp', CSP()), ('lda', LDA())]),
}
```

**Cross-patient evaluation** (no data leakage):

```python
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups=subject_ids):
    model.fit(X[train_idx], y[train_idx])
    acc = model.score(X[test_idx], y[test_idx])
```

**SHAP channel importance + ICA verification**:

```python
import shap
explainer = shap.GradientExplainer(model, X_background)
shap_values = explainer.shap_values(X_test)
importance = np.abs(shap_values).mean(axis=(0, 1, 3))
top_channels = np.argsort(importance)[::-1]
```

Full repository: [github.com/b3rs3rk1ng/pucp-ai-healthcare](https://github.com/b3rs3rk1ng/pucp-ai-healthcare)

## References

- Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation, 101*(23), e215–e220. https://doi.org/10.1161/01.CIR.101.23.e215
- Schalk, G., et al. (2004). BCI2000: A general-purpose brain-computer interface system. *IEEE Trans. Biomed. Eng., 51*(6), 1034–1043. https://doi.org/10.1109/TBME.2004.827072
- Lawhern, V. J., et al. (2018). EEGNet: A compact CNN for EEG-based brain–computer interfaces. *J. Neural Eng., 15*(5), 056013. https://doi.org/10.1088/1741-2552/aace8c
