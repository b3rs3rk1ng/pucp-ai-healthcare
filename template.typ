#set page(margin: 1.5cm, paper: "a4")
#set text(size: 9pt)
#set par(justify: true)
#set heading(numbering: "1.")

#let title = "EEG Motor Imagery: Electrode Reduction Experiment"
#let author = "Kevin Antonio Navarro Carrera — Código 20193292 — AI in Healthcare, PUCP 2026-I"

#align(center)[
  #text(size: 14pt, weight: "bold")[#title]
  #v(4pt)
  #text(size: 9pt)[#author]
  #v(8pt)
  #line(length: 100%, stroke: 0.5pt)
]

#v(8pt)

#columns(2, gutter: 12pt)[

= Problem Definition

*What*: Classify left vs right hand motor imagery from EEG signals, systematically reducing from 64 to 2 electrodes to determine the minimum viable channel set for a wearable BCI device.

*Why*: A 64-electrode laboratory cap is impractical for real-world BCI. Patients with paralysis need portable devices with minimal electrodes. Determining how few electrodes are needed while maintaining accuracy has direct implications for wearable BCI design.

= Dataset Description

PhysioNet EEG Motor Movement/Imagery (eegmmidb): 104 subjects, 64 EEG channels, 160 Hz. Runs 4, 8, and 12 (motor imagery left/right hand), totaling 4,470 balanced trials (50.4% left / 49.6% right). Cross-patient evaluation: 5-fold GroupKFold by subject — no subject appears in both train and test.

= Preprocessing Pipeline

Bandpass filter 8–30 Hz (mu + beta motor rhythms), epoching 0–4s from stimulus onset, artifact rejection >500µV peak-to-peak, z-score normalization per epoch and channel. ICA applied separately to verify channel importance (SHAP analysis).

= Model Selection & Justification

Five models representing different paradigms (DL via braindecode, CSP+LDA via MNE+sklearn):

#table(
  columns: (auto, auto, auto),
  align: (left, left, right),
  stroke: none,
  table.hline(),
  [*Model*], [*Type*], [*Params*],
  table.hline(),
  [CSP+LDA], [Classical], [N/A],
  [EEGNet], [Compact CNN], [\~2K],
  [ShallowConvNet], [Shallow CNN], [\~100K],
  [ATCNet], [Attention+TCN], [\~113K],
  [EEG Conformer], [Transformer], [\~100-200K],
  table.hline(),
)

CSP+LDA as classical baseline. EEGNet designed for EEG-BCI (Lawhern et al., 2018). ShallowConvNet resists overfitting with limited data. ATCNet uses temporal attention. Conformer applies self-attention across timepoints. Multiple models test whether the optimal architecture depends on electrode count.

= Evaluation Metrics

*Accuracy* (primary), *Cohen's Kappa* (chance-corrected), *ROC-AUC* (threshold-independent). All validated with *permutation test* (20 permutations, p < 0.05).

*Results — Electrode reduction:*

#text(size: 8pt)[
#table(
  columns: (auto, auto, auto, auto, auto, auto, auto),
  align: (left, right, right, right, right, right, right),
  stroke: none,
  table.hline(),
  [*Model*], [*64*], [*32*], [*16*], [*8*], [*4*], [*2*],
  table.hline(),
  [ATCNet], [*.751*], [.647], [.616], [.635], [.547], [.532],
  [EEGNet], [.748], [.622], [.621], [.602], [.594], [.478],
  [Shallow], [.734], [.661], [.659], [*.636*], [*.586*], [*.574*],
  [Conformer], [.642], [.622], [.589], [.528], [.571], [.543],
  [CSP+LDA], [.587], [.585], [.570], [.566], [.553], [.534],
  table.hline(),
)
]

Permutation test (p < 0.05): all pass except EEGNet at 2ch (p=0.857).

*SHAP channel importance*: F7/F8 (frontal lateral) dominate over C3/C4 (motor cortex) in 3/4 models. ICA confirmed F7 was ocular artifact (#1→#61 after removal), F8 remained legitimate (#2). Post-ICA top channels: FC2, F8, Cz.

*Motor execution vs imagery*: Execution did not outperform imagery in cross-patient evaluation (0.744 vs 0.755 at 64ch). Imagery patterns are more standardized across subjects.

*Key findings*:
- ATCNet dominates at 64ch (75.1%); ShallowConvNet at few channels (57.4% at 2ch)
- Optimal model depends on electrode count — no single model is universally superior
- Practical validated limit: *4 channels* (all models pass permutation test)
- Anatomical selection (C3/C4) is not optimal — SHAP+ICA yields better candidates

Full repository: #link("https://github.com/b3rs3rk1ng/pucp-ai-healthcare")[github.com/b3rs3rk1ng/pucp-ai-healthcare]

#heading(numbering: none)[References]

#text(size: 8pt)[
- Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. _Circulation, 101_(23), e215–e220.
- Schalk, G., et al. (2004). BCI2000: A general-purpose BCI system. _IEEE Trans. Biomed. Eng., 51_(6), 1034–1043.
- Lawhern, V. J., et al. (2018). EEGNet: A compact CNN for EEG-based BCI. _J. Neural Eng., 15_(5), 056013.
]

]

#pagebreak()

#align(center)[#text(size: 12pt, weight: "bold")[6. Python Code]]

#v(8pt)

#block(
  fill: luma(245),
  inset: 8pt,
  radius: 4pt,
  width: 100%,
)[
*Model selection* — all 5 models, swap in 1 line:
```python
# Deep learning models (braindecode library)
from braindecode.models import (EEGNetv4, ShallowFBCSPNet, ATCNet, EEGConformer)
# Classical model (MNE + sklearn)
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# nc = number of channels (64, 32, 16, 8, 4, or 2)
# nt = number of timepoints (641 = 4 seconds at 160 Hz)
models = {
    'EEGNet':    EEGNetv4(n_chans=nc, n_outputs=2, n_times=nt),
    'Shallow':   ShallowFBCSPNet(n_chans=nc, n_outputs=2, n_times=nt),
    'ATCNet':    ATCNet(n_chans=nc, n_outputs=2, n_times=nt),
    'Conformer': EEGConformer(n_chans=nc, n_outputs=2, n_times=nt),
    'CSP+LDA':   Pipeline([('csp', CSP()), ('lda', LDA())]),
}
```
]

#v(6pt)

#block(
  fill: luma(245),
  inset: 8pt,
  radius: 4pt,
  width: 100%,
)[
*Cross-patient evaluation* — no subject in both train and test:
```python
from sklearn.model_selection import GroupKFold
# Split by subject ID, not by trial — prevents data leakage
gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups=subject_ids):
    model.fit(X[train_idx], y[train_idx])  # train on ~83 subjects
    acc = model.score(X[test_idx], y[test_idx])  # test on ~21 new subjects
```
]

#v(6pt)

#block(
  fill: luma(245),
  inset: 8pt,
  radius: 4pt,
  width: 100%,
)[
*SHAP channel importance* — which electrodes matter most:
```python
import shap
# GradientExplainer computes feature attribution via gradients
explainer = shap.GradientExplainer(model, X_background)
shap_values = explainer.shap_values(X_test)
# Average absolute SHAP over samples and timepoints -> importance per channel
importance = np.abs(shap_values).mean(axis=(0, 1, 3))
top_channels = np.argsort(importance)[::-1]  # rank channels
```
]
