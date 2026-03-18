"""EEG Motor Imagery — Electrode Reduction Experiment (runner script)"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pickle, time
import mne
from mne.datasets import eegbci

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, roc_auc_score

mne.set_log_level('ERROR')
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device('cpu')  # CPU is faster than MPS for small models like EEGNet
print(f"Device: {DEVICE}")

# ============================================================
# 1. LOAD DATA
# ============================================================
SUBJECTS = list(range(1, 110))
RUNS = [4, 8, 12]
EXCLUDE = {88, 92, 100, 104}

all_epochs = []
all_labels = []
all_groups = []

t0 = time.time()
for subj in SUBJECTS:
    if subj in EXCLUDE:
        continue
    try:
        fnames = eegbci.load_data(subj, RUNS, path='./data', update_path=True)
        raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames]
        for raw in raws:
            eegbci.standardize(raw)
        raw = mne.concatenate_raws(raws)

        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage, on_missing='ignore')
        raw.filter(8., 30., fir_design='firwin', verbose=False)

        events, event_id = mne.events_from_annotations(raw, verbose=False)
        event_id_mi = {k: v for k, v in event_id.items() if k in ['T1', 'T2']}

        if len(event_id_mi) < 2:
            continue

        epochs = mne.Epochs(raw, events, event_id_mi, tmin=0., tmax=4.0,
                           baseline=None, preload=True, verbose=False)
        epochs.drop_bad(reject=dict(eeg=500e-6), verbose=False)

        if len(epochs) < 5:
            continue

        data = epochs.get_data()
        labels = epochs.events[:, 2]
        labels = (labels == event_id_mi['T2']).astype(int)

        all_epochs.append(data)
        all_labels.append(labels)
        all_groups.append(np.full(len(labels), subj))

        if subj % 10 == 0:
            print(f"  Subject {subj}/109 ({len(epochs)} epochs) [{time.time()-t0:.0f}s]")
    except Exception as e:
        print(f"  Subject {subj}: error - {e}")

X = np.concatenate(all_epochs, axis=0)
y = np.concatenate(all_labels, axis=0)
groups = np.concatenate(all_groups, axis=0)
ch_names = list(epochs.ch_names)

print(f"\nData loaded in {time.time()-t0:.0f}s")
print(f"  X: {X.shape}, y: {y.shape}, subjects: {len(np.unique(groups))}")

# ============================================================
# 2. NORMALIZE
# ============================================================
X_norm = (X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True) + 1e-8)

# ============================================================
# 3. EEGNET
# ============================================================
class EEGNet(nn.Module):
    def __init__(self, n_channels, n_times, n_classes=2,
                 F1=8, D=2, F2=16, kernel_length=64, dropout=0.5):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding='same', bias=False),
            nn.BatchNorm2d(F1)
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding='same', groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(F2 * (n_times // 32), n_classes)

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        x = x.flatten(1)
        return self.classifier(x)

# ============================================================
# 4. TRAIN / EVAL
# ============================================================
def train_model(model, X_train, y_train, X_val, y_val,
                epochs=100, batch_size=64, lr=1e-3, patience=15):
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.tensor(X_train[:, np.newaxis, :, :], dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.tensor(X_val[:, np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test[:, np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
        outputs = model(X_t)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()

    return {
        'accuracy': accuracy_score(y_test, preds),
        'kappa': cohen_kappa_score(y_test, preds),
        'auc': roc_auc_score(y_test, probs),
        'cm': confusion_matrix(y_test, preds),
    }

# ============================================================
# 5. CHANNEL CONFIGS
# ============================================================
channel_configs = {
    64: list(ch_names),
    32: ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
         'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
         'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
         'F3', 'F1', 'Fz', 'F2', 'F4',
         'P3', 'P1', 'Pz', 'P2', 'P4'],
    16: ['FC3', 'FC1', 'FCz', 'FC2', 'FC4',
         'C3', 'C1', 'Cz', 'C2', 'C4',
         'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'Fz'],
    8:  ['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CP4'],
    4:  ['C3', 'Cz', 'C4', 'CPz'],
    2:  ['C3', 'C4'],
}

# ============================================================
# 6. RUN EXPERIMENT
# ============================================================
all_results = {}

for n_ch, ch_list in channel_configs.items():
    ch_indices = [ch_names.index(c) for c in ch_list]
    X_subset = X_norm[:, ch_indices, :]

    print(f"\n{'='*50}")
    print(f"=== {n_ch} CHANNELS ===")
    print(f"{'='*50}")

    gkf = GroupKFold(n_splits=5)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_subset, y, groups)):
        t1 = time.time()
        X_train_full, X_test = X_subset[train_idx], X_subset[test_idx]
        y_train_full, y_test = y[train_idx], y[test_idx]
        g_train = groups[train_idx]

        unique_subj = np.unique(g_train)
        n_val_subj = max(2, len(unique_subj) // 10)
        rng = np.random.RandomState(fold)
        val_subj = rng.choice(unique_subj, n_val_subj, replace=False)
        val_mask = np.isin(g_train, val_subj)

        X_train, X_val = X_train_full[~val_mask], X_train_full[val_mask]
        y_train, y_val = y_train_full[~val_mask], y_train_full[val_mask]

        model = EEGNet(n_channels=n_ch, n_times=X_subset.shape[2])
        model = train_model(model, X_train, y_train, X_val, y_val)
        result = evaluate_model(model, X_test, y_test)
        fold_results.append(result)

        print(f"  Fold {fold+1}/5: acc={result['accuracy']:.3f}, "
              f"kappa={result['kappa']:.3f}, AUC={result['auc']:.3f} "
              f"[{time.time()-t1:.0f}s]")

    all_results[n_ch] = fold_results
    mean_acc = np.mean([r['accuracy'] for r in fold_results])
    mean_kappa = np.mean([r['kappa'] for r in fold_results])
    mean_auc = np.mean([r['auc'] for r in fold_results])
    print(f"  MEAN: acc={mean_acc:.3f}, kappa={mean_kappa:.3f}, AUC={mean_auc:.3f}")

# ============================================================
# 7. SUMMARY
# ============================================================
print(f"\n{'='*50}")
print("FINAL RESULTS")
print(f"{'='*50}")
print(f"{'Channels':>10} {'Accuracy':>10} {'± Std':>8} {'Kappa':>8} {'AUC':>8}")
print("-" * 48)
for n_ch in sorted(all_results.keys(), reverse=True):
    accs = [r['accuracy'] for r in all_results[n_ch]]
    kappas = [r['kappa'] for r in all_results[n_ch]]
    aucs = [r['auc'] for r in all_results[n_ch]]
    print(f"{n_ch:>10} {np.mean(accs):>10.3f} {np.std(accs):>8.3f} "
          f"{np.mean(kappas):>8.3f} {np.mean(aucs):>8.3f}")

# Save results
with open('experiment_results.pkl', 'wb') as f:
    pickle.dump({'all_results': all_results, 'ch_names': ch_names,
                 'channel_configs': channel_configs}, f)
print("\nResults saved to experiment_results.pkl")
