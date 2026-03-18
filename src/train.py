"""Entrenamiento y evaluación de modelos EEG."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score


def train_model(model, Xtr, ytr, Xv, yv, device,
                epochs=100, batch_size=256, lr=1e-3, patience=15):
    """Entrena un modelo con early stopping."""
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    crit = nn.CrossEntropyLoss()

    ds = TensorDataset(
        torch.tensor(Xtr[:, None, :, :], dtype=torch.float32),
        torch.tensor(ytr, dtype=torch.long)
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    Xvt = torch.tensor(Xv[:, None, :, :], dtype=torch.float32).to(device)
    yvt = torch.tensor(yv, dtype=torch.long).to(device)

    best_vl, best_st, wait = float('inf'), None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xvt), yvt).item()
        sched.step(vl)
        if vl < best_vl:
            best_vl = vl
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_st)
    model.eval()
    return model


def eval_model(model, Xte, yte, device):
    """Evalúa un modelo y retorna métricas."""
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(Xte[:, None, :, :], dtype=torch.float32).to(device))
        probs = torch.softmax(out, 1)[:, 1].cpu().numpy()
        preds = out.argmax(1).cpu().numpy()
    return {
        'accuracy': accuracy_score(yte, preds),
        'kappa': cohen_kappa_score(yte, preds),
        'auc': roc_auc_score(yte, probs),
    }


def run_experiment(model_class, X_norm, y, groups, ch_names, channel_configs,
                   device, save_models_dir=None):
    """Corre el experimento completo de reducción de electrodos.

    Args:
        model_class: clase del modelo (ej. EEGNet)
        X_norm: data normalizada (n_epochs, n_channels, n_times)
        y: labels
        groups: subject IDs
        ch_names: nombres de canales
        channel_configs: dict {n_ch: [lista de canales]}
        device: torch device
        save_models_dir: si no es None, guarda el mejor modelo de cada config

    Returns:
        dict {n_ch: (accs, kappas, aucs)}
    """
    import os, pickle

    results = {}
    best_models = {}

    for n_ch, ch_list in channel_configs.items():
        ch_idx = [ch_names.index(c) for c in ch_list]
        Xs = X_norm[:, ch_idx, :]
        print(f"\n--- {n_ch} canales ---")

        gkf = GroupKFold(n_splits=5)
        accs, kaps, aucs = [], [], []
        best_fold_acc = 0
        best_fold_state = None

        for fold, (tri, tei) in enumerate(gkf.split(Xs, y, groups)):
            import time; t1 = time.time()
            Xtr_f, Xte = Xs[tri], Xs[tei]
            ytr_f, yte = y[tri], y[tei]
            gtr = groups[tri]

            # Val split por sujeto
            usub = np.unique(gtr)
            vs = np.random.RandomState(fold).choice(usub, max(2, len(usub) // 10), replace=False)
            vm = np.isin(gtr, vs)

            model = model_class(n_ch, Xs.shape[2]).to(device)
            model = train_model(model, Xtr_f[~vm], ytr_f[~vm], Xtr_f[vm], ytr_f[vm], device)
            r = eval_model(model, Xte, yte, device)
            accs.append(r['accuracy'])
            kaps.append(r['kappa'])
            aucs.append(r['auc'])

            # Guardar mejor fold
            if r['accuracy'] > best_fold_acc:
                best_fold_acc = r['accuracy']
                best_fold_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            print(f"  F{fold+1} acc={r['accuracy']:.3f} kap={r['kappa']:.3f} auc={r['auc']:.3f} [{time.time()-t1:.0f}s]")

        results[n_ch] = (accs, kaps, aucs)
        print(f"  MEAN acc={np.mean(accs):.3f}")

        # Guardar modelo del mejor fold
        if save_models_dir and best_fold_state:
            os.makedirs(save_models_dir, exist_ok=True)
            path = os.path.join(save_models_dir, f'eegnet_{n_ch}ch.pt')
            torch.save({
                'state_dict': best_fold_state,
                'n_channels': n_ch,
                'n_times': Xs.shape[2],
                'accuracy': best_fold_acc,
                'channels': ch_list,
            }, path)
            print(f"  Modelo guardado: {path}")

    return results
