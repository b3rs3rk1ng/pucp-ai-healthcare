"""Permutation test para todos los modelos via Modal Deploy."""
import modal

app = modal.App("eeg-permutation-all")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy", "torch", "scikit-learn", "braindecode", "mne"
)

vol = modal.Volume.from_name("eeg-data", create_if_missing=True)


@app.function(gpu="T4", image=image, volumes={"/data": vol}, timeout=7200)
def permutation_test_dl(model_name: str):
    """Permutation test para modelos deep learning."""
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np, time, pickle
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import accuracy_score

    np.random.seed(42); torch.manual_seed(42)
    DEVICE = torch.device('cuda')
    print(f"[{model_name}] Permutation test - Device: {DEVICE}")

    d = np.load('/data/data_f32.npz', allow_pickle=True)
    X_norm = d['X_norm']; y = d['y']; groups = d['groups']; ch_names = list(d['ch_names'])
    n_times = X_norm.shape[2]

    def create_model(nc):
        if model_name == 'atcnet':
            from braindecode.models import ATCNet
            return ATCNet(n_chans=nc, n_outputs=2, n_times=n_times)
        elif model_name == 'conformer':
            from braindecode.models import EEGConformer
            return EEGConformer(n_chans=nc, n_outputs=2, n_times=n_times)
        elif model_name == 'shallow':
            from braindecode.models import ShallowFBCSPNet
            return ShallowFBCSPNet(n_chans=nc, n_outputs=2, n_times=n_times)

    def train_eval(nc, Xtr, ytr, Xv, yv, Xte, yte):
        model = create_model(nc).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long))
        dl = DataLoader(ds, batch_size=256, shuffle=True, pin_memory=True, num_workers=2)
        Xvt = torch.tensor(Xv, dtype=torch.float32).to(DEVICE)
        yvt = torch.tensor(yv, dtype=torch.long).to(DEVICE)
        best_vl = 999; best_st = None; wait = 0
        for epoch in range(100):
            model.train()
            for xb, yb in dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
            model.eval()
            with torch.no_grad(): vl = crit(model(Xvt), yvt).item()
            if vl < best_vl:
                best_vl = vl; best_st = {k:v.cpu().clone() for k,v in model.state_dict().items()}; wait = 0
            else:
                wait += 1
                if wait >= 15: break
        model.load_state_dict(best_st); model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(Xte, dtype=torch.float32).to(DEVICE)).argmax(1).cpu().numpy()
        return accuracy_score(yte, preds)

    CC = {
        64: list(ch_names),
        32: ['FC5','FC3','FC1','FCz','FC2','FC4','FC6','C5','C3','C1','Cz','C2','C4','C6','CP5','CP3','CP1','CPz','CP2','CP4','CP6','F3','F1','Fz','F2','F4','P3','P1','Pz','P2','P4','P6'],
        16: ['FC3','FC1','FCz','FC2','FC4','C3','C1','Cz','C2','C4','CP3','CP1','CPz','CP2','CP4','Fz'],
        8: ['FC3','FCz','FC4','C3','Cz','C4','CP3','CP4'],
        4: ['C3','Cz','C4','CPz'],
        2: ['C3','C4']
    }

    results = {}
    for n_ch, ch_list in CC.items():
        ch_idx = [ch_names.index(c) for c in ch_list]
        Xs = X_norm[:, ch_idx, :]
        print(f"\n[{model_name}] --- {n_ch} canales ---")

        gkf = GroupKFold(n_splits=5)
        tri, tei = next(iter(gkf.split(Xs, y, groups)))
        Xtr_f, Xte = Xs[tri], Xs[tei]
        ytr_f, yte = y[tri], y[tei]
        gtr = groups[tri]
        usub = np.unique(gtr)
        vs = np.random.RandomState(0).choice(usub, max(2, len(usub)//10), replace=False)
        vm = np.isin(gtr, vs)

        t1 = time.time()
        acc_real = train_eval(n_ch, Xtr_f[~vm], ytr_f[~vm], Xtr_f[vm], ytr_f[vm], Xte, yte)
        print(f"[{model_name}]   REAL:   {acc_real:.3f} [{time.time()-t1:.0f}s]")

        perm_accs = []
        for perm in range(3):
            t1 = time.time()
            y_shuf = y.copy()
            np.random.RandomState(perm + 100).shuffle(y_shuf)
            ytr_s, yte_s = y_shuf[tri], y_shuf[tei]
            acc_p = train_eval(n_ch, Xtr_f[~vm], ytr_s[~vm], Xtr_f[vm], ytr_s[vm], Xte, yte_s)
            perm_accs.append(acc_p)
            print(f"[{model_name}]   RAND{perm+1}: {acc_p:.3f} [{time.time()-t1:.0f}s]")

        diff = acc_real - np.mean(perm_accs)
        passed = "PASA" if diff > 0.05 else "NO PASA"
        results[n_ch] = (acc_real, np.mean(perm_accs), diff, passed)
        print(f"[{model_name}]   Diff: {diff:+.3f} -> {passed}")

    print(f"\n[{model_name}] PERMUTATION TEST COMPLETO")
    print(f"  Ch   Real   Rand   Diff   Veredicto")
    for nc in sorted(results.keys(), reverse=True):
        r, rn, d, p = results[nc]
        print(f"  {nc:>2}   {r:.3f}  {rn:.3f}  {d:+.3f}  {p}")

    pickle.dump(results, open(f'/data/permutation_{model_name}.pkl', 'wb'))
    vol.commit()
    return model_name, results


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def permutation_test_csp():
    """Permutation test para CSP+LDA."""
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np, time, pickle
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import accuracy_score
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import Pipeline
    from mne.decoding import CSP

    np.random.seed(42)
    d = np.load('/data/data_f32.npz', allow_pickle=True)
    X_norm = d['X_norm']; y = d['y']; groups = d['groups']; ch_names = list(d['ch_names'])
    print("[csp_lda] Permutation test")

    CC = {
        64: list(ch_names),
        32: ['FC5','FC3','FC1','FCz','FC2','FC4','FC6','C5','C3','C1','Cz','C2','C4','C6','CP5','CP3','CP1','CPz','CP2','CP4','CP6','F3','F1','Fz','F2','F4','P3','P1','Pz','P2','P4','P6'],
        16: ['FC3','FC1','FCz','FC2','FC4','C3','C1','Cz','C2','C4','CP3','CP1','CPz','CP2','CP4','Fz'],
        8: ['FC3','FCz','FC4','C3','Cz','C4','CP3','CP4'],
        4: ['C3','Cz','C4','CPz'],
        2: ['C3','C4']
    }

    results = {}
    for n_ch, ch_list in CC.items():
        ch_idx = [ch_names.index(c) for c in ch_list]
        Xs = X_norm[:, ch_idx, :]
        print(f"\n[csp_lda] --- {n_ch} canales ---")

        gkf = GroupKFold(n_splits=5)
        tri, tei = next(iter(gkf.split(Xs, y, groups)))

        pipe = Pipeline([('csp', CSP(n_components=min(4, n_ch), reg='ledoit_wolf')),
                        ('lda', LinearDiscriminantAnalysis())])
        pipe.fit(Xs[tri], y[tri])
        acc_real = accuracy_score(y[tei], pipe.predict(Xs[tei]))
        print(f"[csp_lda]   REAL:   {acc_real:.3f}")

        perm_accs = []
        for perm in range(3):
            y_shuf = y.copy()
            np.random.RandomState(perm + 100).shuffle(y_shuf)
            pipe2 = Pipeline([('csp', CSP(n_components=min(4, n_ch), reg='ledoit_wolf')),
                             ('lda', LinearDiscriminantAnalysis())])
            pipe2.fit(Xs[tri], y_shuf[tri])
            acc_p = accuracy_score(y_shuf[tei], pipe2.predict(Xs[tei]))
            perm_accs.append(acc_p)
            print(f"[csp_lda]   RAND{perm+1}: {acc_p:.3f}")

        diff = acc_real - np.mean(perm_accs)
        passed = "PASA" if diff > 0.05 else "NO PASA"
        results[n_ch] = (acc_real, np.mean(perm_accs), diff, passed)
        print(f"[csp_lda]   Diff: {diff:+.3f} -> {passed}")

    pickle.dump(results, open('/data/permutation_csp_lda.pkl', 'wb'))
    vol.commit()
    print("\n[csp_lda] COMPLETO")
    return 'csp_lda', results


@app.local_entrypoint()
def main():
    import pickle

    # Lanzar 4 en paralelo (3 GPU + 1 CPU)
    handles = []
    for m in ['atcnet', 'conformer', 'shallow']:
        handles.append(permutation_test_dl.spawn(m))
    handles.append(permutation_test_csp.spawn())

    print("4 permutation tests lanzados en paralelo...")

    all_perm = {}
    for h in handles:
        name, results = h.get()
        all_perm[name] = results
        print(f"  {name} completado!")

    pickle.dump(all_perm, open("permutation_all_results.pkl", "wb"))
    print("\nGuardado en permutation_all_results.pkl")
