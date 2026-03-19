"""Permutation test: verifica que el modelo aprende de los labels reales, no de ruido."""
import modal
import os

app = modal.App("eeg-permutation-test")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy", "torch", "scikit-learn"
)

vol = modal.Volume.from_name("eeg-data", create_if_missing=True)


@app.function(gpu="T4", image=image, volumes={"/data": vol}, timeout=3600)
def run_permutation():
    import numpy as np, time
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import accuracy_score

    np.random.seed(42); torch.manual_seed(42)
    DEVICE = torch.device('cuda')
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})")

    d = np.load('/data/data_f32.npz', allow_pickle=True)
    X_norm = d['X_norm']; y = d['y']; groups = d['groups']; ch_names = list(d['ch_names'])
    print(f"Data: {X_norm.shape[0]} epochs, {len(np.unique(groups))} sujetos")

    class EEGNet(nn.Module):
        def __init__(self, nc, nt):
            super().__init__()
            F1, D, F2 = 8, 2, 16
            self.block1 = nn.Sequential(nn.Conv2d(1,F1,(1,64),padding='same',bias=False), nn.BatchNorm2d(F1))
            self.block2 = nn.Sequential(nn.Conv2d(F1,F1*D,(nc,1),groups=F1,bias=False), nn.BatchNorm2d(F1*D), nn.ELU(), nn.AvgPool2d((1,4)), nn.Dropout(0.5))
            self.block3 = nn.Sequential(nn.Conv2d(F1*D,F1*D,(1,16),padding='same',groups=F1*D,bias=False), nn.Conv2d(F1*D,F2,(1,1),bias=False), nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1,8)), nn.Dropout(0.5))
            self.fc = nn.Linear(F2*(nt//32), 2)
        def forward(self, x):
            x = self.block1(x); x = self.block2(x); x = self.block3(x)
            return self.fc(x.flatten(1))

    def train_and_eval(nc, Xtr, ytr, Xv, yv, Xte, yte):
        model = EEGNet(nc, Xtr.shape[2]).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        ds = TensorDataset(torch.tensor(Xtr[:,None,:,:], dtype=torch.float32), torch.tensor(ytr, dtype=torch.long))
        dl = DataLoader(ds, batch_size=256, shuffle=True, pin_memory=True, num_workers=2)
        Xvt = torch.tensor(Xv[:,None,:,:], dtype=torch.float32).to(DEVICE)
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
            preds = model(torch.tensor(Xte[:,None,:,:], dtype=torch.float32).to(DEVICE)).argmax(1).cpu().numpy()
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

        print(f"\n--- {n_ch} canales ---")

        # Solo 1 fold para el permutation test (suficiente para validar)
        gkf = GroupKFold(n_splits=5)
        tri, tei = next(iter(gkf.split(Xs, y, groups)))
        Xtr_f, Xte = Xs[tri], Xs[tei]
        ytr_f, yte = y[tri], y[tei]
        gtr = groups[tri]
        usub = np.unique(gtr)
        vs = np.random.RandomState(0).choice(usub, max(2, len(usub)//10), replace=False)
        vm = np.isin(gtr, vs)

        # Labels reales
        t1 = time.time()
        acc_real = train_and_eval(n_ch, Xtr_f[~vm], ytr_f[~vm], Xtr_f[vm], ytr_f[vm], Xte, yte)
        print(f"  Labels REALES:   acc={acc_real:.3f} [{time.time()-t1:.0f}s]")

        # Labels barajados (3 permutaciones)
        perm_accs = []
        for perm in range(3):
            t1 = time.time()
            y_shuffled = y.copy()
            np.random.RandomState(perm + 100).shuffle(y_shuffled)
            ytr_s, yte_s = y_shuffled[tri], y_shuffled[tei]
            acc_perm = train_and_eval(n_ch, Xtr_f[~vm], ytr_s[~vm], Xtr_f[vm], ytr_s[vm], Xte, yte_s)
            perm_accs.append(acc_perm)
            print(f"  Labels RANDOM {perm+1}: acc={acc_perm:.3f} [{time.time()-t1:.0f}s]")

        mean_perm = np.mean(perm_accs)
        diff = acc_real - mean_perm
        passed = "PASA" if diff > 0.05 else "NO PASA"
        results[n_ch] = (acc_real, mean_perm, diff, passed)
        print(f"  Diferencia: {diff:.3f} -> {passed}")

    print(f"\n{'='*50}")
    print("PERMUTATION TEST - RESULTADOS")
    print(f"{'='*50}")
    print(f"  Ch   Real   Random   Diff   Veredicto")
    for nc in sorted(results.keys(), reverse=True):
        real, rand, diff, passed = results[nc]
        print(f"  {nc:>2}   {real:.3f}  {rand:.3f}   {diff:+.3f}  {passed}")

    return results


@app.local_entrypoint()
def main():
    results = run_permutation.remote()
    import pickle
    pickle.dump(results, open("permutation_results.pkl", "wb"))
    print("\nGuardado en permutation_results.pkl")
