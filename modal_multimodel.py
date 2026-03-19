"""Experimento multi-modelo: EEGNet, ATCNet, EEG Conformer, CSP+LDA."""
import modal
import os

app = modal.App("eeg-multimodel")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy", "torch", "scikit-learn", "braindecode", "mne"
)

vol = modal.Volume.from_name("eeg-data", create_if_missing=True)


@app.function(gpu="T4", image=image, volumes={"/data": vol}, timeout=7200)
def run_model(model_name: str):
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np, time, pickle
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score

    np.random.seed(42); torch.manual_seed(42)
    DEVICE = torch.device('cuda')
    print(f"[{model_name}] Device: {DEVICE} ({torch.cuda.get_device_name(0)})")

    # Cargar data
    d = np.load('/data/data_f32.npz', allow_pickle=True)
    X_norm = d['X_norm']; y = d['y']; groups = d['groups']; ch_names = list(d['ch_names'])
    n_times = X_norm.shape[2]
    print(f"[{model_name}] Data: {X_norm.shape[0]} epochs, {len(np.unique(groups))} sujetos")

    # Seleccionar modelo
    def create_model(nc):
        if model_name == 'eegnet':
            from braindecode.models import EEGNetv4
            return EEGNetv4(n_chans=nc, n_outputs=2, n_times=n_times)
        elif model_name == 'atcnet':
            from braindecode.models import ATCNet
            return ATCNet(n_chans=nc, n_outputs=2, n_times=n_times)
        elif model_name == 'conformer':
            from braindecode.models import EEGConformer
            return EEGConformer(n_chans=nc, n_outputs=2, n_times=n_times)
        elif model_name == 'shallow':
            from braindecode.models import ShallowFBCSPNet
            return ShallowFBCSPNet(n_chans=nc, n_outputs=2, n_times=n_times)
        else:
            raise ValueError(f"Modelo desconocido: {model_name}")

    def train_and_eval(nc, Xtr, ytr, Xv, yv, Xte, yte):
        model = create_model(nc).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
        crit = nn.CrossEntropyLoss()

        # braindecode models expect (batch, channels, times) without the extra dim
        ds = TensorDataset(
            torch.tensor(Xtr, dtype=torch.float32),
            torch.tensor(ytr, dtype=torch.long)
        )
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
            sched.step(vl)
            if vl < best_vl:
                best_vl = vl; best_st = {k:v.cpu().clone() for k,v in model.state_dict().items()}; wait = 0
            else:
                wait += 1
                if wait >= 15: break
        model.load_state_dict(best_st); model.eval()
        with torch.no_grad():
            out = model(torch.tensor(Xte, dtype=torch.float32).to(DEVICE))
            probs = torch.softmax(out, 1)[:, 1].cpu().numpy()
            preds = out.argmax(1).cpu().numpy()
        return {
            'accuracy': accuracy_score(yte, preds),
            'kappa': cohen_kappa_score(yte, preds),
            'auc': roc_auc_score(yte, probs),
            'n_params': n_params,
        }, best_st

    CC = {
        64: list(ch_names),
        32: ['FC5','FC3','FC1','FCz','FC2','FC4','FC6','C5','C3','C1','Cz','C2','C4','C6','CP5','CP3','CP1','CPz','CP2','CP4','CP6','F3','F1','Fz','F2','F4','P3','P1','Pz','P2','P4','P6'],
        16: ['FC3','FC1','FCz','FC2','FC4','C3','C1','Cz','C2','C4','CP3','CP1','CPz','CP2','CP4','Fz'],
        8: ['FC3','FCz','FC4','C3','Cz','C4','CP3','CP4'],
        4: ['C3','Cz','C4','CPz'],
        2: ['C3','C4']
    }

    # Checkpoint
    ckpt_path = f'/data/results_{model_name}.pkl'
    try:
        results = pickle.load(open(ckpt_path, 'rb'))
        print(f"[{model_name}] Checkpoint: {list(results.keys())}")
    except:
        results = {}

    for n_ch, ch_list in CC.items():
        if n_ch in results:
            print(f"[{model_name}] {n_ch}ch HECHO (acc={np.mean(results[n_ch][0]):.3f})")
            continue

        ch_idx = [ch_names.index(c) for c in ch_list]
        Xs = X_norm[:, ch_idx, :]
        print(f"\n[{model_name}] --- {n_ch} canales ---")

        gkf = GroupKFold(n_splits=5)
        accs, kaps, aucs = [], [], []
        best_fold_acc = 0; best_fold_state = None

        for fold, (tri, tei) in enumerate(gkf.split(Xs, y, groups)):
            t1 = time.time()
            Xtr_f, Xte = Xs[tri], Xs[tei]
            ytr_f, yte = y[tri], y[tei]
            gtr = groups[tri]
            usub = np.unique(gtr)
            vs = np.random.RandomState(fold).choice(usub, max(2, len(usub)//10), replace=False)
            vm = np.isin(gtr, vs)

            metrics, state = train_and_eval(n_ch, Xtr_f[~vm], ytr_f[~vm], Xtr_f[vm], ytr_f[vm], Xte, yte)
            accs.append(metrics['accuracy'])
            kaps.append(metrics['kappa'])
            aucs.append(metrics['auc'])
            if metrics['accuracy'] > best_fold_acc:
                best_fold_acc = metrics['accuracy']
                best_fold_state = state

            print(f"[{model_name}]   F{fold+1} acc={metrics['accuracy']:.3f} kap={metrics['kappa']:.3f} auc={metrics['auc']:.3f} [{time.time()-t1:.0f}s]")

        results[n_ch] = (accs, kaps, aucs)
        print(f"[{model_name}]   MEAN acc={np.mean(accs):.3f}")

        # Guardar modelo
        model_path = f'/data/{model_name}_{n_ch}ch.pt'
        torch.save({
            'state_dict': best_fold_state,
            'n_channels': n_ch,
            'n_times': n_times,
            'accuracy': best_fold_acc,
            'channels': ch_list,
            'model_name': model_name,
        }, model_path)

        pickle.dump(results, open(ckpt_path, 'wb'))
        vol.commit()
        print(f"[{model_name}]   [checkpoint]")

    print(f"\n[{model_name}] RESULTADOS FINALES")
    print(f"  Ch    Acc    Std    Kap    AUC")
    for nc in sorted(results.keys(), reverse=True):
        a, k, u = results[nc]
        print(f"  {nc:>2}  {np.mean(a):.3f}  {np.std(a):.3f}  {np.mean(k):.3f}  {np.mean(u):.3f}")

    return model_name, results


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def run_csp_lda():
    """CSP+LDA: clásico, sin GPU."""
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np, time, pickle
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import Pipeline
    from mne.decoding import CSP

    np.random.seed(42)
    print("[csp_lda] Cargando data...")

    d = np.load('/data/data_f32.npz', allow_pickle=True)
    X_norm = d['X_norm']; y = d['y']; groups = d['groups']; ch_names = list(d['ch_names'])
    print(f"[csp_lda] Data: {X_norm.shape[0]} epochs, {len(np.unique(groups))} sujetos")

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
        n_components = min(4, n_ch)

        print(f"\n[csp_lda] --- {n_ch} canales ---")
        gkf = GroupKFold(n_splits=5)
        accs, kaps, aucs = [], [], []

        for fold, (tri, tei) in enumerate(gkf.split(Xs, y, groups)):
            t1 = time.time()
            Xtr, Xte = Xs[tri], Xs[tei]
            ytr, yte = y[tri], y[tei]

            csp = CSP(n_components=n_components, reg='ledoit_wolf')
            lda = LinearDiscriminantAnalysis()
            pipe = Pipeline([('csp', csp), ('lda', lda)])

            pipe.fit(Xtr, ytr)
            preds = pipe.predict(Xte)
            probs = pipe.predict_proba(Xte)[:, 1]

            acc = accuracy_score(yte, preds)
            kap = cohen_kappa_score(yte, preds)
            auc = roc_auc_score(yte, probs)
            accs.append(acc); kaps.append(kap); aucs.append(auc)
            print(f"[csp_lda]   F{fold+1} acc={acc:.3f} kap={kap:.3f} auc={auc:.3f} [{time.time()-t1:.1f}s]")

        results[n_ch] = (accs, kaps, aucs)
        print(f"[csp_lda]   MEAN acc={np.mean(accs):.3f}")

    pickle.dump(results, open('/data/results_csp_lda.pkl', 'wb'))
    vol.commit()

    print(f"\n[csp_lda] RESULTADOS FINALES")
    print(f"  Ch    Acc    Std    Kap    AUC")
    for nc in sorted(results.keys(), reverse=True):
        a, k, u = results[nc]
        print(f"  {nc:>2}  {np.mean(a):.3f}  {np.std(a):.3f}  {np.mean(k):.3f}  {np.mean(u):.3f}")

    return 'csp_lda', results


@app.local_entrypoint()
def main():
    import pickle

    # Lanzar todos en paralelo
    handles = []
    for model_name in ['atcnet', 'conformer', 'shallow']:
        handles.append(run_model.spawn(model_name))
    handles.append(run_csp_lda.spawn())

    print("4 modelos lanzados en paralelo...")
    print("  - ATCNet (GPU)")
    print("  - EEG Conformer (GPU)")
    print("  - ShallowConvNet (GPU)")
    print("  - CSP+LDA (CPU)")

    # Recoger resultados
    all_results = {}
    for handle in handles:
        name, results = handle.get()
        all_results[name] = results
        print(f"\n{name} completado!")

    # Guardar todo junto
    pickle.dump(all_results, open("multimodel_results.pkl", "wb"))
    print("\nTodos los resultados guardados en multimodel_results.pkl")

    # Tabla comparativa
    print("\n" + "=" * 60)
    print("COMPARACION MULTI-MODELO")
    print("=" * 60)

    # Cargar resultados de EEGNet previo
    try:
        eegnet_results = pickle.load(open("experiment_results.pkl", "rb"))
        all_results['eegnet'] = eegnet_results
    except:
        pass

    for nc in [64, 32, 16, 8, 4, 2]:
        print(f"\n  {nc} canales:")
        for model_name in ['csp_lda', 'eegnet', 'shallow', 'atcnet', 'conformer']:
            if model_name in all_results and nc in all_results[model_name]:
                a = all_results[model_name][nc][0]
                print(f"    {model_name:>12}: acc={np.mean(a):.3f} +/-{np.std(a):.3f}")
