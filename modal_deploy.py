"""Entrenamiento remoto via Modal Deploy — no depende de conexión local."""
import modal

app = modal.App("eeg-deploy")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy", "torch", "scikit-learn", "braindecode", "mne"
)

vol = modal.Volume.from_name("eeg-data", create_if_missing=True)


@app.function(gpu="T4", image=image, volumes={"/data": vol}, timeout=7200)
def train(model_name: str, n_channels: int = 0):
    """Entrena un modelo. Guarda resultados y pesos en Volume."""
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np, time, pickle
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score

    np.random.seed(42); torch.manual_seed(42)
    DEVICE = torch.device('cuda')
    print(f"[{model_name}] Device: {DEVICE} ({torch.cuda.get_device_name(0)})")

    d = np.load('/data/data_f32.npz', allow_pickle=True)
    X_norm = d['X_norm']; y = d['y']; groups = d['groups']; ch_names = list(d['ch_names'])
    n_times = X_norm.shape[2]
    print(f"[{model_name}] Data: {X_norm.shape[0]} epochs, {len(np.unique(groups))} sujetos")

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
        opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
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
    ckpt_path = f'/data/deploy_results_{model_name}.pkl'
    try:
        results = pickle.load(open(ckpt_path, 'rb'))
        print(f"[{model_name}] Checkpoint: {list(results.keys())}")
    except:
        results = {}

    for n_ch, ch_list in CC.items():
        if n_channels > 0 and n_ch != n_channels:
            continue
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
        torch.save({
            'state_dict': best_fold_state,
            'n_channels': n_ch, 'n_times': n_times,
            'accuracy': best_fold_acc, 'channels': ch_list,
            'model_name': model_name,
        }, f'/data/{model_name}_{n_ch}ch.pt')

        pickle.dump(results, open(ckpt_path, 'wb'))
        vol.commit()
        print(f"[{model_name}]   [checkpoint + modelo guardado]")

    print(f"\n[{model_name}] COMPLETO")
    for nc in sorted(results.keys(), reverse=True):
        a, k, u = results[nc]
        print(f"  {nc:>2}ch  acc={np.mean(a):.3f}  kap={np.mean(k):.3f}  auc={np.mean(u):.3f}")

    return model_name, results


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def train_csp_lda():
    """CSP+LDA sin GPU."""
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np, time, pickle
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import Pipeline
    from mne.decoding import CSP

    np.random.seed(42)
    d = np.load('/data/data_f32.npz', allow_pickle=True)
    X_norm = d['X_norm']; y = d['y']; groups = d['groups']; ch_names = list(d['ch_names'])
    print(f"[csp_lda] Data: {X_norm.shape[0]} epochs")

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
        accs, kaps, aucs = [], [], []
        for fold, (tri, tei) in enumerate(gkf.split(Xs, y, groups)):
            t1 = time.time()
            pipe = Pipeline([('csp', CSP(n_components=min(4, n_ch), reg='ledoit_wolf')),
                           ('lda', LinearDiscriminantAnalysis())])
            pipe.fit(Xs[tri], y[tri])
            preds = pipe.predict(Xs[tei])
            probs = pipe.predict_proba(Xs[tei])[:, 1]
            acc = accuracy_score(y[tei], preds)
            accs.append(acc); kaps.append(cohen_kappa_score(y[tei], preds)); aucs.append(roc_auc_score(y[tei], probs))
            print(f"[csp_lda]   F{fold+1} acc={acc:.3f} [{time.time()-t1:.1f}s]")
        results[n_ch] = (accs, kaps, aucs)
        print(f"[csp_lda]   MEAN acc={np.mean(accs):.3f}")

    pickle.dump(results, open('/data/deploy_results_csp_lda.pkl', 'wb'))
    vol.commit()
    print("[csp_lda] COMPLETO")
    return 'csp_lda', results


@app.function(volumes={"/data": vol})
def get_results():
    """Consulta resultados sin re-entrenar."""
    import pickle, os, numpy as np
    all_results = {}
    for f in os.listdir('/data'):
        if f.startswith('deploy_results_') and f.endswith('.pkl'):
            name = f.replace('deploy_results_', '').replace('.pkl', '')
            all_results[name] = pickle.load(open(f'/data/{f}', 'rb'))

    # Incluir resultados previos
    for f in os.listdir('/data'):
        if f.startswith('results_') and f.endswith('.pkl') and 'checkpoint' not in f and 'deploy' not in f:
            name = f.replace('results_', '').replace('.pkl', '')
            if name not in all_results:
                all_results[name] = pickle.load(open(f'/data/{f}', 'rb'))

    print("RESULTADOS DISPONIBLES")
    print("=" * 60)
    for nc in [64, 32, 16, 8, 4, 2]:
        print(f"\n  {nc} canales:")
        for model_name in sorted(all_results.keys()):
            if nc in all_results[model_name]:
                a = all_results[model_name][nc][0]
                print(f"    {model_name:>12}: acc={np.mean(a):.3f} +/-{np.std(a):.3f}")

    return all_results


@app.function(volumes={"/data": vol})
def download_file(filename: str) -> bytes:
    """Descarga un archivo del Volume."""
    with open(f'/data/{filename}', 'rb') as f:
        return f.read()


@app.function(volumes={"/data": vol})
def list_files():
    """Lista archivos en el Volume."""
    import os
    files = []
    for f in sorted(os.listdir('/data')):
        size = os.path.getsize(f'/data/{f}')
        files.append(f"{f} ({size/1024:.0f} KB)")
        print(f"  {f} ({size/1024:.0f} KB)")
    return files
