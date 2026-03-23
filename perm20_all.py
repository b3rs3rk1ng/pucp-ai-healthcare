import warnings; warnings.filterwarnings("ignore")
import numpy as np, time, pickle, os
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

np.random.seed(42); torch.manual_seed(42)
DEVICE = torch.device("cuda")
N_PERMS = 20
SAVE = "/home/f3mt0/Documents/github/pucp-ai-healthcare"

d = np.load(f"{SAVE}/data_f32.npz", allow_pickle=True)
X_norm = d["X_norm"]; y = d["y"]; groups = d["groups"]; ch_names = [str(c) for c in d["ch_names"]]
n_times = X_norm.shape[2]
print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
print(f"Data: {X_norm.shape}, {N_PERMS} permutaciones, 5 modelos")

# Models
class EEGNet(nn.Module):
    def __init__(self, nc, nt):
        super().__init__()
        F1, D, F2 = 8, 2, 16
        self.block1 = nn.Sequential(nn.Conv2d(1,F1,(1,64),padding="same",bias=False), nn.BatchNorm2d(F1))
        self.block2 = nn.Sequential(nn.Conv2d(F1,F1*D,(nc,1),groups=F1,bias=False), nn.BatchNorm2d(F1*D), nn.ELU(), nn.AvgPool2d((1,4)), nn.Dropout(0.5))
        self.block3 = nn.Sequential(nn.Conv2d(F1*D,F1*D,(1,16),padding="same",groups=F1*D,bias=False), nn.Conv2d(F1*D,F2,(1,1),bias=False), nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1,8)), nn.Dropout(0.5))
        self.fc = nn.Linear(F2*(nt//32), 2)
    def forward(self, x):
        return self.fc(self.block3(self.block2(self.block1(x))).flatten(1))

def create_model(name, nc):
    if name == "eegnet":
        return EEGNet(nc, n_times)
    elif name == "atcnet":
        from braindecode.models import ATCNet
        return ATCNet(n_chans=nc, n_outputs=2, n_times=n_times)
    elif name == "conformer":
        from braindecode.models import EEGConformer
        return EEGConformer(n_chans=nc, n_outputs=2, n_times=n_times)
    elif name == "shallow":
        from braindecode.models import ShallowFBCSPNet
        return ShallowFBCSPNet(n_chans=nc, n_outputs=2, n_times=n_times)

def train_eval_dl(name, nc, Xtr, ytr, Xv, yv, Xte, yte):
    model = create_model(name, nc).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    # braindecode models: no extra dim needed
    if name == "eegnet":
        Xtr_t = torch.tensor(Xtr[:,None,:,:], dtype=torch.float32)
        Xv_t = torch.tensor(Xv[:,None,:,:], dtype=torch.float32).to(DEVICE)
        Xte_t = torch.tensor(Xte[:,None,:,:], dtype=torch.float32)
    else:
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
        Xv_t = torch.tensor(Xv, dtype=torch.float32).to(DEVICE)
        Xte_t = torch.tensor(Xte, dtype=torch.float32)
    ds = TensorDataset(Xtr_t, torch.tensor(ytr, dtype=torch.long))
    dl = DataLoader(ds, batch_size=256, shuffle=True, pin_memory=True, num_workers=2)
    yv_t = torch.tensor(yv, dtype=torch.long).to(DEVICE)
    best_vl = 999; best_st = None; wait = 0
    for ep in range(100):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = crit(model(Xv_t), yv_t).item()
        if vl < best_vl:
            best_vl = vl; best_st = {k:v.cpu().clone() for k,v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= 15: break
    model.load_state_dict(best_st); model.eval()
    with torch.no_grad():
        preds = model(Xte_t.to(DEVICE)).argmax(1).cpu().numpy()
    return accuracy_score(yte, preds)

def train_eval_csp(nc, Xtr, ytr, Xte, yte):
    from mne.decoding import CSP
    pipe = Pipeline([("csp", CSP(n_components=min(4, nc), reg="ledoit_wolf")),
                     ("lda", LinearDiscriminantAnalysis())])
    pipe.fit(Xtr, ytr)
    return accuracy_score(yte, pipe.predict(Xte))

CC = {
    64: list(ch_names),
    32: ["FC5","FC3","FC1","FCz","FC2","FC4","FC6","C5","C3","C1","Cz","C2","C4","C6","CP5","CP3","CP1","CPz","CP2","CP4","CP6","F3","F1","Fz","F2","F4","P3","P1","Pz","P2","P4","P6"],
    16: ["FC3","FC1","FCz","FC2","FC4","C3","C1","Cz","C2","C4","CP3","CP1","CPz","CP2","CP4","Fz"],
    8: ["FC3","FCz","FC4","C3","Cz","C4","CP3","CP4"],
    4: ["C3","Cz","C4","CPz"],
    2: ["C3","C4"]
}

MODELS = ["eegnet", "shallow", "atcnet", "conformer", "csp_lda"]

ckpt_path = f"{SAVE}/perm20_all_checkpoint.pkl"
try:
    all_results = pickle.load(open(ckpt_path, "rb"))
    print(f"Checkpoint: {list(all_results.keys())}")
except:
    all_results = {}

for model_name in MODELS:
    if model_name in all_results:
        print(f"\n[{model_name}] COMPLETO")
        for nc in sorted(all_results[model_name].keys(), reverse=True):
            r = all_results[model_name][nc]
            print(f"  {nc:>2}ch acc={r[real]:.3f} p={r[p_value]:.4f} {r[passed]}")
        continue

    print(f"\n{"="*50}")
    print(f"  {model_name.upper()}")
    print(f"{chr(61)*50}")

    model_results = {}
    for n_ch, ch_list in CC.items():
        ch_idx = [ch_names.index(c) for c in ch_list]
        Xs = X_norm[:, ch_idx, :]
        print(f"\n[{model_name}] {n_ch}ch")

        gkf = GroupKFold(n_splits=5)
        tri, tei = next(iter(gkf.split(Xs, y, groups)))
        Xtr_f, Xte = Xs[tri], Xs[tei]
        ytr_f, yte = y[tri], y[tei]
        gtr = groups[tri]
        usub = np.unique(gtr)
        vs = np.random.RandomState(0).choice(usub, max(2, len(usub)//10), replace=False)
        vm = np.isin(gtr, vs)

        t1 = time.time()
        if model_name == "csp_lda":
            acc_real = train_eval_csp(n_ch, Xtr_f, ytr_f, Xte, yte)
        else:
            acc_real = train_eval_dl(model_name, n_ch, Xtr_f[~vm], ytr_f[~vm], Xtr_f[vm], ytr_f[vm], Xte, yte)
        print(f"  REAL: {acc_real:.3f} [{time.time()-t1:.0f}s]")

        perm_accs = []
        count_ge = 0
        for perm in range(N_PERMS):
            t1 = time.time()
            y_shuf = y.copy()
            np.random.RandomState(perm + 500).shuffle(y_shuf)
            ytr_s, yte_s = y_shuf[tri], y_shuf[tei]
            if model_name == "csp_lda":
                acc_p = train_eval_csp(n_ch, Xs[tri], ytr_s, Xs[tei], yte_s)
            else:
                acc_p = train_eval_dl(model_name, n_ch, Xtr_f[~vm], ytr_s[~vm], Xtr_f[vm], ytr_s[vm], Xte, yte_s)
            perm_accs.append(acc_p)
            if acc_p >= acc_real:
                count_ge += 1
            if (perm+1) % 5 == 0:
                print(f"  PERM {perm+1}/{N_PERMS} acc={acc_p:.3f} [{time.time()-t1:.0f}s]")

        p_value = (count_ge + 1) / (N_PERMS + 1)
        passed = "PASA" if p_value < 0.05 else "NO PASA"
        model_results[n_ch] = {"real": acc_real, "perm_accs": perm_accs, "p_value": p_value, "passed": passed}
        print(f"  p={p_value:.4f} -> {passed}")

    all_results[model_name] = model_results
    pickle.dump(all_results, open(ckpt_path, "wb"))
    print(f"[{model_name}] checkpoint guardado")

print(f"\n{"="*50}")
print("RESULTADOS FINALES — 20 PERMUTACIONES")
print(f"{chr(61)*50}")
for model_name in MODELS:
    print(f"\n  {model_name}:")
    for nc in sorted(all_results[model_name].keys(), reverse=True):
        r = all_results[model_name][nc]
        print(f"    {nc:>2}ch  acc={r[real]:.3f}  p={r[p_value]:.4f}  {r[passed]}")
print("\nDONE")
