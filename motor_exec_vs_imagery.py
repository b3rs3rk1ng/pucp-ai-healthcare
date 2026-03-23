import warnings; warnings.filterwarnings("ignore")
import numpy as np, time, pickle, mne
from mne.datasets import eegbci
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score

mne.set_log_level("ERROR")
np.random.seed(42); torch.manual_seed(42)
DEVICE = torch.device("cuda")
EEG_PATH = "/home/f3mt0/Documents/github/pucp-ai-healthcare/eeg-raw"
SUBJECTS = list(range(1, 110))
EXCLUDE = {88, 92, 100, 104}

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

def load_data(runs, label):
    all_epochs, all_labels, all_groups = [], [], []
    t0 = time.time()
    for subj in SUBJECTS:
        if subj in EXCLUDE: continue
        try:
            fnames = eegbci.load_data(subj, runs, path=EEG_PATH, update_path=True)
            raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames]
            for r in raws: eegbci.standardize(r)
            raw = mne.concatenate_raws(raws)
            raw.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="ignore")
            raw.filter(8., 30., fir_design="firwin", verbose=False)
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            eid_mi = {k:v for k,v in event_id.items() if k in ["T1","T2"]}
            if len(eid_mi) < 2: continue
            ep = mne.Epochs(raw, events, eid_mi, tmin=0., tmax=4.0, baseline=None, preload=True, verbose=False)
            ep.drop_bad(reject=dict(eeg=500e-6), verbose=False)
            if len(ep) < 5: continue
            d = ep.get_data()
            lab = (ep.events[:,2] == eid_mi["T2"]).astype(int)
            all_epochs.append(d); all_labels.append(lab); all_groups.append(np.full(len(lab), subj))
        except: pass
    X = np.concatenate(all_epochs); y = np.concatenate(all_labels); groups = np.concatenate(all_groups)
    X = (X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True) + 1e-8)
    X = X.astype(np.float32)
    print(f"  {label}: {X.shape[0]} trials, {len(np.unique(groups))} subjects [{time.time()-t0:.0f}s]")
    return X, y, groups

def train_eval_5fold(X, y, groups, nc):
    gkf = GroupKFold(n_splits=5)
    accs = []
    for fold, (tri, tei) in enumerate(gkf.split(X, y, groups)):
        Xtr, Xte = X[tri], X[tei]; ytr, yte = y[tri], y[tei]
        gtr = groups[tri]; usub = np.unique(gtr)
        vs = np.random.RandomState(fold).choice(usub, max(2, len(usub)//10), replace=False)
        vm = np.isin(gtr, vs)
        model = EEGNet(nc, X.shape[2]).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        ds = TensorDataset(torch.tensor(Xtr[~vm,None,:,:], dtype=torch.float32), torch.tensor(ytr[~vm], dtype=torch.long))
        dl = DataLoader(ds, batch_size=256, shuffle=True, pin_memory=True, num_workers=2)
        Xvt = torch.tensor(Xtr[vm,None,:,:], dtype=torch.float32).to(DEVICE)
        yvt = torch.tensor(ytr[vm], dtype=torch.long).to(DEVICE)
        best_vl = 999; best_st = None; wait = 0
        for ep in range(100):
            model.train()
            for xb, yb in dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
            model.eval()
            with torch.no_grad(): vl = crit(model(Xvt), yvt).item()
            if vl < best_vl: best_vl=vl; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}; wait=0
            else:
                wait += 1
                if wait >= 15: break
        model.load_state_dict(best_st); model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(Xte[:,None,:,:], dtype=torch.float32).to(DEVICE)).argmax(1).cpu().numpy()
        accs.append(accuracy_score(yte, preds))
        print(f"    F{fold+1} acc={accs[-1]:.3f}")
    return accs

print("=== MOTOR EXECUTION vs IMAGERY ===\n")

print("Loading imagery (runs 4, 8, 12)...")
X_img, y_img, g_img = load_data([4, 8, 12], "Imagery")

print("\nLoading execution (runs 3, 7, 11)...")
X_exe, y_exe, g_exe = load_data([3, 7, 11], "Execution")

ch_names = list(mne.io.read_raw_edf(eegbci.load_data(1, [3], path=EEG_PATH, update_path=True)[0], preload=False, verbose=False).ch_names)
eegbci.standardize(mne.io.read_raw_edf(eegbci.load_data(1, [3], path=EEG_PATH, update_path=True)[0], preload=False, verbose=False))

print("\n--- Imagery (64ch) ---")
accs_img = train_eval_5fold(X_img, y_img, g_img, 64)

print("\n--- Execution (64ch) ---")
accs_exe = train_eval_5fold(X_exe, y_exe, g_exe, 64)

print("\n--- Imagery (2ch: C3+C4) ---")
c3c4 = [ch_names.index("C3.."), ch_names.index("C4..")]
accs_img2 = train_eval_5fold(X_img[:, c3c4, :], y_img, g_img, 2)

print("\n--- Execution (2ch: C3+C4) ---")
accs_exe2 = train_eval_5fold(X_exe[:, c3c4, :], y_exe, g_exe, 2)

print("\n" + "="*50)
print("RESULTADOS")
print("="*50)
mi = np.mean(accs_img); me = np.mean(accs_exe)
mi2 = np.mean(accs_img2); me2 = np.mean(accs_exe2)
print(f"                      64ch    2ch(C3+C4)")
print(f"  Imagery:           {mi:.3f}      {mi2:.3f}")
print(f"  Execution:         {me:.3f}      {me2:.3f}")
print(f"  Diferencia:       {me-mi:+.3f}     {me2-mi2:+.3f}")

results = {"imagery_64": accs_img, "execution_64": accs_exe, "imagery_2": accs_img2, "execution_2": accs_exe2}
pickle.dump(results, open("motor_exec_vs_imagery_results.pkl", "wb"))
print("\nGuardado: motor_exec_vs_imagery_results.pkl")
