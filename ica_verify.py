import warnings; warnings.filterwarnings("ignore")
import numpy as np, time, mne, torch, torch.nn as nn, torch.optim as optim, shap, pickle
from mne.datasets import eegbci
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score

mne.set_log_level("ERROR")
np.random.seed(42); torch.manual_seed(42)
DEVICE = torch.device("cuda")

# Step 1: Preprocess with ICA (remove eye artifacts)
print("=== STEP 1: ICA preprocessing ===")
SUBJECTS = list(range(1, 110))
EXCLUDE = {88, 92, 100, 104}
all_epochs, all_labels, all_groups = [], [], []
t0 = time.time()

for subj in SUBJECTS:
    if subj in EXCLUDE: continue
    try:
        fnames = eegbci.load_data(subj, [4,8,12], path="/home/f3mt0/Documents/github/pucp-ai-healthcare/eeg-raw", update_path=True)
        raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames]
        for r in raws: eegbci.standardize(r)
        raw = mne.concatenate_raws(raws)
        raw.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="ignore")
        
        # Filter wider for ICA (1-40 Hz)
        raw.filter(1., 40., verbose=False)
        
        # Run ICA and remove eye components
        ica = mne.preprocessing.ICA(n_components=20, random_state=42, max_iter=300)
        ica.fit(raw, verbose=False)
        
        # Auto-detect eye components using frontal channels
        eog_indices = []
        for ch in ["Fp1", "Fp2", "F7", "F8"]:
            try:
                idx, scores = ica.find_bads_eog(raw, ch_name=ch, verbose=False)
                eog_indices.extend(idx)
            except:
                pass
        eog_indices = list(set(eog_indices))
        ica.exclude = eog_indices
        raw = ica.apply(raw, verbose=False)
        
        # Now bandpass to 8-30 Hz (same as original)
        raw.filter(8., 30., verbose=False)
        
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        eid_mi = {k:v for k,v in event_id.items() if k in ["T1","T2"]}
        if len(eid_mi) < 2: continue
        ep = mne.Epochs(raw, events, eid_mi, tmin=0., tmax=4.0, baseline=None, preload=True, verbose=False)
        ep.drop_bad(reject=dict(eeg=500e-6), verbose=False)
        if len(ep) < 5: continue
        d = ep.get_data()
        lab = (ep.events[:,2] == eid_mi["T2"]).astype(int)
        all_epochs.append(d); all_labels.append(lab); all_groups.append(np.full(len(lab), subj))
        if subj % 20 == 0:
            print(f"  S{subj} (removed {len(eog_indices)} ICA components) [{time.time()-t0:.0f}s]")
    except:
        pass

X = np.concatenate(all_epochs); y = np.concatenate(all_labels); groups = np.concatenate(all_groups)
ch_names = [str(c) for c in ep.ch_names]
X_ica = (X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True) + 1e-8)
X_ica = X_ica.astype(np.float32)
print(f"ICA data: {X_ica.shape}, {len(np.unique(groups))} subjects [{time.time()-t0:.0f}s]")

# Step 2: Train EEGNet on ICA-cleaned data
print("\n=== STEP 2: Train EEGNet (ICA-cleaned) ===")

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

gkf = GroupKFold(n_splits=5)
tri, tei = next(iter(gkf.split(X_ica, y, groups)))
Xtr = X_ica[tri]; ytr = y[tri]; gtr = groups[tri]
usub = np.unique(gtr)
vs = np.random.RandomState(0).choice(usub, max(2, len(usub)//10), replace=False)
vm = np.isin(gtr, vs)

model = EEGNet(64, X_ica.shape[2]).to(DEVICE)
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
    if vl < best_vl:
        best_vl = vl; best_st = {k:v.cpu().clone() for k,v in model.state_dict().items()}; wait = 0
    else:
        wait += 1
        if wait >= 15: break

model.load_state_dict(best_st); model.eval()
Xte = X_ica[tei]
with torch.no_grad():
    preds = model(torch.tensor(Xte[:,None,:,:], dtype=torch.float32).to(DEVICE)).argmax(1).cpu().numpy()
acc_ica = accuracy_score(y[tei], preds)
print(f"  Accuracy (ICA-cleaned): {acc_ica:.3f}")

# Step 3: SHAP on ICA-cleaned model
print("\n=== STEP 3: SHAP (ICA-cleaned) ===")
model_cpu = EEGNet(64, X_ica.shape[2])
model_cpu.load_state_dict(best_st)
model_cpu.eval()

X_bg = torch.tensor(X_ica[np.random.choice(len(X_ica), 200, replace=False), None, :, :], dtype=torch.float32)
X_test = torch.tensor(X_ica[np.random.choice(len(X_ica), 300, replace=False), None, :, :], dtype=torch.float32)
explainer = shap.GradientExplainer(model_cpu, X_bg)
sv = np.array(explainer.shap_values(X_test))
importance = np.abs(sv[:, :, :, :, 1]).mean(axis=(0, 1, 3))
ranked = [int(i) for i in np.argsort(importance)[::-1]]

print("\nTop 10 (after ICA):")
for i in range(10):
    print(f"  {i+1:>2}. {ch_names[ranked[i]]:>4} = {importance[ranked[i]]:.6f}")

c3r = ranked.index(ch_names.index("C3")) + 1
c4r = ranked.index(ch_names.index("C4")) + 1
f7r = ranked.index(ch_names.index("F7")) + 1
f8r = ranked.index(ch_names.index("F8")) + 1

print(f"\nC3: #{c3r}/64, C4: #{c4r}/64")
print(f"F7: #{f7r}/64, F8: #{f8r}/64")

# Load original SHAP for comparison
orig = np.load("shap_results.npz", allow_pickle=True)
orig_ranked = [int(i) for i in orig["ranked"]]
print(f"\n=== COMPARACION ===")
print(f"Sin ICA top 5: {[ch_names[orig_ranked[i]] for i in range(5)]}")
print(f"Con ICA top 5: {[ch_names[ranked[i]] for i in range(5)]}")

if f7r > 10 and f8r > 10:
    print("\n-> F7/F8 BAJARON tras ICA. ERAN ARTEFACTOS OCULARES.")
else:
    print("\n-> F7/F8 SIGUEN ARRIBA tras ICA. ES SENAL LEGITIMA.")

pickle.dump({"importance": importance, "ranked": ranked, "acc_ica": acc_ica}, open("ica_shap_results.pkl", "wb"))
print("\nGuardado: ica_shap_results.pkl")
