"""Preprocesamiento de EEG: carga EDF raw, filtra, epocha, normaliza."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import mne
from mne.datasets import eegbci
import argparse, time, os

mne.set_log_level('ERROR')

SUBJECTS = list(range(1, 110))
EXCLUDE = {88, 92, 100, 104}


def preprocess(low_freq=8.0, high_freq=30.0, tmin=0.0, tmax=4.0,
               reject_uv=500e-6, data_path='./data'):
    """Carga y preprocesa EEG de PhysioNet eegmmidb.

    Args:
        low_freq: frecuencia baja del bandpass (Hz)
        high_freq: frecuencia alta del bandpass (Hz)
        tmin: inicio del epoch (s)
        tmax: fin del epoch (s)
        reject_uv: umbral de rechazo de artefactos (V)
        data_path: directorio donde están/se descargan los EDF

    Returns:
        X_norm, y, groups, ch_names
    """
    all_epochs, all_labels, all_groups = [], [], []
    t0 = time.time()

    for subj in SUBJECTS:
        if subj in EXCLUDE:
            continue
        try:
            fnames = eegbci.load_data(subj, [4, 8, 12], path=data_path, update_path=True)
            raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames]
            for r in raws:
                eegbci.standardize(r)
            raw = mne.concatenate_raws(raws)
            raw.set_montage(mne.channels.make_standard_montage('standard_1005'), on_missing='ignore')
            raw.filter(low_freq, high_freq, fir_design='firwin', verbose=False)

            events, event_id = mne.events_from_annotations(raw, verbose=False)
            eid_mi = {k: v for k, v in event_id.items() if k in ['T1', 'T2']}
            if len(eid_mi) < 2:
                continue

            ep = mne.Epochs(raw, events, eid_mi, tmin=tmin, tmax=tmax,
                           baseline=None, preload=True, verbose=False)
            ep.drop_bad(reject=dict(eeg=reject_uv), verbose=False)
            if len(ep) < 5:
                continue

            d = ep.get_data()
            lab = (ep.events[:, 2] == eid_mi['T2']).astype(int)
            all_epochs.append(d)
            all_labels.append(lab)
            all_groups.append(np.full(len(lab), subj))

            if subj % 20 == 0:
                print(f"  S{subj} [{time.time()-t0:.0f}s]")
        except:
            pass

    X = np.concatenate(all_epochs)
    y = np.concatenate(all_labels)
    groups = np.concatenate(all_groups)
    ch_names = list(ep.ch_names)

    # Z-score por epoch y canal
    X_norm = (X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True) + 1e-8)
    X_norm = X_norm.astype(np.float32)

    print(f"Listo: {X_norm.shape[0]} epochs, {len(np.unique(groups))} sujetos [{time.time()-t0:.0f}s]")
    return X_norm, y, groups, ch_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocesamiento EEG')
    parser.add_argument('--output', default='data_f32.npz', help='archivo de salida')
    parser.add_argument('--low', type=float, default=8.0, help='freq baja (Hz)')
    parser.add_argument('--high', type=float, default=30.0, help='freq alta (Hz)')
    parser.add_argument('--data-path', default='./data', help='directorio EDF')
    args = parser.parse_args()

    X_norm, y, groups, ch_names = preprocess(
        low_freq=args.low, high_freq=args.high, data_path=args.data_path
    )
    np.savez_compressed(args.output, X_norm=X_norm, y=y, groups=groups, ch_names=ch_names)
    size = os.path.getsize(args.output) / 1024 / 1024
    print(f"Guardado: {args.output} ({size:.0f} MB)")
