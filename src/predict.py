"""Inferencia: carga modelo y predice sobre señal EEG."""
import torch
import numpy as np
import argparse
from model import EEGNet


def predict(model_path, signal):
    """Predice izquierda/derecha desde una señal EEG.

    Args:
        model_path: ruta al .pt guardado
        signal: array (n_channels, n_times)

    Returns:
        dict con prediction, confidence, probabilities
    """
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    n_ch = checkpoint['n_channels']
    n_times = checkpoint['n_times']

    model = EEGNet(n_ch, n_times)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Normalizar señal
    signal = signal.astype(np.float32)
    signal = (signal - signal.mean(axis=1, keepdims=True)) / (signal.std(axis=1, keepdims=True) + 1e-8)

    x = torch.tensor(signal[None, None, :, :], dtype=torch.float32)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, 1)[0].numpy()

    pred = 'LEFT' if probs[0] > probs[1] else 'RIGHT'
    conf = max(probs)

    return {'prediction': pred, 'confidence': float(conf),
            'prob_left': float(probs[0]), 'prob_right': float(probs[1])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediccion EEG Motor Imagery')
    parser.add_argument('--model', required=True, help='ruta al modelo .pt')
    parser.add_argument('--input', required=True, help='archivo .npy con la señal')
    args = parser.parse_args()

    signal = np.load(args.input)
    result = predict(args.model, signal)
    print(f"Prediccion: {result['prediction']} (confianza: {result['confidence']:.1%})")
    print(f"  P(left):  {result['prob_left']:.3f}")
    print(f"  P(right): {result['prob_right']:.3f}")
