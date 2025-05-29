import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from librosa.display import specshow
from matplotlib.colors import Normalize


def plot_spectrogram_to_numpy(spectrogram, figsize=(10, 4), cmap="viridis"):
    """Визуализация Mel-спектрограммы."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", cmap=cmap, norm=Normalize(vmin=-10, vmax=0))
    plt.colorbar(im, ax=ax, format="%+2.0f dB")
    plt.xlabel("Кадры")
    plt.ylabel("Частотные каналы")
    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    data = np.asarray(buf, dtype=np.uint8)
    plt.close(fig)
    return data


def plot_pitch_to_numpy(pitch_values, figsize=(12, 3)):
    """Визуализация кривой pitch (F0)."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(pitch_values, color="r", linewidth=1, label="Pitch")
    ax.set_xlabel("Кадры")
    ax.set_ylabel("Частота (Гц)")
    ax.legend(loc="upper right")
    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    data = np.asarray(buf, dtype=np.uint8)
    plt.close(fig)
    return data
