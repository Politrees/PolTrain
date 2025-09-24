import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
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


def mel_spectrogram_similarity(y_hat_mel, y_mel):
    """Сходство между сгенерированной и реальной мел-спектрограммами"""
    device = y_hat_mel.device
    y_mel = y_mel.to(device)

    if y_hat_mel.shape != y_mel.shape:
        trimmed_shape = tuple(min(dim_a, dim_b) for dim_a, dim_b in zip(y_hat_mel.shape, y_mel.shape))
        y_hat_mel = y_hat_mel[..., : trimmed_shape[-1]]
        y_mel = y_mel[..., : trimmed_shape[-1]]

    loss_mel = F.l1_loss(y_hat_mel, y_mel)
    mel_spec_similarity = 100.0 - (loss_mel * 100.0)
    return mel_spec_similarity.clamp(0.0, 100.0)
