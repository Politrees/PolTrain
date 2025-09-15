import matplotlib.pyplot as plt
import numpy as np
import torch
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


def mel_spectrogram_similarity(y_hat_mel, y_mel, epsilon=1e-8):
    """Вычисляет сходство между сгенерированной и реальной мел-спектрограммами."""
    device = y_hat_mel.device
    y_mel = y_mel.to(device)

    # Выравнивание размеров по всем измерениям
    if y_hat_mel.shape != y_mel.shape:
        min_batch = min(y_hat_mel.shape[0], y_mel.shape[0])
        min_channels = min(y_hat_mel.shape[1], y_mel.shape[1]) if len(y_hat_mel.shape) > 2 else 1
        min_freq = min(y_hat_mel.shape[-2], y_mel.shape[-2]) if len(y_hat_mel.shape) > 1 else 1
        min_time = min(y_hat_mel.shape[-1], y_mel.shape[-1])

        if len(y_hat_mel.shape) == 4:  # [batch, channels, freq, time]
            y_hat_mel = y_hat_mel[:min_batch, :min_channels, :min_freq, :min_time]
            y_mel = y_mel[:min_batch, :min_channels, :min_freq, :min_time]
        elif len(y_hat_mel.shape) == 3:  # [batch, freq, time]
            y_hat_mel = y_hat_mel[:min_batch, :min_freq, :min_time]
            y_mel = y_mel[:min_batch, :min_freq, :min_time]
        else:  # [freq, time]
            y_hat_mel = y_hat_mel[:min_freq, :min_time]
            y_mel = y_mel[:min_freq, :min_time]

    # Нормализованная L1 метрика
    loss_mel = torch.nn.functional.l1_loss(y_hat_mel, y_mel)

    # Нормализация относительно среднего абсолютного значения реальной спектрограммы
    norm_factor = torch.mean(torch.abs(y_mel)) + epsilon
    normalized_loss = loss_mel / norm_factor

    # Преобразование в процент сходства
    mel_spec_similarity = 100.0 * torch.exp(-normalized_loss)
    return mel_spec_similarity.clamp(0.0, 100.0)
