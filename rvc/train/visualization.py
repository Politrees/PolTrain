import math

import matplotlib.pyplot as plt
import numpy as np
import torch
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


def _estimate_f0_autocorr(wav, sample_rate, hop_length, f0_min=50.0, f0_max=1100.0, frame_ms=40.0):
    """Быстрая оценка F0 через автокорреляцию для онлайн-метрики."""
    wav = wav.squeeze(1) if wav.dim() == 3 else wav
    original_device = wav.device

    # На MPS torch.fft в некоторых версиях PyTorch может быть нестабилен/недоступен.
    wav = wav.detach().float().cpu() if wav.device.type == "mps" else wav.detach().float()

    frame_length = int(sample_rate * frame_ms / 1000.0)
    frame_length = max(frame_length, int(sample_rate / f0_min) * 2, hop_length * 2)
    frame_length = min(frame_length, wav.size(-1))
    if frame_length <= 4 or wav.size(-1) < frame_length:
        return wav.new_zeros((wav.size(0), 1)).to(original_device)

    frames = wav.unfold(-1, frame_length, hop_length)
    frames = frames - frames.mean(dim=-1, keepdim=True)
    window = torch.hann_window(frame_length, device=wav.device, dtype=wav.dtype)
    frames = frames * window

    rms = frames.pow(2).mean(dim=-1).sqrt()
    n_fft = 1 << int(math.ceil(math.log2(frame_length * 2 - 1)))
    spectrum = torch.fft.rfft(frames, n=n_fft)
    autocorr = torch.fft.irfft(spectrum * spectrum.conj(), n=n_fft)[..., :frame_length]

    min_lag = max(1, int(sample_rate / f0_max))
    max_lag = min(frame_length - 1, int(sample_rate / f0_min))
    if max_lag <= min_lag:
        return wav.new_zeros(rms.shape).to(original_device)

    peak_region = autocorr[..., min_lag : max_lag + 1]
    peak_values, peak_indices = peak_region.max(dim=-1)
    lag = peak_indices + min_lag
    f0 = sample_rate / lag.clamp_min(1).float()

    periodicity = peak_values / autocorr[..., 0].clamp_min(1e-8)
    voiced = (rms > 0.003) & (periodicity > 0.25)
    return torch.where(voiced, f0, torch.zeros_like(f0)).to(original_device)


def _slice_pitch(pitchf, ids_slice, segment_frames, device):
    """Берет тот же срез pitchf, что и аудио-сегмент для y_hat."""
    target = pitchf.unsqueeze(1)
    target_slices = []
    for idx, start in enumerate(ids_slice.tolist()):
        target_slice = target[idx : idx + 1, :, start : start + segment_frames]
        if target_slice.size(-1) < segment_frames:
            target_slice = F.pad(target_slice, (0, segment_frames - target_slice.size(-1)))
        target_slices.append(target_slice)
    return torch.cat(target_slices, dim=0).squeeze(1).to(device).float()


def f0_error_cents(y_hat, pitchf, ids_slice, sample_rate, hop_length, segment_frames):
    """
    Средняя ошибка F0 в центах между сгенерированным аудио и pitchf датасета.

    Меньше = лучше. 100 cents = 1 полутон. Метрика считается только на voiced
    кадрах, где и y_hat, и target pitchf имеют ненулевой F0.
    """
    fake_f0 = _estimate_f0_autocorr(y_hat, sample_rate, hop_length)
    target_f0 = _slice_pitch(pitchf, ids_slice, segment_frames, fake_f0.device)

    frames = min(fake_f0.size(-1), target_f0.size(-1))
    fake_f0 = fake_f0[..., :frames]
    target_f0 = target_f0[..., :frames]

    voiced_mask = (fake_f0 > 0.0) & (target_f0 > 0.0)
    if not voiced_mask.any():
        return torch.as_tensor(2400.0, device=y_hat.device)

    cents_error = 1200.0 * torch.abs(
        torch.log2(fake_f0[voiced_mask].clamp_min(1e-5) / target_f0[voiced_mask].clamp_min(1e-5))
    )
    return cents_error.clamp(max=2400.0).mean()
