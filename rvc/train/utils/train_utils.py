import glob
import os
import traceback
from collections import OrderedDict

import torch


def replace_keys_in_dict(d, old_key_part, new_key_part):
    updated_dict = OrderedDict() if isinstance(d, OrderedDict) else {}
    for key, value in d.items():
        new_key = key.replace(old_key_part, new_key_part) if isinstance(key, str) else key
        updated_dict[new_key] = replace_keys_in_dict(value, old_key_part, new_key_part) if isinstance(value, dict) else value
    return updated_dict


def save_checkpoint_atomic(data, path):
    """Атомарное сохранение чекпоинта через временный файл."""
    temp_path = path + ".tmp"
    torch.save(data, temp_path)
    os.replace(temp_path, path)


def save_checkpoint(net_g, optim_g, net_d, optim_d, learning_rate, epoch, checkpoint_path):
    """Сохранение единого чекпоинта (G + D + optimizers)."""
    g_state = net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict()
    d_state = net_d.module.state_dict() if hasattr(net_d, "module") else net_d.state_dict()

    checkpoint_data = {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "generator": {
            "model": g_state,
            "optimizer": optim_g.state_dict(),
        },
        "discriminator": {
            "model": d_state,
            "optimizer": optim_d.state_dict(),
        },
    }

    checkpoint_data = replace_keys_in_dict(
        replace_keys_in_dict(checkpoint_data, ".parametrizations.weight.original1", ".weight_v"),
        ".parametrizations.weight.original0",
        ".weight_g",
    )

    save_checkpoint_atomic(checkpoint_data, checkpoint_path)
    print(f"Сохранён чекпоинт '{os.path.basename(checkpoint_path)}' (эпоха {epoch})", flush=True)


def load_unified_checkpoint(checkpoint_path, net_g, optim_g, net_d, optim_d):
    """Загрузка единого чекпоинта."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    checkpoint = replace_keys_in_dict(
        replace_keys_in_dict(checkpoint, ".weight_v", ".parametrizations.weight.original1"),
        ".weight_g",
        ".parametrizations.weight.original0",
    )

    # Загрузка генератора
    g_model = net_g.module if hasattr(net_g, "module") else net_g
    g_state_dict = g_model.state_dict()
    g_new_state = {k: checkpoint["generator"]["model"].get(k, v) for k, v in g_state_dict.items()}
    g_model.load_state_dict(g_new_state, strict=False)
    optim_g.load_state_dict(checkpoint["generator"]["optimizer"])

    # Загрузка дискриминатора
    d_model = net_d.module if hasattr(net_d, "module") else net_d
    d_state_dict = d_model.state_dict()
    d_new_state = {k: checkpoint["discriminator"]["model"].get(k, v) for k, v in d_state_dict.items()}
    d_model.load_state_dict(d_new_state, strict=False)
    optim_d.load_state_dict(checkpoint["discriminator"]["optimizer"])

    epoch = checkpoint["epoch"]
    print(f"Загружен чекпоинт '{os.path.basename(checkpoint_path)}' (эпоха {epoch})", flush=True)
    return epoch


def load_legacy_checkpoint(checkpoint_path, model, optimizer=None):
    """Загрузка старого формата чекпоинта (G или D отдельно)."""
    try:
        checkpoint_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    checkpoint_dict = replace_keys_in_dict(
        replace_keys_in_dict(checkpoint_dict, ".weight_v", ".parametrizations.weight.original1"),
        ".weight_g",
        ".parametrizations.weight.original0",
    )

    model_to_load = model.module if hasattr(model, "module") else model
    model_state_dict = model_to_load.state_dict()

    new_state_dict = {k: checkpoint_dict["model"].get(k, v) for k, v in model_state_dict.items()}
    model_to_load.load_state_dict(new_state_dict, strict=False)

    if optimizer:
        optimizer.load_state_dict(checkpoint_dict.get("optimizer", {}))

    return checkpoint_dict.get("iteration", checkpoint_dict.get("epoch", 0))


def load_legacy_checkpoints(model_dir, net_g, optim_g, net_d, optim_d):
    """Загрузка старых G/D чекпоинтов с fallback на бэкапы."""
    checkpoint_pairs = [
        ("G_checkpoint.pth", "D_checkpoint.pth"),
        ("G_checkpoint_backup.pth", "D_checkpoint_backup.pth"),
    ]

    for g_file, d_file in checkpoint_pairs:
        g_path = os.path.join(model_dir, g_file)
        d_path = os.path.join(model_dir, d_file)

        if os.path.exists(g_path) and os.path.exists(d_path):
            try:
                epoch_g = load_legacy_checkpoint(g_path, net_g, optim_g)
                epoch_d = load_legacy_checkpoint(d_path, net_d, optim_d)

                if epoch_g != epoch_d:
                    print(f"Несоответствие эпох: G={epoch_g}, D={epoch_d}. Пробуем бэкап...", flush=True)
                    continue

                print(f"Загружены чекпоинты '{g_file}' и '{d_file}' (эпоха {epoch_g})", flush=True)
                return epoch_g
            except Exception as e:
                print(f"Ошибка загрузки {g_file}/{d_file}: {e}. Пробуем бэкап...", flush=True)
                continue

    return None


def attempt_load_checkpoint(net_g, optim_g, net_d, optim_d, model_dir):
    """
    Универсальная загрузка чекпоинтов с обратной совместимостью.
    
    Приоритет:
    1. checkpoint.pth (новый единый формат)
    2. G_checkpoint.pth + D_checkpoint.pth (старый формат)
    3. *_backup.pth (бэкапы старого формата)
    
    Returns:
        epoch (int) или None если ничего не загрузилось
    """
    unified_path = os.path.join(model_dir, "checkpoint.pth")

    # 1. Пробуем новый единый формат
    if os.path.exists(unified_path):
        try:
            return load_unified_checkpoint(unified_path, net_g, optim_g, net_d, optim_d)
        except Exception as e:
            print(f"Ошибка загрузки checkpoint.pth: {e}", flush=True)

    # 2. Пробуем старый формат (G/D отдельно)
    epoch = load_legacy_checkpoints(model_dir, net_g, optim_g, net_d, optim_d)
    if epoch is not None:
        print("\n⚠️  Обнаружен старый формат чекпоинтов (G/D раздельно).", flush=True)
        print("    После сохранения будет использоваться новый формат (checkpoint.pth).", flush=True)
        print("    Старые файлы можно удалить вручную.\n", flush=True)
        return epoch

    return None


def extract_model(hps, ckpt, epoch, step, filepath):
    """
    Извлекает и сохраняет модель для инференса.
    
    Args:
        hps: Гиперпараметры модели.
        ckpt: State dict модели (генератора).
        epoch: Номер эпохи.
        step: Номер шага.
        filepath: Полный путь для сохранения файла.
    
    Returns:
        Сообщение об успехе или ошибке.
    """
    try:
        opt = OrderedDict(weight={key: value.half() for key, value in ckpt.items() if "enc_q" not in key})
        opt["config"] = [
            hps.data.filter_length // 2 + 1,
            32,
            hps.model.inter_channels,
            hps.model.hidden_channels,
            hps.model.filter_channels,
            hps.model.n_heads,
            hps.model.n_layers,
            hps.model.kernel_size,
            hps.model.p_dropout,
            hps.model.resblock,
            hps.model.resblock_kernel_sizes,
            hps.model.resblock_dilation_sizes,
            hps.model.upsample_rates,
            hps.model.upsample_initial_channel,
            hps.model.upsample_kernel_sizes,
            hps.model.spk_embed_dim,
            hps.model.gin_channels,
            hps.data.sample_rate,
        ]

        # Основные метаданные модели
        opt["model_name"] = hps.model_name
        opt["epoch"] = epoch
        opt["step"] = step
        opt["sr"] = hps.data.sample_rate
        opt["f0"] = True
        opt["version"] = "v2"
        opt["vocoder"] = hps.model.vocoder

        # Дополнительные метаданные
        opt["learning_environment"] = "PolTrain"

        # Сохранение модели
        torch.save(
            replace_keys_in_dict(
                replace_keys_in_dict(opt, ".parametrizations.weight.original1", ".weight_v"),
                ".parametrizations.weight.original0",
                ".weight_g",
            ),
            filepath,
        )

        filename = os.path.basename(filepath)
        return f"Модель '{filename}' успешно сохранена!"
    except Exception:
        return f"Ошибка при сохранении модели: {traceback.format_exc()}"


class TrainingMonitor:
    """
    Монитор обучения с debiased EMA и отслеживанием перетренировки.
    
    Обеспечивает точное соответствие с графиками TensorBoard.
    Определяет статус обучения на основе деградации mel_sim:
    - Деградация 3%+ → предупреждение
    - Деградация 7%+ → перетренировка
    """

    # Фиксированные параметры
    WARMUP_EPOCHS = 100           # Эпох до начала отслеживания рекордов
    WARNING_THRESHOLD = 3.0       # Процент деградации для предупреждения
    OVERTRAINING_THRESHOLD = 7.0  # Процент деградации для перетренировки
    SMOOTHING = 0.987             # Коэффициент сглаживания EMA

    def __init__(self):
        # Debiased EMA состояние (как в TensorBoard)
        self.ema_numerator = {}
        self.ema_denominator = {}
        self.smoothed_values = {}

        # Рекорды (отслеживаются после warmup)
        self.best_values = {
            "metrics/mel_sim": {"value": -float("inf"), "epoch": 0},
        }

    def update(self, key: str, value: float, epoch: int) -> float:
        """
        Обновляет метрику и возвращает сглаженное значение (debiased EMA).
        """
        value = float(value)

        # Инициализация при первом вызове
        if key not in self.ema_numerator:
            self.ema_numerator[key] = 0.0
            self.ema_denominator[key] = 0.0

        # Debiased EMA (как в TensorBoard)
        self.ema_numerator[key] = self.ema_numerator[key] * self.SMOOTHING + value * (1.0 - self.SMOOTHING)
        self.ema_denominator[key] = self.ema_denominator[key] * self.SMOOTHING + (1.0 - self.SMOOTHING)
        smoothed = self.ema_numerator[key] / self.ema_denominator[key]
        self.smoothed_values[key] = smoothed

        # Обновление рекордов (только после warmup, только mel_sim)
        if epoch >= self.WARMUP_EPOCHS and key == "metrics/mel_sim":
            if smoothed >= self.best_values[key]["value"]:
                self.best_values[key] = {"value": smoothed, "epoch": epoch}

        return smoothed

    def get_smoothed(self, key: str) -> float:
        """Возвращает текущее сглаженное значение."""
        return self.smoothed_values.get(key, 0.0)

    def get_best(self, key: str) -> dict:
        """Возвращает информацию о рекорде."""
        return self.best_values.get(key, {"value": 0.0, "epoch": 0})

    def get_status(self, epoch: int) -> tuple:
        """
        Определяет статус обучения на основе деградации mel_sim.
        
        Логика:
        - mel_sim упал на 3%+ от рекорда → предупреждение
        - mel_sim упал на 7%+ от рекорда → перетренировка
        
        Returns:
            (status_code, message)
            status_code: "warmup", "normal", "warning", "overtraining"
        """
        if epoch < self.WARMUP_EPOCHS:
            return ("warmup", "")

        mel_best = self.best_values["metrics/mel_sim"]["value"]
        mel_current = self.smoothed_values.get("metrics/mel_sim", 0.0)

        # Защита от некорректных значений
        if mel_best <= 0 or mel_current <= 0:
            return ("normal", "")

        # Вычисляем деградацию в процентных пунктах
        degradation = mel_best - mel_current

        if degradation >= self.OVERTRAINING_THRESHOLD:
            return ("overtraining", f"🔴 Возможна перетренировка (Mel снизился на {degradation:.1f}%)")

        if degradation >= self.WARNING_THRESHOLD:
            return ("warning", f"⚠️ Качество начало снижаться (Mel снизился на {degradation:.1f}%)")

        return ("normal", "")

    def restore_from_tensorboard(self, log_dir: str, current_epoch: int):
        """
        Восстанавливает состояние EMA и рекордов из логов TensorBoard.
        """
        try:
            from tensorboard.backend.event_processing import event_accumulator

            ea = event_accumulator.EventAccumulator(log_dir, size_guidance={"scalars": 0})
            ea.Reload()

            if not ea.Tags().get("scalars"):
                return

            print("\nВосстановление метрик из TensorBoard...", flush=True)

            for tag in ea.Tags()["scalars"]:
                events = ea.Scalars(tag)
                if not events:
                    continue

                # Сортируем по шагам и фильтруем до текущей эпохи
                step_values = {e.step: float(e.value) for e in events if e.step < current_epoch}
                sorted_steps = sorted(step_values.keys())

                if not sorted_steps:
                    continue

                # Восстанавливаем EMA
                ema_n, ema_d = 0.0, 0.0
                for step in sorted_steps:
                    val = step_values[step]
                    ema_n = ema_n * self.SMOOTHING + val * (1.0 - self.SMOOTHING)
                    ema_d = ema_d * self.SMOOTHING + (1.0 - self.SMOOTHING)
                    smoothed = ema_n / ema_d

                    # Обновляем рекорд mel_sim (после warmup)
                    if step >= self.WARMUP_EPOCHS and tag == "metrics/mel_sim":
                        if smoothed >= self.best_values[tag]["value"]:
                            self.best_values[tag] = {"value": smoothed, "epoch": step}

                # Сохраняем финальное состояние EMA
                self.ema_numerator[tag] = ema_n
                self.ema_denominator[tag] = ema_d
                self.smoothed_values[tag] = ema_n / ema_d if ema_d > 0 else 0.0

            # Выводим восстановленные значения
            mel_smoothed = self.smoothed_values.get("metrics/mel_sim", 0.0)
            mel_best = self.best_values["metrics/mel_sim"]

            if mel_best["epoch"] > 0:
                print(f"Mel: {mel_smoothed:.2f}% | Рекорд: {mel_best['value']:.2f}% (эпоха {mel_best['epoch']})", flush=True)
            else:
                print(f"Mel: {mel_smoothed:.2f}%", flush=True)

        except ImportError:
            print("TensorBoard не установлен, пропуск восстановления метрик.", flush=True)
        except Exception as e:
            print(f"Ошибка восстановления из TensorBoard: {e}", flush=True)


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = HParams(**v) if isinstance(v, dict) else v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return repr(self.__dict__)
