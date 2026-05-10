import logging
import os
import sys
import warnings

# Настройка окружения
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Настройка логирования и подавление предупреждений
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")

import argparse
import datetime
import json
import pathlib
import glob
import re
from collections import defaultdict
from distutils.util import strtobool
from random import randint
from time import time as ttime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.getcwd()))
from rvc.lib.algorithm.commons import grad_norm, slice_segments
from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.train.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from rvc.train.mel_processing import MultiScaleMelSpectrogramLoss, mel_spectrogram_torch, spec_to_mel_torch
from rvc.train.utils.data_utils import DistributedBucketSampler, TextAudioCollateMultiNSFsid, TextAudioLoaderMultiNSFsid
from rvc.train.utils.train_utils import HParams, load_checkpoint, save_checkpoint
from rvc.train.visualization import mel_spectrogram_similarity, plot_spectrogram_to_numpy

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

global_step = 0


class MetricsAccumulator:
    """Аккумулятор метрик для вычисления средних значений за эпоху."""

    def __init__(self):
        self.sums = defaultdict(float)
        self.count = 0

    def update(self, **kwargs):
        self.count += 1
        for k, v in kwargs.items():
            self.sums[k] += v.item() if hasattr(v, "item") else v

    def average(self):
        return {k: v / self.count for k, v in self.sums.items()} if self.count else {}


def generate_config(config_save_path, optimizer):
    config_path = os.path.join("rvc", "configs", f"48000.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            with open(config_path, "r", encoding="utf-8") as config_file:
                config_data = json.load(config_file)
                config_data["model"]["optimizer"] = optimizer
                json.dump(config_data, f, ensure_ascii=False, indent=2)


def get_hparams():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--total_epoch", type=int, choices=range(1, 10001), default=300)
    parser.add_argument("--save_every_epoch", type=int, choices=range(1, 101), default=25)
    parser.add_argument("--batch_size", type=int, choices=range(1, 129), default=8)
    parser.add_argument("--optimizer", type=str, choices=["AdamW", "AdaBelief"], default="AdamW")
    parser.add_argument("--pretrain_g", type=str, default=None)
    parser.add_argument("--pretrain_d", type=str, default=None)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--half_precision", type=lambda x: bool(strtobool(x)), choices=[True, False], default=True)
    args = parser.parse_args()

    experiment_dir = os.path.join(args.experiment_dir, args.model_name)
    config_save_path = os.path.join(experiment_dir, "data", "config.json")

    # Генерация файла конфигурации
    if not os.path.exists(config_save_path):
        generate_config(config_save_path, args.optimizer)

    # Загрузка файла конфигурации
    with open(config_save_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    hparams = HParams(**config)
    hparams.model_dir = experiment_dir
    hparams.model_name = args.model_name
    hparams.total_epoch = args.total_epoch
    hparams.save_every_epoch = args.save_every_epoch
    hparams.batch_size = args.batch_size
    hparams.pretrain_g = args.pretrain_g
    hparams.pretrain_d = args.pretrain_d
    hparams.gpus = args.gpus
    hparams.half_precision = args.half_precision
    hparams.data.training_files = f"{experiment_dir}/data/filelist.txt"

    print("\n\nПАРАМЕТРЫ ОБУЧЕНИЯ ")
    print("=" * 70)
    print(f"{'Папка сохранения:':<30} {hparams.model_dir}")
    print(f"{'Имя модели:':<30} {hparams.model_name}")
    print(f"{'Эпох обучения:':<30} {hparams.total_epoch}")
    print(f"{'Сохранение каждые:':<30} {hparams.save_every_epoch} эпох")
    print(f"{'Размер батча:':<30} {hparams.batch_size}")
    print(f"{'Частота дискретизации:':<30} {hparams.data.sample_rate} Hz")
    print(f"{'Вокодер:':<30} {hparams.model.vocoder}")
    print(f"{'Оптимизатор:':<30} {hparams.model.optimizer}")
    if args.pretrain_g:
        print(f"{'Pretrain G:':<30} {hparams.pretrain_g}")
    if args.pretrain_d:
        print(f"{'Pretrain D:':<30} {hparams.pretrain_d}")
    print(f"{'Точность обучения:':<30} {'float16 (AMP)' if hparams.half_precision else 'float32'}")
    print(f"{'Сохранение моделей:':<30} {'float16' if hparams.half_precision else 'float32'}")
    print("=" * 70 + "\n")
    return hparams


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = round(now_time - self.last_time, 1)
        self.last_time = now_time
        return f"[{str(datetime.timedelta(seconds=int(elapsed_time)))}]"


def main():
    hps = get_hparams()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))

    if not torch.cuda.is_available():
        print("CUDA недоступна! Обучение требует GPU с поддержкой CUDA.", flush=True)
        sys.exit(1)

    gpus = [int(item) for item in hps.gpus.split("-")]
    n_gpus = len(gpus)

    children = []
    for rank, device_id in enumerate(gpus):
        subproc = mp.Process(
            target=run,
            args=(hps, rank, n_gpus, device_id),
        )
        children.append(subproc)
        subproc.start()

    for subproc in children:
        subproc.join()

    sys.exit(0)


def run(hps, rank, n_gpus, device_id):
    global global_step

    try:
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval")) if rank == 0 else None
        fn_mel_loss = MultiScaleMelSpectrogramLoss(sample_rate=hps.data.sample_rate)

        dist.init_process_group(backend="nccl", init_method="env://", world_size=n_gpus, rank=rank)

        torch.manual_seed(hps.train.seed)
        torch.cuda.set_device(device_id)

        # Определяем dtype на основе half_precision
        use_amp = hps.half_precision
        dtype = torch.float16 if use_amp else torch.float32

        collate_fn = TextAudioCollateMultiNSFsid()
        train_dataset = TextAudioLoaderMultiNSFsid(hps.data)
        train_sampler = DistributedBucketSampler(
            train_dataset,
            hps.batch_size,
            [50, 100, 200, 300, 400, 500, 600, 700, 800, 900],
            num_replicas=n_gpus,
            rank=rank,
            shuffle=True,
        )
        train_loader = DataLoader(
            train_dataset,
            num_workers=2,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            batch_sampler=train_sampler,
            persistent_workers=True,
            prefetch_factor=8,
        )

        net_g = Synthesizer(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            sr=hps.data.sample_rate,
            checkpointing=False,
            randomized=True,
        )
        net_d = MultiPeriodDiscriminator(checkpointing=False)

        net_g = net_g.cuda(device_id)
        net_d = net_d.cuda(device_id)

        if hps.model.optimizer == "AdaBelief":
            from rvc.train.utils.optimizers.AdaBelief import AdaBelief

            optim_g = AdaBelief(net_g.parameters(), lr=hps.train.learning_rate, betas=hps.train.betas, eps=1e-8)
            optim_d = AdaBelief(net_d.parameters(), lr=hps.train.learning_rate, betas=hps.train.betas, eps=1e-8)
        else:
            optim_g = torch.optim.AdamW(net_g.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
            optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

        if n_gpus > 1:
            net_g = DDP(net_g, device_ids=[device_id])
            net_d = DDP(net_d, device_ids=[device_id])

        # Инициализация GradScaler для AMP
        scaler_g = GradScaler(enabled=use_amp)
        scaler_d = GradScaler(enabled=use_amp)

        # Загрузка чекпоинта
        epoch_str = None
        checkpoint_files = glob.glob(os.path.join(hps.model_dir, "checkpoint_e*.pth"))
        if checkpoint_files:
            def extract_epoch(filepath):
                filename = os.path.basename(filepath)
                match = re.search(r"checkpoint_e(\d+)\.pth", filename)
                return int(match.group(1)) if match else -1

            checkpoint_files.sort(key=extract_epoch, reverse=True)
            for ckpt_path in checkpoint_files:
                if rank == 0:
                    print(f"Попытка загрузки чекпоинта: '{os.path.basename(ckpt_path)}'...", flush=True)
                try:
                    epoch_str = load_checkpoint(ckpt_path, net_g, optim_g, net_d, optim_d)
                    break
                except Exception as e:
                    if rank == 0:
                        print(f"⚠️ Ошибка при загрузке '{os.path.basename(ckpt_path)}':\n{e}")
                        print("Переход к предыдущему чекпоинту...\n", flush=True)
                    epoch_str = None

        if epoch_str is not None:
            epoch_str += 1
            global_step = (epoch_str - 1) * len(train_loader)
        else:
            epoch_str = 1
            global_step = 0

            # Загрузка претрейнов если чекпоинт не найден
            if hps.pretrain_g not in ("", "None", None):
                if rank == 0:
                    print(f"Загрузка претрейна генератора: '{hps.pretrain_g}'", flush=True)
                g_model = net_g.module if hasattr(net_g, "module") else net_g
                try:
                    g_model.load_state_dict(torch.load(hps.pretrain_g, map_location="cpu", weights_only=True)["model"])
                except Exception:
                    print("Загрузка претрейна генератора в небезопасном режиме...", flush=True)
                    g_model.load_state_dict(torch.load(hps.pretrain_g, map_location="cpu", weights_only=False)["model"])

            if hps.pretrain_d not in ("", "None", None):
                if rank == 0:
                    print(f"Загрузка претрейна дискриминатора: '{hps.pretrain_d}'", flush=True)
                d_model = net_d.module if hasattr(net_d, "module") else net_d
                try:
                    d_model.load_state_dict(torch.load(hps.pretrain_d, map_location="cpu", weights_only=True)["model"])
                except Exception:
                    print("Загрузка претрейна дискриминатора в небезопасном режиме...", flush=True)
                    d_model.load_state_dict(torch.load(hps.pretrain_d, map_location="cpu", weights_only=False)["model"])

        # Настройка scheduler
        if hps.model.optimizer == "AdaBelief":
            scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optim_g, T_max=hps.total_epoch, eta_min=1e-6, last_epoch=epoch_str - 2)
            scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optim_d, T_max=hps.total_epoch, eta_min=1e-6, last_epoch=epoch_str - 2)
        else:
            scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
            scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

        # Проверка: не превышает ли загруженная эпоха целевую
        if epoch_str > hps.total_epoch:
            if rank == 0:
                print(
                    f"\n⚠️  Загруженный чекпоинт (эпоха {epoch_str - 1}) уже превышает указанное количество эпох ({hps.total_epoch}).",
                    flush=True,
                )
            return

        print("\nЗапуск процесса обучения модели...", flush=True)
        epoch_recorder = EpochRecorder() if rank == 0 else None
        for epoch in range(epoch_str, hps.total_epoch + 1):
            train_and_evaluate(
                hps,
                rank,
                epoch,
                [net_g, net_d],
                [optim_g, optim_d],
                [scaler_g, scaler_d],
                train_loader,
                writer_eval,
                fn_mel_loss,
                device_id,
                epoch_recorder,
                use_amp,
            )
            scheduler_g.step()
            scheduler_d.step()
    finally:
        # Уничтожение группы процессов для корректного закрытия программы
        if dist.is_initialized():
            dist.destroy_process_group()


def train_and_evaluate(hps, rank, epoch, nets, optims, scalers, train_loader, writer_eval, fn_mel_loss, device_id, epoch_recorder=None, use_amp=False):
    global global_step

    net_g, net_d = nets
    optim_g, optim_d = optims
    scaler_g, scaler_d = scalers
    train_loader.batch_sampler.set_epoch(epoch)

    net_g.train()
    net_d.train()

    acc = MetricsAccumulator()
    last_batch = None

    for batch_idx, info in enumerate(train_loader):
        info = [tensor.cuda(device_id, non_blocking=True) for tensor in info]
        phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, _, sid = info

        with autocast(enabled=use_amp):
            model_output = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
            y_hat, ids_slice, _, z_mask, (_, z_p, m_p, logs_p, _, logs_q) = model_output
            wave = slice_segments(wave, ids_slice * hps.data.hop_length, hps.train.segment_size, dim=3)

        # Discriminator loss
        for _ in range(1):
            with autocast(enabled=use_amp):
                y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
                loss_disc = discriminator_loss(y_d_hat_r, y_d_hat_g)

            scaler_d.scale(loss_disc).backward()
            scaler_d.unscale_(optim_d)
            grad_norm_d = grad_norm(net_d.parameters())
            scaler_d.step(optim_d)
            scaler_d.update()
            optim_d.zero_grad()

        # Generator loss
        for _ in range(1):
            with autocast(enabled=use_amp):
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
                loss_mel = fn_mel_loss(wave, y_hat) * hps.train.c_mel / 3.0
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

            scaler_g.scale(loss_gen_all).backward()
            scaler_g.unscale_(optim_g)
            grad_norm_g = grad_norm(net_g.parameters())
            scaler_g.step(optim_g)
            scaler_g.update()
            optim_g.zero_grad()

        # Аккумуляция метрик
        acc.update(**{
            "loss/avg/d": loss_disc,
            "loss/avg/g": loss_gen,
            "loss/g/fm": loss_fm,
            "loss/g/mel": loss_mel,
            "loss/g/kl": loss_kl,
            "loss/g/total": loss_gen_all,
            "grad/norm_d": grad_norm_d,
            "grad/norm_g": grad_norm_g,
        })

        # Сохраняем данные последнего батча для визуализации
        if rank == 0:
            last_batch = (spec, ids_slice, y_hat)

        global_step += 1

    if rank == 0 and epoch % hps.train.log_interval == 0:
        avg = acc.average()

        spec, ids_slice, y_hat = last_batch
        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sample_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )
        y_mel = slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length, dim=3)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.float().squeeze(1),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sample_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )
        mel_similarity = mel_spectrogram_similarity(y_hat_mel, y_mel)

        scalar_dict = {
            **avg,
            "metrics/mel_sim": mel_similarity,
            "Learning Rate/G": optim_g.param_groups[0]["lr"],
            "Learning Rate/D": optim_d.param_groups[0]["lr"],
        }
        image_dict = {
            "mel/slice/real": plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "mel/slice/fake": plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
        }
        for k, v in scalar_dict.items():
            writer_eval.add_scalar(k, v, epoch)
        for k, v in image_dict.items():
            writer_eval.add_image(k, v, epoch, dataformats="HWC")

    # Вывод в консоль
    if rank == 0:
        print(
            f"{epoch_recorder.record()}: {hps.model_name} ▸ "
            f"Эпоха {epoch}/{hps.total_epoch} (Шаг {global_step})",
            flush=True,
        )

    # Сохранение моделей
    if rank == 0:
        if epoch % hps.save_every_epoch == 0:
            checkpoint_path = os.path.join(hps.model_dir, f"checkpoint_e{epoch}.pth")
            save_checkpoint(net_g, optim_g, net_d, optim_d, epoch, checkpoint_path)

        # Финальная эпоха
        if epoch >= hps.total_epoch:
            print("\nОбучение успешно завершено!", flush=True)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
