import logging
import os
import sys
import warnings

# Настройка окружения
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_LIBUV"] = "0" if sys.platform == "win32" else "1"

# Настройка логирования и подавление предупреждений
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")

import argparse
import datetime
import json
import pathlib
from distutils.util import strtobool
from random import randint
from time import sleep
from time import time as ttime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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
from rvc.train.utils.train_utils import HParams, extract_model, load_checkpoint, save_checkpoint
from rvc.train.visualization import mel_spectrogram_similarity, plot_spectrogram_to_numpy

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

global_step = 0


def generate_config(config_save_path, sample_rate, vocoder):
    config_path = os.path.join("rvc", "configs", f"{sample_rate}.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            with open(config_path, "r", encoding="utf-8") as config_file:
                config_data = json.load(config_file)
                config_data["model"]["vocoder"] = vocoder
                json.dump(config_data, f, ensure_ascii=False, indent=2)


def get_hparams():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--total_epoch", type=int, choices=range(1, 10001), default=300)
    parser.add_argument("--save_every_epoch", type=int, choices=range(1, 101), default=25)
    parser.add_argument("--batch_size", type=int, choices=range(1, 129), default=8)
    parser.add_argument("--sample_rate", type=int, choices=[32000, 40000, 48000], default=48000)
    parser.add_argument("--vocoder", type=str, choices=["HiFi-GAN", "MRF HiFi-GAN", "RefineGAN"], default="HiFi-GAN")
    parser.add_argument("--optimizer", type=str, choices=["AdamW", "AdaBelief", "AdaBeliefV2"], default="AdamW")
    parser.add_argument("--pretrain_g", type=str, default=None)
    parser.add_argument("--pretrain_d", type=str, default=None)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--save_to_zip", type=lambda x: bool(strtobool(x)), choices=[True, False], default=False)
    args = parser.parse_args()

    experiment_dir = os.path.join(args.experiment_dir, args.model_name)
    config_save_path = os.path.join(experiment_dir, "data", "config.json")

    # Генерация файла конфигурации
    if not os.path.exists(config_save_path):
        generate_config(config_save_path, args.sample_rate, args.vocoder)

    # Загрузка файла конфигурации
    with open(config_save_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    hparams = HParams(**config)
    hparams.model_dir = experiment_dir
    hparams.model_name = args.model_name
    hparams.total_epoch = args.total_epoch
    hparams.save_every_epoch = args.save_every_epoch
    hparams.batch_size = args.batch_size
    hparams.optimizer = args.optimizer
    hparams.pretrain_g = args.pretrain_g
    hparams.pretrain_d = args.pretrain_d
    hparams.gpus = args.gpus
    hparams.save_to_zip = args.save_to_zip
    hparams.data.training_files = f"{experiment_dir}/data/filelist.txt"

    print(" \n\nПАРАМЕТРЫ ОБУЧЕНИЯ ")
    print("="*70)
    print(f"{'Папка сохранения:':<30} {hparams.model_dir}")
    print(f"{'Имя модели:':<30} {hparams.model_name}")
    print(f"{'Эпох обучения:':<30} {hparams.total_epoch}")
    print(f"{'Сохранение каждые:':<30} {hparams.save_every_epoch} эпох")
    print(f"{'Размер батча:':<30} {hparams.batch_size}")
    print(f"{'Частота дискретизации:':<30} {hparams.data.sample_rate} Hz")
    print(f"{'Вокодер:':<30} {hparams.model.vocoder}")
    print(f"{'Оптимизатор:':<30} {hparams.optimizer}")
    if args.pretrain_g:
        print(f"{'Pretrain G:':<30} {hparams.pretrain_g}")
    if args.pretrain_d:
        print(f"{'Pretrain D:':<30} {hparams.pretrain_d}")
    print(f"{'GPU:':<30} {hparams.gpus}")
    print(f"{'Сохранение в ZIP:':<30} {'Да' if hparams.save_to_zip else 'Нет'}")
    print("="*70 + "\n")
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

    device = torch.device(
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() else 
        "cpu"
    )
    gpus = [int(item) for item in hps.gpus.split("-")] if device.type == "cuda" else [0]
    n_gpus = len(gpus)
    if device.type == "cpu":
        print("Обучение с использованием процессора займёт много времени.", flush=True)

    children = []
    for rank, device_id in enumerate(gpus):
        subproc = mp.Process(
            target=run,
            args=(hps, rank, n_gpus, device, device_id),
        )
        children.append(subproc)
        subproc.start()

    for subproc in children:
        subproc.join()

    sys.exit(0)


def run(hps, rank, n_gpus, device, device_id):
    global global_step

    try:
        metrics_ema = {} if rank == 0 else None
        best_metrics = {"metrics/mel_sim": {"value": -float('inf'), "epoch": 0}} if rank == 0 else None
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval")) if rank == 0 else None
        fn_mel_loss = MultiScaleMelSpectrogramLoss(sample_rate=hps.data.sample_rate)

        dist.init_process_group(
            backend="gloo" if sys.platform == "win32" or device.type != "cuda" else "nccl",
            init_method="env://",
            world_size=n_gpus if device.type == "cuda" else 1,
            rank=rank if device.type == "cuda" else 0,
        )

        torch.manual_seed(hps.train.seed)
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)

        collate_fn = TextAudioCollateMultiNSFsid()
        train_dataset = TextAudioLoaderMultiNSFsid(hps.data)
        train_sampler = DistributedBucketSampler(
            train_dataset,
            hps.batch_size * n_gpus,
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

        if device.type == "cuda":
            net_g = net_g.cuda(device_id)
            net_d = net_d.cuda(device_id)
        else:
            net_g = net_g.to(device)
            net_d = net_d.to(device)

        if hps.optimizer == "AdaBelief":
            from rvc.train.utils.optimizers.AdaBelief import AdaBelief
            optim_g = AdaBelief(net_g.parameters(), lr=hps.train.learning_rate, betas=hps.train.betas, eps=1e-8)
            optim_d = AdaBelief(net_d.parameters(), lr=hps.train.learning_rate, betas=hps.train.betas, eps=1e-8)
        elif hps.optimizer == "AdaBeliefV2":
            from rvc.train.utils.optimizers.AdaBeliefV2 import AdaBeliefV2, get_inverse_sqrt_scheduler
            optim_g = AdaBeliefV2(net_g.parameters(), lr=hps.train.learning_rate, betas=(0.9, 0.999), eps=1e-8, amsgrad=True)
            optim_d = AdaBeliefV2(net_d.parameters(), lr=hps.train.learning_rate, betas=(0.9, 0.999), eps=1e-8, amsgrad=True)
        else:
            optim_g = torch.optim.AdamW(net_g.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
            optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

        if n_gpus > 1 and device.type == "cuda":
            net_g = DDP(net_g, device_ids=[device_id])
            net_d = DDP(net_d, device_ids=[device_id])

        # Загрузка чекпоинта
        epoch_str = None
        checkpoint_path = os.path.join(hps.model_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_path):
            try:
                epoch_str = load_checkpoint(checkpoint_path, net_g, optim_g, net_d, optim_d)
            except Exception as e:
                print(f"Ошибка загрузки checkpoint.pth:\n{e}", flush=True)

        if epoch_str is not None:
            epoch_str += 1
            global_step = (epoch_str - 1) * len(train_loader)

            # Пересчёт EMA и лучших значений из TensorBoard
            if rank == 0:
                try:
                    from tensorboard.backend.event_processing import event_accumulator
                    ea = event_accumulator.EventAccumulator(os.path.join(hps.model_dir, "eval"), size_guidance={'scalars': 0})
                    ea.Reload()

                    if ea.Tags().get('scalars'):
                        print(f"\nСинхронизация метрик из TensorBoard...", flush=True)
                        for tag in ea.Tags()['scalars']:
                            events = ea.Scalars(tag)
                            if not events:
                                continue

                            step_values = {e.step: float(e.value) for e in events if e.step < epoch_str}
                            sorted_steps = sorted(step_values.keys())

                            smoothing, ema_n, ema_d = 0.987, 0.0, 0.0
                            for step in sorted_steps:
                                val = step_values[step]
                                ema_n = ema_n * smoothing + val * (1.0 - smoothing)
                                ema_d = ema_d * smoothing + (1.0 - smoothing)
                                current = ema_n / ema_d
                                if tag == "metrics/mel_sim" and current >= best_metrics["metrics/mel_sim"]["value"]:
                                    best_metrics["metrics/mel_sim"] = {"value": current, "epoch": step}

                            if sorted_steps:
                                metrics_ema[tag] = current

                    curr_mel = metrics_ema.get('metrics/mel_sim', 0.0)
                    best_val = best_metrics['metrics/mel_sim']['value']
                    best_ep = best_metrics['metrics/mel_sim']['epoch']

                    if best_val == -float('inf'):
                        best_val, best_ep = 0.0, 0

                    print(f"Last Mel: {curr_mel:.2f}% | Best Mel: {best_val:.2f}% (на эпохе {best_ep})", flush=True)

                except Exception as e:
                    print(f"Ошибка чтения TensorBoard: {e}", flush=True)
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
        if hps.optimizer == "AdaBelief":
            scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optim_g, T_max=hps.total_epoch, eta_min=1e-6, last_epoch=epoch_str - 2)
            scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optim_d, T_max=hps.total_epoch, eta_min=1e-6, last_epoch=epoch_str - 2)
        elif hps.optimizer == "AdaBeliefV2":
            scheduler_g = get_inverse_sqrt_scheduler(optim_g, warmup_epochs=10, last_epoch=epoch_str - 2)
            scheduler_d = get_inverse_sqrt_scheduler(optim_d, warmup_epochs=10, last_epoch=epoch_str - 2)
        else:
            scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
            scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

        # Проверка: не превышает ли загруженная эпоха целевую
        if epoch_str > hps.total_epoch:
            if rank == 0:
                print(f"\n⚠️  Загруженный чекпоинт (эпоха {epoch_str - 1}) уже превышает указанное количество эпох ({hps.total_epoch}).", flush=True)
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
                train_loader,
                writer_eval,
                fn_mel_loss,
                device,
                device_id,
                metrics_ema,
                best_metrics,
                epoch_recorder,
            )
            scheduler_g.step()
            scheduler_d.step()
    finally:
        # Уничтожение группы процессов для корректного закрытия программы
        if dist.is_initialized():
            dist.destroy_process_group()


def train_and_evaluate(hps, rank, epoch, nets, optims, train_loader, writer_eval, fn_mel_loss, device, device_id, metrics_ema=None, best_metrics=None, epoch_recorder=None):
    global global_step

    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader.batch_sampler.set_epoch(epoch)

    net_g.train()
    net_d.train()

    loss_disc = loss_gen = loss_fm = loss_mel = loss_kl = loss_gen_all = 0
    grad_norm_d = grad_norm_g = 0

    def smooth(key, value, smoothing=0.987):
        """Сглаживание метрики с EMA"""
        if metrics_ema is None:
            return value

        v = float(value)
        metrics_ema[key] = v if key not in metrics_ema else metrics_ema[key] * smoothing + v * (1.0 - smoothing)
        return metrics_ema[key]
    
    for _, info in enumerate(train_loader):
        if device.type == "cuda":
            info = [tensor.cuda(device_id, non_blocking=True) for tensor in info]
        else:
            info = [tensor.to(device) for tensor in info]

        phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, _, sid = info
        model_output = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
        y_hat, ids_slice, _, z_mask, (_, z_p, m_p, logs_p, _, logs_q) = model_output
        wave = slice_segments(wave, ids_slice * hps.data.hop_length, hps.train.segment_size, dim=3)

        # Discriminator loss
        for _ in range(1):
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            loss_disc = discriminator_loss(y_d_hat_r, y_d_hat_g)
            optim_d.zero_grad()
            loss_disc.backward()
            grad_norm_d = grad_norm(net_d.parameters())
            optim_d.step()

        # Generator loss
        for _ in range(1):
            _, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            loss_mel = fn_mel_loss(wave, y_hat) * hps.train.c_mel / 3.0
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
            optim_g.zero_grad()
            loss_gen_all.backward()
            grad_norm_g = grad_norm(net_g.parameters())
            optim_g.step()

        global_step += 1

    if rank == 0 and epoch % hps.train.log_interval == 0:
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
            "grad/norm_d": grad_norm_d,
            "grad/norm_g": grad_norm_g,
            "loss/avg/d": loss_disc,
            "loss/avg/g": loss_gen,
            "loss/g/fm": loss_fm,
            "loss/g/mel": loss_mel,
            "loss/g/kl": loss_kl,
            "loss/g/total": loss_gen_all,
            "metrics/mel_sim": mel_similarity,
            "Learning Rate/G": optim_g.param_groups[0]['lr'],
            "Learning Rate/D": optim_d.param_groups[0]['lr'],
        }
        image_dict = {
            "mel/slice/real": plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "mel/slice/fake": plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
        }
        for k, v in scalar_dict.items():
            writer_eval.add_scalar(k, v, epoch)
        for k, v in image_dict.items():
            writer_eval.add_image(k, v, epoch, dataformats="HWC")

        # Применяем сглаживание
        smoothed_dict = {k: smooth(k, v) for k, v in scalar_dict.items()}

        # Обновление лучших значений (сглаженных)
        if best_metrics is not None:
            current_mel = smoothed_dict.get("metrics/mel_sim", 0.0)
            if current_mel >= best_metrics["metrics/mel_sim"]["value"]:
                best_metrics["metrics/mel_sim"] = {"value": current_mel, "epoch": epoch}

    # Вывод в консоль
    if rank == 0:
        mel_sim_display = metrics_ema.get("metrics/mel_sim", 0.0) if metrics_ema else 0.0
        
        best_val = best_metrics['metrics/mel_sim']['value']
        best_ep = best_metrics['metrics/mel_sim']['epoch']

        print(
            f"{epoch_recorder.record()}: {hps.model_name} ▸ "
            f"Эпоха {epoch}/{hps.total_epoch} (Шаг {global_step}) ││ "
            f"Mel: {mel_sim_display:.2f}% ▸ Рекорд: {best_val:.2f}% (Эпоха {best_ep})",
            flush=True,
        )

    # Сохранение моделей
    if rank == 0:
        is_final_epoch = epoch >= hps.total_epoch
        should_save_checkpoint = (epoch % hps.save_every_epoch == 0) or is_final_epoch

        if should_save_checkpoint:
            # Сохранение чекпоинта
            checkpoint_path = os.path.join(hps.model_dir, "checkpoint.pth")
            save_checkpoint(net_g, optim_g, net_d, optim_d, hps.train.learning_rate, epoch, checkpoint_path)

            # Сохранение промежуточной модели
            weights_dir = os.path.join(hps.model_dir, "weights")
            os.makedirs(weights_dir, exist_ok=True)

            checkpoint_state = net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict()
            intermediate_path = os.path.join(weights_dir, f"{hps.model_name}_e{epoch}_s{global_step}.pth")
            print(extract_model(hps, checkpoint_state, epoch, global_step, intermediate_path), flush=True)

            # Финальная эпоха
            if is_final_epoch:
                # Сохранение last модели
                last_path = os.path.join(hps.model_dir, f"{hps.model_name}_e{epoch}_s{global_step}_last.pth")
                print(extract_model(hps, checkpoint_state, epoch, global_step, last_path), flush=True)

                # Архивирование
                if hps.save_to_zip:
                    import zipfile

                    zip_filename = os.path.join(hps.model_dir, f"{hps.model_name}.zip")
                    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
                        # Добавляем last модель
                        if os.path.exists(last_path):
                            zipf.write(last_path, os.path.basename(last_path))
                        # Добавляем index
                        index_path = os.path.join(hps.model_dir, f"{hps.model_name}.index")
                        if os.path.exists(index_path):
                            zipf.write(index_path, os.path.basename(index_path))
                    print(f"Файлы модели заархивированы в '{zip_filename}'", flush=True)

                print("\nОбучение успешно завершено!", flush=True)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
