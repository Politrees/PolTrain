import datetime
import logging
import os
import sys
import warnings
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

# Настройка уровня логирования для различных библиотек
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.WARNING)
logging.getLogger("h5py").setLevel(logging.WARNING)
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("numexpr").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# Подавление предупреждений
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.getcwd()))
from rvc.lib.algorithm.commons import grad_norm, slice_segments
from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.train.data_utils import DistributedBucketSampler, TextAudioCollateMultiNSFsid, TextAudioLoaderMultiNSFsid
from rvc.train.extract.extract_model import extract_model
from rvc.train.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from rvc.train.mel_processing import MultiScaleMelSpectrogramLoss, mel_spectrogram_torch, spec_to_mel_torch
from rvc.train.utils import get_hparams, get_logger, latest_checkpoint_path, load_checkpoint, save_checkpoint, summarize
from rvc.train.visualization import plot_pitch_to_numpy, plot_spectrogram_to_numpy

hps = get_hparams()
os.environ["USE_LIBUV"] = "0" if os.name == "nt" else "1"
os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
n_gpus = len(hps.gpus.split("-"))

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

global_step = 0


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time = round(elapsed_time, 1)
        elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        return f"Скорость: [{elapsed_time_str}]"


def main():
    n_gpus = torch.cuda.device_count()

    if not torch.cuda.is_available():
        print("NO GPU DETECTED: falling back to CPU - this may take a while")
        n_gpus = 1

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))

    children = []
    logger = get_logger(hps.model_dir)
    for i in range(n_gpus):
        subproc = mp.Process(
            target=run,
            args=(i, n_gpus, hps, logger),
        )
        children.append(subproc)
        subproc.start()

    for i in range(n_gpus):
        children[i].join()


def run(rank, n_gpus, hps, logger: logging.Logger):
    global global_step

    writer_eval = None
    if rank == 0:
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    backend = "gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl"
    dist.init_process_group(backend=backend, init_method="env://", world_size=n_gpus, rank=rank)

    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

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
        num_workers=2,  # 4
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
        use_f0=True,
        sr=hps.sample_rate,
        vocoder=hps.vocoder,
        checkpointing=False,
        randomized=True,
    )
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm, checkpointing=False)

    if torch.cuda.is_available():
        net_g = net_g.cuda(rank)
        net_d = net_d.cuda(rank)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    fn_mel_loss = MultiScaleMelSpectrogramLoss(sample_rate=hps.sample_rate)

    if torch.cuda.is_available():
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])

    try:
        _, _, _, epoch_str = load_checkpoint(latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
        _, _, _, epoch_str = load_checkpoint(latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)

        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0

        if hps.pretrainG not in ("", "None"):
            if rank == 0:
                logger.info(f"Загрузка претрейна {hps.pretrainG}")
            g_model = net_g.module if hasattr(net_g, "module") else net_g
            logger.info(g_model.load_state_dict(torch.load(hps.pretrainG, map_location="cpu", weights_only=True)["model"]))

        if hps.pretrainD not in ("", "None"):
            if rank == 0:
                logger.info(f"Загрузка претрейна {hps.pretrainD}")
            d_model = net_d.module if hasattr(net_d, "module") else net_d
            logger.info(d_model.load_state_dict(torch.load(hps.pretrainD, map_location="cpu", weights_only=True)["model"]))

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    for epoch in range(epoch_str, hps.total_epoch + 1):
        train_and_evaluate(
            hps,
            rank,
            epoch,
            [net_g, net_d],
            [optim_g, optim_d],
            [train_loader, None],
            logger,
            [writer_eval],
            fn_mel_loss
        )
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(hps, rank, epoch, nets, optims, loaders, logger, writers, fn_mel_loss):
    global global_step

    if writers is not None:
        writer = writers[0]

    epoch_recorder = EpochRecorder()

    net_g, net_d = nets
    optim_g, optim_d = optims

    train_loader = loaders[0] if loaders is not None else None
    train_loader.batch_sampler.set_epoch(epoch)

    net_g.train()
    net_d.train()

    data_iterator = enumerate(train_loader)
    for batch_idx, info in data_iterator:
        if torch.cuda.is_available():
            info = [tensor.cuda(rank, non_blocking=True) for tensor in info]

        phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid = info
        model_output = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
        y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = model_output

        wave = slice_segments(wave, ids_slice * hps.data.hop_length, hps.train.segment_size, dim=3)

        # Mel-Spectrogram
        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.sample_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )
        y_mel = slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length, dim=3)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.float().squeeze(1),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.sample_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )

        # Discriminator loss
        for _ in range(1):  # default x1
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
            optim_d.zero_grad()
            loss_disc.backward()
            grad_norm_d = grad_norm(net_d.parameters())
            optim_d.step()

        # Generator loss
        _, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
        loss_mel = fn_mel_loss(wave, y_hat) * hps.train.c_mel / 3.0
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, _ = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        optim_g.zero_grad()
        loss_gen_all.backward()
        grad_norm_g = grad_norm(net_g.parameters())
        optim_g.step()

        # learning rates
        current_lr_d = optim_d.param_groups[0]["lr"]
        current_lr_g = optim_g.param_groups[0]["lr"]

        global_step += 1

    if rank == 0 and epoch % hps.train.log_interval == 0:
        scalar_dict = {
            "grad/norm_d": grad_norm_d,                                                     # Норма градиентов Дискриминатора
            "grad/norm_g": grad_norm_g,                                                     # Норма градиентов Генератора
            "learning_rate/d": current_lr_d,                                                # Скорость обучения Дискриминатора
            "learning_rate/g": current_lr_g,                                                # Скорость обучения Генератора
            "loss/avg/d": loss_disc,                                                        # Потеря Дискриминатора
            "loss/avg/g": loss_gen,                                                         # Потеря Генератора
            "loss/g/fm": loss_fm,                                                           # Потеря на основе совпадения признаков между реальными и сгенерированными данными
            "loss/g/mel": loss_mel,                                                         # Потеря на основе мел-спектрограммы
            "loss/g/kl": loss_kl,                                                           # Потеря на основе расхождения распределений в модели
            "loss/g/total": loss_gen_all,                                                   # Общая потеря Генератора
            "metrics/mse_wave": F.mse_loss(y_hat, wave),                                    # Среднеквадратичная ошибка между реальными и сгенерированными аудиосигналами
            "metrics/mse_pitch": F.mse_loss(pitchf, pitch),                                 # Среднеквадратичная ошибка между реальными и сгенерированными интонациями
        }
        image_dict = {
            "mel/slice/real": plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),       # Мел-спектрограмма реальных данных
            "mel/slice/fake": plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),   # Мел-спектрограмма сгенерированных данных
            "pitch/real": plot_pitch_to_numpy(pitch[0].data.cpu().numpy()),                 # Интонация реальных данных
            "pitch/fake": plot_pitch_to_numpy(pitchf[0].data.cpu().numpy()),                # Интонация сгенерированных данных
        }
        summarize(writer=writer, tracking=epoch, scalars=scalar_dict, images=image_dict)

    if rank == 0:
        logger.info(f"====> Эпоха: {epoch}/{hps.total_epoch} | Шаг: {global_step} | {epoch_recorder.record()}")

        save_final = epoch >= hps.total_epoch
        save_checkpoint_cond = (epoch % hps.save_every_epoch == 0) or save_final

        if save_checkpoint_cond:
            # Сохраняем чекпоинты в любом случае (регулярное или финальное сохранение)
            save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_checkpoint.pth"))
            save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_checkpoint.pth"))

            # Определяем тип сохранения модели
            checkpoint = net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict()
            save_model = extract_model(hps, checkpoint, hps.name, epoch, global_step, hps.sample_rate, hps.model_dir, hps.vocoder, final_save=save_final)
            logger.info(save_model)

        if save_final:
            # Действия при завершении обучения
            if hps.save_to_zip == "True":
                zip_filename = os.path.join(hps.model_dir, f"{hps.name}.zip")

                import zipfile
                with zipfile.ZipFile(zip_filename, "w") as zipf:
                    for ext in (".pth", ".index"):
                        file_path = os.path.join(hps.model_dir, f"{hps.name}{ext}")
                        zipf.write(file_path, os.path.basename(file_path))
                logger.info(f"Файлы модели заархивированы в `{zip_filename}`")

            logger.info("Обучение успешно завершено.")
            sleep(1)
            os._exit(2333333)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
