import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel
from scipy.signal import medfilt

N_MELS = 128
N_CLASS = 360


# ==================== Общие блоки ====================

class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super().__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1)) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv(x)
        return out + self.shortcut(x) if self.shortcut else out + x


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super().__init__()
        self.conv = nn.ModuleList([ConvBlockRes(in_channels if i == 0 else out_channels, out_channels, momentum) for i in range(n_blocks)])
        self.pool = nn.AvgPool2d(kernel_size=kernel_size) if kernel_size else None

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return (x, self.pool(x)) if self.pool else x


class Encoder(nn.Module):
    def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels=16, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        for _ in range(n_encoders):
            self.layers.append(ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum))
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x):
        concat_tensors = []
        x = self.bn(x)
        for layer in self.layers:
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum))
        for _ in range(n_inters - 1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, (3, 3), stride, (1, 1), out_padding, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList([ConvBlockRes(out_channels * 2 if i == 0 else out_channels, out_channels, momentum) for i in range(n_blocks)])

    def forward(self, x, concat_tensor):
        x = torch.cat((self.conv1(x), concat_tensor), dim=1)
        for conv in self.conv2:
            x = conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_decoders):
            out_channels = in_channels // 2
            self.layers.append(ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum))
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.encoder = Encoder(in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks)
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        return self.decoder(self.intermediate(x), concat_tensors)


class MelSpectrogram(nn.Module):
    def __init__(self, n_mel_channels, sample_rate, win_length, hop_length, n_fft=None, mel_fmin=0, mel_fmax=None, clamp=1e-5):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax, htk=True)
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.clamp = clamp

    def forward(self, audio, keyshift=0, speed=1, center=True):
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))

        keyshift_key = f"{keyshift}_{audio.device}"
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(audio.device)

        fft = torch.stft(
            audio,
            n_fft=n_fft_new,
            hop_length=hop_length_new,
            win_length=win_length_new,
            window=self.hann_window[keyshift_key],
            center=center,
            return_complex=True,
        )
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))

        if keyshift != 0:
            size = self.n_fft // 2 + 1
            if magnitude.size(1) < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size - magnitude.size(1)))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new

        mel_output = torch.matmul(self.mel_basis, magnitude)
        return torch.log(torch.clamp(mel_output, min=self.clamp))


# ==================== HPA-специфичные блоки ====================

def autopad(k, p=None):
    return p if p is not None else (k // 2 if isinstance(k, int) else [x // 2 for x in k])


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DSConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, act=True):
        super().__init__()
        self.dwconv = nn.Conv2d(c1, c1, k, s, autopad(k, p), groups=c1, bias=False)
        self.pwconv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.pwconv(self.dwconv(x))))


class DS_Bottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, shortcut=True):
        super().__init__()
        self.dsconv1 = DSConv(c1, c1, k=3, s=1)
        self.dsconv2 = DSConv(c1, c2, k=k, s=1)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        return x + self.dsconv2(self.dsconv1(x)) if self.shortcut else self.dsconv2(self.dsconv1(x))


class DS_C3k(nn.Module):
    def __init__(self, c1, c2, n=1, k=3, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[DS_Bottleneck(c_, c_, k=k, shortcut=True) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class DS_C3k2(nn.Module):
    def __init__(self, c1, c2, n=1, k=3, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.m = DS_C3k(c_, c_, n=n, k=k, e=1.0)
        self.cv2 = Conv(c_, c2, 1, 1)

    def forward(self, x):
        return self.cv2(self.m(self.cv1(x)))


class AdaptiveHyperedgeGeneration(nn.Module):
    def __init__(self, in_channels, num_hyperedges, num_heads):
        super().__init__()
        self.num_hyperedges = num_hyperedges
        self.num_heads = num_heads
        self.head_dim = max(1, in_channels // num_heads)
        self.global_proto = nn.Parameter(torch.randn(num_hyperedges, in_channels))
        self.context_mapper = nn.Linear(2 * in_channels, num_hyperedges * in_channels, bias=False)
        self.query_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, N, C = x.shape
        x_t = x.permute(0, 2, 1)
        context = torch.cat((F.adaptive_avg_pool1d(x_t, 1).squeeze(-1), F.adaptive_max_pool1d(x_t, 1).squeeze(-1)), dim=1)
        P = self.global_proto.unsqueeze(0) + self.context_mapper(context).view(B, self.num_hyperedges, C)
        q = self.query_proj(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = P.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        attn = ((q @ k) * self.scale).mean(dim=1).permute(0, 2, 1)
        return F.softmax(attn, dim=-1)


class HypergraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.W_e = nn.Linear(in_channels, in_channels, bias=False)
        self.W_v = nn.Linear(in_channels, out_channels, bias=False)
        self.act = nn.SiLU()

    def forward(self, x, A):
        return x + self.act(self.W_v(A.transpose(1, 2).bmm(self.act(self.W_e(A.bmm(x))))))


class AdaptiveHypergraphComputation(nn.Module):
    def __init__(self, in_channels, out_channels, num_hyperedges, num_heads):
        super().__init__()
        self.adaptive_hyperedge_gen = AdaptiveHyperedgeGeneration(in_channels, num_hyperedges, num_heads)
        self.hypergraph_conv = HypergraphConvolution(in_channels, out_channels)

    def forward(self, x):
        B, _, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)
        return self.hypergraph_conv(x_flat, self.adaptive_hyperedge_gen(x_flat)).permute(0, 2, 1).view(B, -1, H, W)


class C3AH(nn.Module):
    def __init__(self, c1, c2, num_hyperedges, num_heads, e=0.5):
        super().__init__()
        c_ = int(c1 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.ahc = AdaptiveHypergraphComputation(c_, c_, num_hyperedges, num_heads)
        self.cv3 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        return self.cv3(torch.cat((self.ahc(self.cv2(x)), self.cv1(x)), dim=1))


class HyperACE(nn.Module):
    def __init__(self, in_channels, out_channels, num_hyperedges=16, num_heads=8, k=2, l=1, c_h=0.5, c_l=0.25):
        super().__init__()
        c2, c3, c4, c5 = in_channels
        c_mid = c4
        self.fuse_conv = Conv(c2 + c3 + c4 + c5, c_mid, 1, 1)
        self.c_h = int(c_mid * c_h)
        self.c_l = int(c_mid * c_l)
        self.c_s = c_mid - self.c_h - self.c_l
        self.high_order_branch = nn.ModuleList([C3AH(self.c_h, self.c_h, num_hyperedges, num_heads, e=1.0) for _ in range(k)])
        self.high_order_fuse = Conv(self.c_h * k, self.c_h, 1, 1)
        self.low_order_branch = nn.Sequential(*[DS_C3k(self.c_l, self.c_l, n=1, k=3, e=1.0) for _ in range(l)])
        self.final_fuse = Conv(self.c_h + self.c_l + self.c_s, out_channels, 1, 1)

    def forward(self, x):
        B2, B3, B4, B5 = x
        _, _, H4, W4 = B4.shape
        fused = torch.cat((
            F.interpolate(B2, size=(H4, W4), mode='bilinear', align_corners=False),
            F.interpolate(B3, size=(H4, W4), mode='bilinear', align_corners=False),
            B4,
            F.interpolate(B5, size=(H4, W4), mode='bilinear', align_corners=False)
        ), dim=1)
        x_h, x_l, x_s = self.fuse_conv(fused).split([self.c_h, self.c_l, self.c_s], dim=1)
        h_out = self.high_order_fuse(torch.cat([m(x_h) for m in self.high_order_branch], dim=1))
        return self.final_fuse(torch.cat((h_out, self.low_order_branch(x_l), x_s), dim=1))


class GatedFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, f_in, h):
        return f_in + self.gamma * h


class YOLO13Encoder(nn.Module):
    def __init__(self, in_channels, base_channels=32):
        super().__init__()
        self.stem = DSConv(in_channels, base_channels, k=3, s=1)
        self.p2 = nn.Sequential(DSConv(base_channels, base_channels * 2, k=3, s=(2, 2)), DS_C3k2(base_channels * 2, base_channels * 2, n=1))
        self.p3 = nn.Sequential(DSConv(base_channels * 2, base_channels * 4, k=3, s=(2, 2)), DS_C3k2(base_channels * 4, base_channels * 4, n=2))
        self.p4 = nn.Sequential(DSConv(base_channels * 4, base_channels * 8, k=3, s=(2, 2)), DS_C3k2(base_channels * 8, base_channels * 8, n=2))
        self.p5 = nn.Sequential(DSConv(base_channels * 8, base_channels * 16, k=3, s=(2, 2)), DS_C3k2(base_channels * 16, base_channels * 16, n=1))
        self.out_channels = [base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]

    def forward(self, x):
        x = self.stem(x)
        p2 = self.p2(x)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return [p2, p3, p4, p5]


class YOLO13FullPADDecoder(nn.Module):
    def __init__(self, encoder_channels, hyperace_out_c, out_channels_final):
        super().__init__()
        c_p2, c_p3, c_p4, c_p5 = encoder_channels
        c_d5, c_d4, c_d3, c_d2 = c_p5, c_p4, c_p3, c_p2

        self.h_to_d5 = Conv(hyperace_out_c, c_d5, 1, 1)
        self.h_to_d4 = Conv(hyperace_out_c, c_d4, 1, 1)
        self.h_to_d3 = Conv(hyperace_out_c, c_d3, 1, 1)
        self.h_to_d2 = Conv(hyperace_out_c, c_d2, 1, 1)

        self.fusion_d5 = GatedFusion(c_d5)
        self.fusion_d4 = GatedFusion(c_d4)
        self.fusion_d3 = GatedFusion(c_d3)
        self.fusion_d2 = GatedFusion(c_d2)

        self.skip_p5 = Conv(c_p5, c_d5, 1, 1)
        self.skip_p4 = Conv(c_p4, c_d4, 1, 1)
        self.skip_p3 = Conv(c_p3, c_d3, 1, 1)
        self.skip_p2 = Conv(c_p2, c_d2, 1, 1)

        self.up_d5 = DS_C3k2(c_d5, c_d4, n=1)
        self.up_d4 = DS_C3k2(c_d4, c_d3, n=1)
        self.up_d3 = DS_C3k2(c_d3, c_d2, n=1)

        self.final_d2 = DS_C3k2(c_d2, c_d2, n=1)
        self.final_conv = Conv(c_d2, out_channels_final, 1, 1)

    def _interpolate_h(self, h_ace, target_shape, h_conv):
        return h_conv(F.interpolate(h_ace, size=target_shape, mode='bilinear', align_corners=False))

    def forward(self, enc_feats, h_ace):
        p2, p3, p4, p5 = enc_feats

        d5 = self.skip_p5(p5)
        d5_fused = self.fusion_d5(d5, self._interpolate_h(h_ace, d5.shape[2:], self.h_to_d5))
        d4 = self.up_d5(F.interpolate(d5_fused, size=p4.shape[2:], mode='bilinear', align_corners=False)) + self.skip_p4(p4)

        d4_fused = self.fusion_d4(d4, self._interpolate_h(h_ace, d4.shape[2:], self.h_to_d4))
        d3 = self.up_d4(F.interpolate(d4_fused, size=p3.shape[2:], mode='bilinear', align_corners=False)) + self.skip_p3(p3)

        d3_fused = self.fusion_d3(d3, self._interpolate_h(h_ace, d3.shape[2:], self.h_to_d3))
        d2 = self.up_d3(F.interpolate(d3_fused, size=p2.shape[2:], mode='bilinear', align_corners=False)) + self.skip_p2(p2)

        d2_fused = self.fusion_d2(d2, self._interpolate_h(h_ace, d2.shape[2:], self.h_to_d2))
        return self.final_conv(self.final_d2(d2_fused))


class HPADeepUnet(nn.Module):
    def __init__(self, in_channels=1, en_out_channels=16, base_channels=64, hyperace_k=2, hyperace_l=1, num_hyperedges=16, num_heads=8):
        super().__init__()
        self.encoder = YOLO13Encoder(in_channels, base_channels)
        enc_ch = self.encoder.out_channels
        self.hyperace = HyperACE(enc_ch, enc_ch[-1], num_hyperedges, num_heads, hyperace_k, hyperace_l)
        self.decoder = YOLO13FullPADDecoder(enc_ch, enc_ch[-1], en_out_channels)

    def forward(self, x):
        features = self.encoder(x)
        h_ace = self.hyperace(features)
        decoded = self.decoder(features, h_ace)
        return F.interpolate(decoded, size=x.shape[2:], mode='bilinear', align_corners=False)


# ==================== E2E модели ====================

class E2E(nn.Module):
    def __init__(self, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16, hpa=False):
        super().__init__()
        if hpa:
            self.unet = HPADeepUnet(in_channels, en_out_channels, 64, 2, 1, 16, 4)
        else:
            self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)

        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(BiGRU(3 * 128, 256, n_gru), nn.Linear(512, N_CLASS), nn.Dropout(0.25), nn.Sigmoid())
        else:
            self.fc = nn.Sequential(nn.Linear(3 * N_MELS, N_CLASS), nn.Dropout(0.25), nn.Sigmoid())

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.unet(mel)
        x = self.cnn(x).transpose(1, 2).flatten(-2)
        return self.fc(x)


# ==================== Основные классы ====================

class RMVPE:
    def __init__(self, model_path, device=None, hpa=False):
        self.device = device
        self.hpa = hpa

        self.model = E2E(4, 1, (2, 2), hpa=hpa)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self.model.eval().to(device)

        self.mel_extractor = MelSpectrogram(N_MELS, 16000, 1024, 160, None, 30, 8000).to(device)
        cents_mapping = 20 * np.arange(N_CLASS) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))

    def mel2hidden(self, mel, chunk_size=32000):
        with torch.no_grad():
            n_frames = mel.shape[-1]
            mel = F.pad(mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode="reflect")

            if self.hpa and mel.shape[-1] > chunk_size:
                output_chunks = []
                for start in range(0, mel.shape[-1], chunk_size):
                    mel_chunk = mel[..., start:min(start + chunk_size, mel.shape[-1])]
                    output_chunks.append(self.model(mel_chunk))
                hidden = torch.cat(output_chunks, dim=1)
            else:
                hidden = self.model(mel)

            return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03):
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        return f0

    def infer_from_audio(self, audio, thred=0.03):
        audio = torch.from_numpy(audio).float().to(self.device).unsqueeze(0)
        mel = self.mel_extractor(audio, center=True)
        hidden = self.mel2hidden(mel).squeeze(0).cpu().numpy()
        return self.decode(hidden, thred=thred)

    def infer_from_audio_medfilt(self, audio, thred=0.03, f0_min=50, f0_max=1100, window_size=3):
        """RMVPE+ с медианной фильтрацией"""
        f0 = self.infer_from_audio(audio, thred)
        f0[(f0 < f0_min) | (f0 > f0_max)] = 0
        return medfilt(f0, kernel_size=window_size)

    def to_local_average_cents(self, salience, thred=0.05):
        center = np.argmax(salience, axis=1)
        salience = np.pad(salience, ((0, 0), (4, 4)))
        center += 4
        todo_salience = []
        todo_cents_mapping = []
        starts = center - 4
        ends = center + 5
        for idx in range(salience.shape[0]):
            todo_salience.append(salience[:, starts[idx] : ends[idx]][idx])
            todo_cents_mapping.append(self.cents_mapping[starts[idx] : ends[idx]])
        todo_salience = np.array(todo_salience)
        todo_cents_mapping = np.array(todo_cents_mapping)
        product_sum = np.sum(todo_salience * todo_cents_mapping, 1)
        weight_sum = np.sum(todo_salience, 1)
        devided = product_sum / weight_sum
        maxx = np.max(salience, axis=1)
        devided[maxx <= thred] = 0
        return devided
