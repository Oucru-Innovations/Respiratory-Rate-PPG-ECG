"""Models for respiratory-spectrogram regression.

``SpectroCNN``: small CNN that pools along time only, so the frequency axis --
which *is* the answer -- keeps full resolution until the head. Inputs are a few
channels of ~60x45 log-power spectrograms, so an ImageNet backbone (and
224x224 resizing) is unnecessary; ~0.4 M parameters.

``SpectroTransformer``: transformer encoder over spectrogram frames or patches
with attention-pooling readout; ~0.8 M parameters at the defaults.
"""
import math

import torch
from torch import nn


def _block(cin, cout, pool_t):
    layers = [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU(),
              nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU()]
    if pool_t:
        layers.append(nn.MaxPool2d(kernel_size=(1, 2)))
    return nn.Sequential(*layers)


class SpectroCNN(nn.Module):
    def __init__(self, in_channels, n_freq, n_out, width=32, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            _block(in_channels, width, pool_t=True),
            _block(width, 2 * width, pool_t=True),
            _block(2 * width, 2 * width, pool_t=False),
        )
        self.reduce = nn.Conv1d(2 * width, 16, kernel_size=1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(16 * n_freq, 256), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(256, n_out))

    def forward(self, x):                  # x: (B, C, F, T)
        h = self.features(x)               # (B, W, F, T')
        h = h.mean(dim=-1)                 # average over time -> (B, W, F)
        return self.head(self.reduce(h))  # (B, n_out)


class SpectroTransformer(nn.Module):
    """Transformer encoder over a respiratory spectrogram.

    Tokenisation (``tokens``):

    ``"frame"`` (default)
        One token per STFT time frame: the whole (channels x frequency) column
        is linearly embedded, and self-attention runs across time. Breathing is
        a frequency ridge that persists over the window while artefacts are
        local in time, so attention can weight the clean frames.
    ``"patch"``
        Non-overlapping ``patch_f x patch_t`` patches as in the Audio Spectrogram
        Transformer (Gong et al. 2021, Interspeech), so attention can also relate
        frequency regions (e.g. sidebands around the cardiac line in the raw STFT).

    Absolute frequency is the quantity being estimated, so the positional
    embedding is factorised into learned frequency and time parts. The
    readout is attention pooling with a single learned query; its weights
    (``forward(..., return_attention=True)``) show which tokens a prediction
    relied on.
    """

    def __init__(self, in_channels, n_freq, n_out, tokens="frame", d_model=128, depth=4,
                 heads=4, dropout=0.1, patch_f=6, patch_t=3, max_time=256):
        super().__init__()
        self.tokens = tokens
        if tokens == "frame":
            self.embed = nn.Linear(in_channels * n_freq, d_model)
            n_fpos = 1
        elif tokens == "patch":
            self.patch_f, self.patch_t = patch_f, patch_t
            self.embed = nn.Conv2d(in_channels, d_model, kernel_size=(patch_f, patch_t),
                                   stride=(patch_f, patch_t))
            n_fpos = -(-n_freq // patch_f)  # ceil
        else:
            raise ValueError(f"tokens must be 'frame' or 'patch', got {tokens!r}")
        self.freq_pos = nn.Parameter(torch.zeros(1, n_fpos, 1, d_model))
        self.time_pos = nn.Parameter(torch.zeros(1, 1, max_time, d_model))
        nn.init.trunc_normal_(self.freq_pos, std=0.02)
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.pool = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_out))

    def _tokens(self, x):                  # x: (B, C, F, T) -> (B, Nf, Nt, D)
        if self.tokens == "frame":
            b, c, f, t = x.shape
            return self.embed(x.permute(0, 3, 1, 2).reshape(b, t, c * f)).unsqueeze(1)
        pad_f = (-x.shape[2]) % self.patch_f
        pad_t = (-x.shape[3]) % self.patch_t
        x = nn.functional.pad(x, (0, pad_t, 0, pad_f))
        return self.embed(x).permute(0, 2, 3, 1)

    def forward(self, x, return_attention=False):
        h = self._tokens(x)
        nf, nt = h.shape[1], h.shape[2]
        h = h + self.freq_pos[:, :nf] + self.time_pos[:, :, :nt]
        h = self.norm(self.encoder(self.drop(h.flatten(1, 2))))   # (B, Nf*Nt, D)
        pooled, attn = self.pool(self.query.expand(len(h), -1, -1), h, h)
        out = self.head(pooled.squeeze(1))
        if return_attention:
            return out, attn.squeeze(1).reshape(len(h), nf, nt)
        return out


class _AttnPool(nn.Module):
    """Softmax-weighted mean over the time axis (dim 1) with a learned score."""

    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h):                       # (B, T, D) -> (B, D), (B, T)
        w = torch.softmax(self.score(h).squeeze(-1), dim=1)
        return (w.unsqueeze(-1) * h).sum(1), w


class SpectroRNN(nn.Module):
    """Small bidirectional LSTM/GRU over spectrogram time frames (each frame's
    channels x frequency column is one input vector), attention-pooled over time."""

    def __init__(self, in_channels, n_freq, n_out, cell="lstm", width=64, depth=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_channels * n_freq, width), nn.LayerNorm(width), nn.GELU())
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(width, width, num_layers=depth, batch_first=True, bidirectional=True,
                       dropout=dropout if depth > 1 else 0.0)
        self.pool = _AttnPool(2 * width)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * width, n_out))

    def forward(self, x):                        # (B, C, F, T)
        b, c, f, t = x.shape
        h, _ = self.rnn(self.proj(x.permute(0, 3, 1, 2).reshape(b, t, c * f)))
        return self.head(self.pool(h)[0])


def _conv2(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU())


class SpectroUNet(nn.Module):
    """Three-level 2-D U-Net that maps the spectrogram to a one-channel
    "breathing-ridge" map of the same size; the map is averaged over time and
    the per-frequency profile is mapped linearly to the RR bins."""

    def __init__(self, in_channels, n_freq, n_out, width=16):
        super().__init__()
        w = width
        self.enc1, self.enc2, self.mid = _conv2(in_channels, w), _conv2(w, 2 * w), _conv2(2 * w, 4 * w)
        self.up2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)
        self.dec2 = _conv2(4 * w, 2 * w)
        self.up1 = nn.ConvTranspose2d(2 * w, w, 2, stride=2)
        self.dec1 = _conv2(2 * w, w)
        self.out = nn.Conv2d(w, 1, 1)
        self.head = nn.Linear(n_freq, n_out)

    def forward(self, x):                        # (B, C, F, T)
        f, t = x.shape[-2:]
        x = nn.functional.pad(x, (0, (-t) % 4, 0, (-f) % 4))
        e1 = self.enc1(x)
        e2 = self.enc2(nn.functional.max_pool2d(e1, 2))
        m = self.mid(nn.functional.max_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([self.up2(m), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        ridge = self.out(d1)[:, 0, :f, :t]       # (B, F, T)
        return self.head(ridge.mean(-1))


class SpectrumMLP(nn.Module):
    """Floor model: the time-averaged spectrum of each channel into a two-layer
    MLP. Whatever the other models gain over this comes from modelling time."""

    def __init__(self, in_channels, n_freq, n_out, width=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_channels * n_freq, width), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(width, n_out))

    def forward(self, x):                        # (B, C, F, T)
        return self.net(x.mean(-1).flatten(1))


# ---------------------------------------------------------------- BreathNet
class ResonatorBank(nn.Module):
    """Bank of complex Gabor (damped-oscillator) filters, one per candidate
    breathing rate, applied to each respiratory pathway.

    Physiology: one central rhythm generator drives the chest at a single
    rate; a filter tuned to that rate resonates. Centre frequencies are fixed
    to the output RR bins (so filter k *is* rate k); each filter's duration
    (hence bandwidth) is learned. Kernels are zero-mean so slow drift does not
    excite the lowest-rate filters.
    """

    def __init__(self, rates_bpm, fs=4.0, kernel_s=32.0, init_sigma_s=8.0, hop=4):
        super().__init__()
        self.fs, self.hop = fs, hop
        self.register_buffer("freq_hz", torch.as_tensor(rates_bpm, dtype=torch.float32) / 60.0)
        n = int(kernel_s * fs) | 1                                  # odd length
        self.register_buffer("t", (torch.arange(n) - n // 2) / fs)
        # at least ~1.5 cycles per filter, at most init_sigma_s
        sigma0 = torch.clamp(1.5 / self.freq_hz, max=init_sigma_s)
        self.log_sigma = nn.Parameter(torch.log(sigma0))

    def kernels(self):
        sigma = self.log_sigma.exp().clamp(min=0.5, max=self.t.max().item() / 2)
        env = torch.exp(-0.5 * (self.t[None] / sigma[:, None]) ** 2)       # (K, L)
        ph = 2 * torch.pi * self.freq_hz[:, None] * self.t[None]
        re, im = env * torch.cos(ph), env * torch.sin(ph)
        re = re - (re.sum(1, keepdim=True) / env.sum(1, keepdim=True)) * env  # zero-mean
        norm = torch.sqrt((re ** 2 + im ** 2).sum(1, keepdim=True))
        return torch.cat([re / norm, im / norm])[:, None]                   # (2K, 1, L)

    def response(self, x):                       # (B, P, N) -> real, imaginary parts, each (B, P, K, T)
        b, p, n = x.shape
        k = self.kernels()
        y = nn.functional.conv1d(x.reshape(b * p, 1, n), k, padding=k.shape[-1] // 2, stride=self.hop)
        nk = k.shape[0] // 2
        return y[:, :nk].reshape(b, p, nk, -1), y[:, nk:].reshape(b, p, nk, -1)

    def forward(self, x):                        # (B, P, N) -> log energy (B, P, K, T)
        re, im = self.response(x)
        return torch.log(re ** 2 + im ** 2 + 1e-6)


class BreathNet(nn.Module):
    """Physiology-inspired RR estimator on the respiratory *waveforms*
    (BW, AM, FM at 4 Hz; feature kind ``modwave``).

    1. Pathways: BW (intrathoracic pressure -> venous return), AM (stroke
       volume) and FM (vagal sinus arrhythmia) are separate evidence streams.
    2. Resonator bank: damped oscillators at every candidate rate
       (:class:`ResonatorBank`); output bin k = breathing rate k.
    3. Harmonic summation: inspiration is shorter than expiration, so breathing
       waveforms carry energy at 2x and 3x the rate; learned weights add it back
       to the fundamental and a half-rate term lets the model veto octave errors.
    4. Rhythm persistence: breathing is steady over a window while coughs and
       movement are brief, so frames are attention-pooled over time.
    5. Learned smart fusion: all pathways share one rate; a reliability gate
       weights each pathway's evidence per window and the weighted evidence is
       summed in the log domain (product of experts), cf. Karlen et al. 2013.

    The output logits are indexed by the same rates as the resonators, so the
    model returns a distribution over 4-60 breaths/min directly.
    """

    def __init__(self, in_channels, n_samples, n_out, fs=4.0, rr_min=4.0, step=1.0, width=32, dropout=0.1):
        super().__init__()
        rates = rr_min + step * torch.arange(n_out)
        self.bank = ResonatorBank(rates.tolist(), fs=fs)
        self.p, self.k = in_channels, n_out
        # harmonic summation: evidence[k] += a2 * E[2k] + a3 * E[3k] + a_half * E[k/2]
        idx = {h: torch.round((rates * h - rr_min) / step).long() for h in (2.0, 3.0, 0.5)}
        for h, name in ((2.0, "i2"), (3.0, "i3"), (0.5, "ih")):
            valid = (idx[h] >= 0) & (idx[h] < n_out)
            self.register_buffer(name, torch.where(valid, idx[h], torch.zeros_like(idx[h])))
            self.register_buffer(name + "_ok", valid.float())
        self.harm = nn.Parameter(torch.tensor([[0.5, 0.25, 0.0]]).repeat(in_channels, 1))  # (P, 3)
        # per-frame evidence refinement (local frequency context) and frame scorer
        self.refine = nn.Sequential(nn.Conv1d(in_channels, in_channels * width // 4, 5, padding=2, groups=in_channels),
                                    nn.GELU(),
                                    nn.Conv1d(in_channels * width // 4, in_channels, 5, padding=2, groups=in_channels))
        self.frame_score = nn.ModuleList(nn.Sequential(nn.Linear(n_out, width), nn.GELU(), nn.Linear(width, 1))
                                         for _ in range(in_channels))
        # reliability gate from evidence shape (peak prominence, spread, entropy) + attention entropy
        self.gate = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1))
        self.rate_bias = nn.Parameter(torch.zeros(n_out))  # prior over rates / residual 1/f background
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def _harmonics(self, e):                     # e: (B, P, K, T)
        def take(idx, ok):
            return e.index_select(2, idx) * ok[None, None, :, None]
        a = self.harm[None, :, :, None, None]    # (1, P, 3, 1, 1)
        return (e + a[:, :, 0] * take(self.i2, self.i2_ok) + a[:, :, 1] * take(self.i3, self.i3_ok)
                + a[:, :, 2] * take(self.ih, self.ih_ok))

    def forward(self, x, return_details=False):  # x: (B, P, N)
        e = self.bank(x)                                          # (B, P, K, T)
        e = (e - e.mean(dim=(2, 3), keepdim=True)) / (e.std(dim=(2, 3), keepdim=True) + 1e-5)
        e = self._harmonics(e)
        b, p, k, t = e.shape
        frames = e.permute(0, 3, 1, 2).reshape(b * t, p, k)       # refine along frequency, per frame
        e = (frames + self.refine(frames)).reshape(b, t, p, k)
        evid, attn = [], []
        for i in range(p):
            w = torch.softmax(self.frame_score[i](e[:, :, i]).squeeze(-1), dim=1)   # (B, T)
            evid.append((w.unsqueeze(-1) * e[:, :, i]).sum(1))                     # (B, K)
            attn.append(w)
        evid, attn = torch.stack(evid, 1), torch.stack(attn, 1)                     # (B, P, K), (B, P, T)
        prob = torch.softmax(evid, dim=-1)
        ent = -(prob * torch.log(prob + 1e-9)).sum(-1)
        att_ent = -(attn * torch.log(attn + 1e-9)).sum(-1)
        feats = torch.stack([evid.max(-1).values - evid.mean(-1), evid.std(-1), ent, att_ent], -1)
        gate = torch.softmax(self.gate(feats).squeeze(-1), dim=1)                  # (B, P)
        logits = self.temperature * (gate.unsqueeze(-1) * evid).sum(1) * p + self.rate_bias
        if return_details:
            return logits, {"pathway_evidence": evid, "pathway_weight": gate, "frame_attention": attn,
                            "bandwidth_s": self.bank.log_sigma.exp().detach()}
        return logits


class BreathNet2(BreathNet):
    """BreathNet plus more respiratory physiology, each piece switchable through
    ``variant`` (comma-separated subset of whiten, coherent, track, hr, conv).
    Notebook 04 showed what BreathNet improvised to reach its accuracy; each
    piece here builds that knowledge in explicitly.

    whiten
        Explicit 1/f background. BreathNet learned to turn one pathway into an
        estimate of the slow vasomotor background and subtract it, losing that
        pathway's evidence. Here a power law is fitted to every frame's
        log-energy spectrum (weighted least squares on log rate, peaks
        down-weighted; no parameters) and removed: the correction the
        signal-processing methods use.
    coherent
        A fourth evidence stream. The three modulations are driven by the same
        breath, so their resonator outputs are phase-locked with lags set by
        physiology (BW and AM follow intrathoracic pressure, FM the vagal
        response). A learned gain, phase and lag per pathway sums the complex
        outputs coherently: the shared rhythm adds up, unrelated fluctuations
        cancel. Nine parameters.
    track
        Rhythm persistence as dynamics instead of attention: a hidden Markov
        model over rate x time whose state moves by a Gaussian step (learned
        width, bpm per frame) from the learned rate prior. The forward-backward
        posterior, averaged over the window, is the output. Replaces the frame
        scorer (91 % of BreathNet's parameters) with one.
    hr
        The heart rate constrains the breathing rate: RR/HR is tight (median
        0.28 in this cohort) and beat-sampled modulations cannot show rates
        above about HR/2. A learned log-normal prior on RR/HR plus a soft
        ceiling at a learned fraction of HR, from the mean of the
        instantaneous-HR channel (feature kind ``modwave_hr``). Four parameters.
    conv
        Instead of attention and gate, a small 2-D convolution over the
        (pathway, rate, time) evidence map decides, averaged over time: tests
        whether the CNN's edge lies in free time-frequency pattern matching.
        Mutually exclusive with ``track``.
    """

    MECHANISMS = ("whiten", "coherent", "track", "hr", "conv")

    def __init__(self, in_channels, n_samples, n_out, variant="whiten,coherent,track,hr", hr_channel=False,
                 fs=4.0, rr_min=4.0, step=1.0, width=32, dropout=0.1):
        self.variant = frozenset(v for v in variant.split(",") if v)
        unknown = self.variant - set(self.MECHANISMS)
        if unknown:
            raise ValueError(f"unknown BreathNet2 mechanism(s) {sorted(unknown)}")
        if {"track", "conv"} <= self.variant:
            raise ValueError("track and conv both replace attention; choose one")
        if "hr" in self.variant and not hr_channel:
            raise ValueError("the hr prior needs the heart-rate channel (feature kind modwave_hr)")
        self.hr_channel = bool(hr_channel)
        pathways = in_channels - self.hr_channel
        super().__init__(pathways + ("coherent" in self.variant), n_samples, n_out, fs, rr_min, step, width, dropout)
        self.pathways = pathways
        rates = rr_min + step * torch.arange(n_out)
        self.register_buffer("rates", rates)
        if "whiten" in self.variant:
            u = torch.log(rates)
            self.register_buffer("log_rate_c", u - u.mean())
        if "coherent" in self.variant:
            self.coh_log_gain = nn.Parameter(torch.zeros(pathways))
            self.coh_phase = nn.Parameter(torch.zeros(pathways))      # radians
            self.coh_lag = nn.Parameter(torch.zeros(pathways))        # seconds
        if "track" in self.variant:
            del self.frame_score
            self.log_trans_sigma = nn.Parameter(torch.log(torch.tensor(1.5)))   # bpm per frame
            self.register_buffer("rate_diff2", (rates[:, None] - rates[None, :]) ** 2)
            self.temperature.data.fill_(0.25)   # 60 frames multiply up: start from a soft posterior
        if "conv" in self.variant:
            del self.frame_score, self.gate
            self.head = nn.Sequential(nn.Conv2d(self.p, 16, 5, padding=2), nn.GELU(),
                                      nn.Conv2d(16, 16, 5, padding=2), nn.GELU(), nn.Conv2d(16, 1, 1))
        if "hr" in self.variant:
            self.hr_ratio_mu = nn.Parameter(torch.log(torch.tensor(0.27)))      # log RR/HR
            self.hr_ratio_log_sd = nn.Parameter(torch.log(torch.tensor(0.5)))
            self.hr_ceiling = nn.Parameter(torch.tensor(0.5))                   # fraction of HR
            self.hr_ceiling_slope = nn.Parameter(torch.tensor(1.0))             # logit per bpm above it

    # ---- mechanisms -------------------------------------------------------
    def _whiten(self, e):
        """Subtract a per-frame, per-pathway power-law fit of the log energy over
        log rate (two weighted least-squares passes, the second down-weighting
        bins that stood above the first fit, i.e. the breathing peak)."""
        u = self.log_rate_c[None, None, :, None]

        def fit(w):
            sw, su, suu = w.sum(2, keepdim=True), (w * u).sum(2, keepdim=True), (w * u * u).sum(2, keepdim=True)
            se, sue = (w * e).sum(2, keepdim=True), (w * u * e).sum(2, keepdim=True)
            slope = (sw * sue - su * se) / (sw * suu - su * su + 1e-6)
            return (se - slope * su) / sw + slope * u

        bg = fit(torch.ones_like(e))
        return e - fit(torch.sigmoid(-(e - bg) / 0.5) + 0.05)

    def _coherent(self, re, im):
        """Log energy of the phase-aligned sum of the pathways' complex responses."""
        theta = self.coh_phase[None, :, None, None] + 2 * torch.pi * (self.freq_hz_rates()[None, None, :, None]
                                                                        * self.coh_lag[None, :, None, None])
        g = self.coh_log_gain.exp()[None, :, None, None]
        c, s = torch.cos(theta), torch.sin(theta)
        sr = (g * (c * re - s * im)).sum(1, keepdim=True)
        si = (g * (s * re + c * im)).sum(1, keepdim=True)
        return torch.log(sr ** 2 + si ** 2 + 1e-6)

    def freq_hz_rates(self):
        return self.rates / 60.0

    def _track(self, s):
        """Forward-backward over rate x time; ``s`` (B, K, T) are per-frame log
        evidences. Returns the log of the window-averaged posterior (B, K)."""
        log_a = -0.5 * self.rate_diff2 / self.log_trans_sigma.exp() ** 2
        log_a = log_a - torch.logsumexp(log_a, 1, keepdim=True)              # row k -> column k'
        n_t = s.shape[-1]
        alpha = [torch.log_softmax(self.rate_bias, 0)[None] + s[..., 0]]
        for t in range(1, n_t):
            alpha.append(torch.logsumexp(alpha[-1][:, :, None] + log_a[None], 1) + s[..., t])
        beta = [torch.zeros_like(alpha[0])]
        for t in range(n_t - 2, -1, -1):
            beta.append(torch.logsumexp(log_a[None] + (s[..., t + 1] + beta[-1])[:, None, :], 2))
        gamma = torch.stack([a + b for a, b in zip(alpha, beta[::-1])], -1)   # (B, K, T)
        gamma = torch.log_softmax(gamma, 1)
        return torch.logsumexp(gamma, -1) - math.log(n_t)

    def _hr_prior(self, hr):
        """Log prior over rates given the heart rate (B,) in bpm."""
        f = self.rates[None]
        z = (torch.log(f) - torch.log(hr[:, None]) - self.hr_ratio_mu) / self.hr_ratio_log_sd.exp()
        ceiling = nn.functional.softplus(self.hr_ceiling_slope) * nn.functional.relu(f - self.hr_ceiling * hr[:, None])
        return -0.5 * z ** 2 - ceiling

    # ---- forward ----------------------------------------------------------
    def forward(self, x, return_details=False):   # x: (B, P [+1 heart-rate channel], N)
        hr = None
        if self.hr_channel:
            hr, x = (x[:, -1].mean(1) * 100.0).clamp(30.0, 220.0), x[:, :-1]
        re, im = self.bank.response(x)
        e = torch.log(re ** 2 + im ** 2 + 1e-6)
        if "coherent" in self.variant:
            e = torch.cat([e, self._coherent(re, im)], 1)
        e = (e - e.mean(dim=(2, 3), keepdim=True)) / (e.std(dim=(2, 3), keepdim=True) + 1e-5)
        if "whiten" in self.variant:
            e = self._whiten(e)
        e = self._harmonics(e)
        b, p, k, t = e.shape
        frames = e.permute(0, 3, 1, 2).reshape(b * t, p, k)
        e = (frames + self.refine(frames)).reshape(b, t, p, k)
        details = {"bandwidth_s": self.bank.log_sigma.exp().detach()}
        if "conv" in self.variant:
            logits = self.temperature * self.head(e.permute(0, 2, 3, 1)).squeeze(1).mean(-1) + self.rate_bias
        else:
            if "track" in self.variant:
                evid = e.mean(1)                                                  # (B, P, K)
                fr = torch.softmax(e, -1)
                att_ent = -(fr * torch.log(fr + 1e-9)).sum(-1).mean(1)            # mean per-frame entropy
            else:
                evid, attn = [], []
                for i in range(p):
                    w = torch.softmax(self.frame_score[i](e[:, :, i]).squeeze(-1), dim=1)
                    evid.append((w.unsqueeze(-1) * e[:, :, i]).sum(1))
                    attn.append(w)
                evid, attn = torch.stack(evid, 1), torch.stack(attn, 1)
                att_ent = -(attn * torch.log(attn + 1e-9)).sum(-1)
                details["frame_attention"] = attn
            prob = torch.softmax(evid, dim=-1)
            ent = -(prob * torch.log(prob + 1e-9)).sum(-1)
            feats = torch.stack([evid.max(-1).values - evid.mean(-1), evid.std(-1), ent, att_ent], -1)
            gate = torch.softmax(self.gate(feats).squeeze(-1), dim=1)              # (B, P)
            details.update(pathway_evidence=evid, pathway_weight=gate)
            if "track" in self.variant:
                s = self.temperature * (gate[:, None, :, None] * e).sum(2) * p    # (B, T, K)
                logits = self._track(s.permute(0, 2, 1))
            else:
                logits = self.temperature * (gate.unsqueeze(-1) * evid).sum(1) * p + self.rate_bias
        if "hr" in self.variant:
            logits = logits + self._hr_prior(hr)
            details["heart_rate"] = hr
        return (logits, details) if return_details else logits


ARCHS = {"cnn": SpectroCNN, "transformer": SpectroTransformer, "lstm": SpectroRNN, "gru": SpectroRNN,
         "unet": SpectroUNet, "mlp": SpectrumMLP, "breathnet": BreathNet, "breathnet2": BreathNet2}
# feature kinds each architecture accepts: spectrograms (C, F, T) or waveforms (P, N)
WAVEFORM_ARCHS = {"breathnet", "breathnet2"}


def build_model(arch="cnn", **kwargs):
    """Construct a model from the ``"model"`` entry of a checkpoint (checkpoints
    written before the transformer existed have no ``arch`` key -> CNN)."""
    if arch in ("lstm", "gru"):
        kwargs.setdefault("cell", arch)
    return ARCHS[arch](**kwargs)


class LabelDistribution(nn.Module):
    """Deep label-distribution learning for an ordinal target (Gao et al. 2017,
    IEEE TIP 26:2825): the network outputs a distribution over RR bins and is
    trained against a Gaussian centred on the reference, so predicting 17 when
    the truth is 18 costs far less than predicting 30. The estimate is the
    expectation of the predicted distribution."""

    def __init__(self, rr_min=4.0, rr_max=60.0, step=1.0, sigma=1.5):
        super().__init__()
        self.register_buffer("bins", torch.arange(rr_min, rr_max + step / 2, step))
        self.sigma = sigma

    @property
    def n_bins(self):
        return len(self.bins)

    def target(self, y):
        y = y.clamp(self.bins[0], self.bins[-1])  # keep the Gaussian inside the bin range
        t = torch.exp(-0.5 * ((self.bins[None] - y[:, None]) / self.sigma) ** 2)
        return t / t.sum(dim=1, keepdim=True)

    def loss(self, logits, y):
        return nn.functional.cross_entropy(logits, self.target(y))

    def decode(self, logits):
        return torch.softmax(logits, dim=1) @ self.bins
