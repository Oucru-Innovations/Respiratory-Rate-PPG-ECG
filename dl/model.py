"""Small CNN for respiratory-spectrogram regression.

Pooling is applied along time only, so the frequency axis -- which *is* the
answer -- keeps full resolution until the head. Inputs are a few channels of
~60x45 log-power spectrograms, so an ImageNet backbone (and 224x224 resizing)
is unnecessary; this model has ~0.5 M parameters.
"""
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
