"""DeepJSCC (Bourtsoulatze, Burth Kurka, Gunduz, "Deep joint source-channel coding for
wireless image transmission", IEEE TCCN 5(3), 2019; survey ref. [55]).

Architecture from the paper (arXiv:1809.01733v4, Fig. 2): encoder ``5x5x16/2, 5x5x32/2,
5x5x32/1, 5x5x32/1, 5x5xc/1`` (kernel x kernel x filters / stride), each convolution
followed by PReLU; decoder mirroring it with transposed convolutions, ``5x5x32/1`` x3 and
``5x5x16/2`` with PReLU, then ``5x5x3/2`` with a sigmoid. The paper's /255 normalisation
and denormalisation are the benchmark's [0, 1] convention; its power normalisation
(eq. (1), P = 1) is done by the channel.

Choices the paper leaves open follow the PyTorch implementation
https://github.com/chunbaobao/Deep-JSCC-PyTorch (cited as inspiration; it has no license
and no code is copied): padding 2, output padding 1 for stride-2 transposed convolutions,
one PReLU parameter per layer, Kaiming-normal initialisation (``fan_out``) of the
PReLU layers and Xavier-normal of the last (sigmoid) layer. Its optimiser settings (Adam,
lr 1e-3, weight decay 5e-4, learning rate halved on plateau) are in
``configs/models/deepjscc.yaml``. Downsampling factor 4.
"""

from __future__ import annotations

import torch
from torch import nn

from uwbench.models.base import JSCCModel, latent_channels


def _layer(conv: nn.Module, last: bool = False) -> nn.Sequential:
    if last:
        nn.init.xavier_normal_(conv.weight)
        return nn.Sequential(conv, nn.Sigmoid())
    nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="leaky_relu")
    return nn.Sequential(conv, nn.PReLU())


def _conv(cin: int, cout: int, stride: int) -> nn.Sequential:
    return _layer(nn.Conv2d(cin, cout, 5, stride, 2))


def _deconv(cin: int, cout: int, stride: int, last: bool = False) -> nn.Sequential:
    return _layer(nn.ConvTranspose2d(cin, cout, 5, stride, 2, output_padding=stride - 1), last)


class DeepJSCC(JSCCModel):
    downsample = 4

    def __init__(self, bandwidth_ratio: float = 1 / 6, symbols: str = "complex"):
        super().__init__()
        self.symbols = symbols
        c = latent_channels(bandwidth_ratio, self.downsample, symbols)
        self.encoder = nn.Sequential(_conv(3, 16, 2), _conv(16, 32, 2), _conv(32, 32, 1),
                                     _conv(32, 32, 1), _conv(32, c, 1))
        self.decoder = nn.Sequential(_deconv(c, 32, 1), _deconv(32, 32, 1), _deconv(32, 32, 1),
                                     _deconv(32, 16, 2), _deconv(16, 3, 2, last=True))

    def encode(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
