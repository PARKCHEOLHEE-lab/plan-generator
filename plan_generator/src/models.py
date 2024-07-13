import os
import sys

if os.path.abspath(os.path.join(__file__, "../../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

import torch

from torch import nn
from typing import List


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, steps: List[int]):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.steps = steps

        self.encoders = nn.ModuleList([])
        self.encoders += [self.encode(self.in_channels, self.steps[0])]
        self.encoders += [self.encode(self.steps[i], self.steps[i + 1]) for i in range(len(steps) - 1)]

        self.upconvs = nn.ModuleList([])
        self.upconvs += [
            nn.ConvTranspose2d(steps[i], steps[i - 1], kernel_size=2, stride=2) for i in range(len(steps) - 1, 0, -1)
        ]

        self.decoders = nn.ModuleList([])
        self.decoders += [
            nn.Conv2d(steps[i], steps[i - 1], kernel_size=3, padding=1) for i in range(len(steps) - 1, 0, -1)
        ]

        self.decoders += [nn.Conv2d(steps[0], out_channels, kernel_size=1)]

        self.to("cuda")

    def encode(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = []
        for eci, encoder in enumerate(self.encoders):
            x = encoder(x)
            if eci != 0:
                x = nn.functional.max_pool2d(x, kernel_size=2)

            encoded.append(x)

        for uci, upconv in enumerate(self.upconvs):
            x = torch.cat([upconv(x), encoded[len(encoded) - uci - 2]], dim=1)
            x = self.decoders[uci](x)

        x = self.decoders[-1](x)

        return x


class FCEncoder(nn.Module):
    def __init__(self, in_features: int, first_out_features: int, repeat: int):
        super().__init__()

        self.in_features = in_features
        self.first_out_features = first_out_features
        self.repeat = repeat

        modules = []
        modules += [nn.Linear(in_features, first_out_features), nn.ReLU(True)]

        temp_out_features = first_out_features
        for _ in range(repeat // 2):
            if temp_out_features // 2 == 0:
                break

            modules += [nn.Linear(temp_out_features, temp_out_features // 2), nn.ReLU(True)]
            temp_out_features //= 2

        for _ in range(repeat // 2):
            modules += [nn.Linear(temp_out_features, temp_out_features * 2), nn.ReLU(True)]
            temp_out_features *= 2

        modules += [nn.Linear(temp_out_features, in_features), nn.ReLU(True)]

        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        return self.fc(x)


class WallGenerator(UNet):
    def __init__(self, in_channels: int, out_channels: int, steps: List[int], size: int, repeat: int):
        super().__init__(in_channels, out_channels, steps)

        self.size = size
        self.repeat = repeat
        self.fc = FCEncoder(self.size * self.size, self.size * 2, self.repeat)

        self.to("cuda")

    def forward(self, floor):
        encoded = self.fc(floor.reshape(-1, self.size * self.size)).reshape(1, 1, self.size, self.size)
        decoded = super().forward(encoded)

        return torch.sigmoid(decoded)


class RoomAllocator(UNet):
    def __init__(self, in_channels: int, out_channels: int, steps: List[int], size: int, repeat: int):
        super().__init__(in_channels, out_channels, steps)

        self.size = size
        self.repeat = repeat
        self.fc = FCEncoder(self.size * self.size, self.size * 2, self.repeat)

        self.to("cuda")

    def forward(self, walls):
        encoded = self.fc(walls.reshape(-1, self.size * self.size)).reshape(1, 1, self.size, self.size)
        decoded = super().forward(encoded)

        return decoded


class DiceLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()

        self.epsilon = epsilon

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, softmax: bool = True) -> torch.Tensor:
        y_one_hot = torch.zeros_like(y_hat).scatter(1, y.unsqueeze(1), 1)

        if softmax:
            y_hat = nn.functional.softmax(y_hat, dim=1)

        batch_size = y.shape[0]

        y_flattened = y_one_hot.reshape(batch_size, -1)
        y_hat_flattened = y_hat.reshape(batch_size, -1)

        intersection = (y_flattened * y_hat_flattened).sum(dim=1)
        union = y_flattened.sum(dim=1) + y_hat_flattened.sum(dim=1)

        dice = (2 * intersection) / (union + self.epsilon)
        loss = 1 - dice

        return loss.mean()
