import torch

from torch import nn
from torch.nn import functional


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoding_1 = self.encode(self.in_channels, 32)
        self.encoding_2 = self.encode(32, 64)
        self.encoding_3 = self.encode(64, 128)
        self.encoding_4 = self.encode(128, 256)
        self.encoding_5 = self.encode(256, 512)
        self.encoding_6 = self.encode(512, 1024)

        self.upconv_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv_5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.conv_1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv_4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv_6 = nn.Conv2d(32, self.out_channels, kernel_size=1)

        self.to("cuda")

    def encode(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, floor):
        encoded_1 = self.encoding_1(floor)
        encoded_2 = functional.max_pool2d(self.encoding_2(encoded_1), kernel_size=2)
        encoded_3 = functional.max_pool2d(self.encoding_3(encoded_2), kernel_size=2)
        encoded_4 = functional.max_pool2d(self.encoding_4(encoded_3), kernel_size=2)
        encoded_5 = functional.max_pool2d(self.encoding_5(encoded_4), kernel_size=2)
        encoded_6 = functional.max_pool2d(self.encoding_6(encoded_5), kernel_size=2)

        decoded_1 = self.conv_1(torch.cat([self.upconv_1(encoded_6), encoded_5], dim=1))
        decoded_2 = self.conv_2(torch.cat([self.upconv_2(decoded_1), encoded_4], dim=1))
        decoded_3 = self.conv_3(torch.cat([self.upconv_3(decoded_2), encoded_3], dim=1))
        decoded_4 = self.conv_4(torch.cat([self.upconv_4(decoded_3), encoded_2], dim=1))
        decoded_5 = self.conv_5(torch.cat([self.upconv_5(decoded_4), encoded_1], dim=1))
        decoded_6 = self.conv_6(decoded_5)

        return decoded_6


class FCEncoder(nn.Module):
    def __init__(self, in_features: int, first_out_features: int, repeat: int):
        super().__init__()

        self.in_features = in_features
        self.first_out_features = first_out_features
        self.repeat = repeat

        modules = []
        modules.extend(
            [
                nn.Linear(in_features, first_out_features),
                nn.ReLU(True),
            ]
        )

        temp_out_features = first_out_features
        for _ in range(repeat // 2):
            if temp_out_features // 2 == 0:
                break

            modules.extend(
                [
                    nn.Linear(temp_out_features, temp_out_features // 2),
                    nn.ReLU(True),
                ]
            )

            temp_out_features //= 2

        for _ in range(repeat // 2):
            modules.extend(
                [
                    nn.Linear(temp_out_features, temp_out_features * 2),
                    nn.ReLU(True),
                ]
            )

            temp_out_features *= 2

        modules.extend(
            [
                nn.Linear(temp_out_features, in_features),
                nn.ReLU(True),
            ]
        )

        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        return self.fc(x)


class WallGenerator(UNet, nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(in_channels, out_channels)

        self.fc = FCEncoder(256 * 256, 512, 5)

        self.to("cuda")

    def forward(self, floor):
        encoded = self.fc(floor.reshape(-1, 256 * 256)).reshape(1, 1, 256, 256)
        decoded = super().forward(encoded)

        return torch.sigmoid(decoded)


class RoomAllocator(UNet, nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(in_channels, out_channels)

        self.fc = FCEncoder(256 * 256, 512, 5)

        self.to("cuda")

    def forward(self, walls):
        encoded = self.fc(walls.reshape(-1, 256 * 256)).reshape(1, 1, 256, 256)
        decoded = super().forward(encoded)

        return torch.sigmoid(decoded)