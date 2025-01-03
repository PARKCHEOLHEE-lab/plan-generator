import os
import sys
import cv2
import torch
import numpy as np

from PIL import Image
from torch import nn
from typing import List, Tuple, Optional, Union

if os.path.abspath(os.path.join(__file__, "../../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

from plan_generator.src.config import Configuration
from plan_generator.src.enums import Colors


class UNetDecoder(nn.Module):
    """U-Net shaped decoder

    Example:
        When the arguments are as follows:
            in_channels=1, out_channels=1, channels_step=[64, 128, 256, 512, 1024]

        the forward process follows the below structure where the input [1, 1, 256, 256]:
            1 → 64 [256x256]   ───────────────────────────────────────────>  64 [256x256] ··· 1 [256x256]
                ↓                                                                     ▲
                64 → 128 [128x128] ─────────────────────────────────> 128 [128x128]
                    ↓                                                          ▲
                    128 → 256 [64x64]  ─────────────────────────> 256 [64x64]
                        ↓                                               ▲
                        256 → 512 [32x32]  ───────────────> 512 [32x32]
                            ↓                                      ▲
                            512 → 1024 [16x16]  ─────>  1024 [32x32]

        legends:
            →     Conv2d(3x3)
            ↓     MaxPool2d(2x2)
            ─>    Skip connection
            ▲     ConvTranspose2d(2x2)
            ···   Conv2d(1x1)
    """

    def __init__(self, in_channels: int, out_channels: int, channels_step: List[int]):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels_step = channels_step

        self.encoder_blocks = self._create_unet_encoder_blocks()
        self.upconv_blocks = self._create_unet_upconv_blocks()
        self.decoder_blocks = self._create_unet_decoder_blocks()

    def _encoder_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _create_unet_encoder_blocks(self) -> nn.ModuleList:
        encoder_blocks = nn.ModuleList([])
        encoder_blocks += [self._encoder_block(self.in_channels, self.channels_step[0])]
        encoder_blocks += [
            self._encoder_block(self.channels_step[i], self.channels_step[i + 1])
            for i in range(len(self.channels_step) - 1)
        ]

        return encoder_blocks

    def _create_unet_upconv_blocks(self) -> nn.ModuleList:
        upconv_blocks = nn.ModuleList([])
        upconv_blocks += [
            nn.ConvTranspose2d(self.channels_step[i], self.channels_step[i - 1], kernel_size=2, stride=2)
            for i in range(len(self.channels_step) - 1, 0, -1)
        ]

        return upconv_blocks

    def _create_unet_decoder_blocks(self) -> nn.ModuleList:
        decoder_blocks = nn.ModuleList([])
        decoder_blocks += [
            nn.Conv2d(self.channels_step[i - 1] * 2, self.channels_step[i - 1], kernel_size=3, padding=1)
            for i in range(len(self.channels_step) - 1, 0, -1)
        ]

        decoder_blocks += [nn.Conv2d(self.channels_step[0], self.out_channels, kernel_size=1)]

        return decoder_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = []
        for eci, encoder in enumerate(self.encoder_blocks):
            x = encoder(x)
            if eci != 0:
                x = nn.functional.max_pool2d(x, kernel_size=2)

            encoded.append(x)

        decoded = []
        for uci, upconv_block in enumerate(self.upconv_blocks):
            x = torch.cat([upconv_block(x), encoded[len(encoded) - uci - 2]], dim=1)
            x = self.decoder_blocks[uci](x)
            decoded.append(x)

        x = self.decoder_blocks[-1](x)

        return x


class MlpEncoder(nn.Module):
    """Autoencoder shaped mlp encoder

    Example:
        When the arguments are as follows:
            in_features=256x256(65536), initial_out_features=256x2(512), repeat=6

        the forward process follows the below structure:
            256x256(65536) → 256x2(512) → 256 → 128 → 64 → 128 → 256 → 256x2(512) → 256x256(65536)

        legends:
            →     Linear with σ
    """

    def __init__(self, in_features: int, initial_out_features: int, repeat: int):
        super().__init__()

        self.in_features = in_features
        self.initial_out_features = initial_out_features
        self.repeat = repeat

        self.mlp_encoder = self._create_mlp_encoder()

    def _create_mlp_encoder(self) -> nn.Sequential:
        """Create mlp encoder"""

        mlp_encoder_blocks = []
        mlp_encoder_blocks += [nn.Linear(self.in_features, self.initial_out_features), nn.ReLU(inplace=True)]

        repeat_half = self.repeat // 2
        out_features = self.initial_out_features
        for _ in range(repeat_half):
            mlp_encoder_blocks += [nn.Linear(out_features, out_features // 2), nn.ReLU(inplace=True)]
            out_features //= 2

        for _ in range(repeat_half):
            mlp_encoder_blocks += [nn.Linear(out_features, out_features * 2), nn.ReLU(inplace=True)]
            out_features *= 2

        mlp_encoder_blocks += [nn.Linear(out_features, self.in_features), nn.ReLU(inplace=True)]

        mlp_encoder = nn.Sequential(*mlp_encoder_blocks)

        return mlp_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_encoder(x)


class WallGenerator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, channels_step: List[int], size: int, encoder_repeat: int):
        super().__init__()

        self.size = size
        self.encoder_repeat = encoder_repeat

        self.encoder = MlpEncoder(self.size * self.size, self.size * 2, self.encoder_repeat)
        self.decoder = UNetDecoder(in_channels, out_channels, channels_step)

    def forward(self, floor: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(floor.reshape(-1, self.size * self.size)).reshape(-1, 1, self.size, self.size)
        decoded = self.decoder(encoded)

        return torch.sigmoid(decoded)


class RoomAllocator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, channels_step: List[int], size: int, encoder_repeat: int):
        super().__init__()

        self.size = size
        self.encoder_repeat = encoder_repeat

        self.encoder = MlpEncoder(self.size * self.size, self.size * 2, self.encoder_repeat)
        self.decoder = UNetDecoder(in_channels, out_channels, channels_step)

    def forward(self, walls: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(walls.reshape(-1, self.size * self.size)).reshape(-1, 1, self.size, self.size)
        decoded = self.decoder(encoded)

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


class PlanGenerator(nn.Module):
    """Floor plan generator consisting of the `WallGenerator` and `RoomAllocator`"""

    def __init__(self, configuration: Configuration):
        super().__init__()

        self.configuration = configuration

        self.wall_generator = WallGenerator(
            in_channels=self.configuration.WALL_GENERATOR_IN_CHANNELS,
            out_channels=self.configuration.WALL_GENERATOR_OUT_CHANNELS,
            size=self.configuration.IMAGE_SIZE,
            channels_step=self.configuration.WALL_GENERATOR_CHANNELS_STEP,
            encoder_repeat=self.configuration.WALL_GENERATOR_REPEAT,
        )

        self.room_allocator = RoomAllocator(
            in_channels=self.configuration.ROOM_ALLOCATOR_IN_CHANNELS,
            out_channels=self.configuration.ROOM_ALLOCATOR_OUT_CHANNELS,
            size=self.configuration.IMAGE_SIZE,
            channels_step=self.configuration.ROOM_ALLOCATOR_CHANNELS_STEP,
            encoder_repeat=self.configuration.ROOM_ALLOCATOR_REPEAT,
        )

        self.to(self.configuration.DEVICE)

    def forward(
        self, floor_batch: torch.Tensor, walls_batch: torch.Tensor, masking: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        generated_walls = self.wall_generator(floor_batch)
        allocated_rooms = self.room_allocator(walls_batch)

        if masking:
            generated_walls_masked = self.mask(generated_walls, floor_batch)
            allocated_rooms_masked = self.mask(allocated_rooms, floor_batch)

            return generated_walls_masked, allocated_rooms_masked

        return generated_walls, allocated_rooms

    def mask(self, generated_batch: torch.Tensor, floor_batch: torch.Tensor) -> torch.Tensor:
        """Mask cells of `generated_batch` where the cells from the `floor_batch` are 0

        Args:
            generated_batch (torch.Tensor): image to mask
            floor_batch (torch.Tensor): criterion for masking

        Returns:
            torch.Tensor: masked
        """

        masked = generated_batch.clone()

        if generated_batch.shape != floor_batch.shape:
            masked[floor_batch.expand_as(masked) == 0] = 0

        else:
            masked[floor_batch == 0] = 0

        return masked

    def erode_and_dilate(self, images: List[np.ndarray], kernel_size: Optional[Tuple[int]] = None) -> List[np.ndarray]:
        """Erode and dilate a given image

        Args:
            images (List[np.ndarray]): List of images to process
            kernel_size (Optional[Tuple[int]], optional): kernel size to process. Defaults to None.

        Returns:
            List[np.ndarray]: eroded and dilated images
        """

        kernel = np.ones(kernel_size or self.configuration.DEFAULT_EROSION_AND_DILATION_KERNEL_SIZE, np.uint8)

        processed = []
        for image in images:
            image = cv2.erode(image.astype(np.uint8), kernel, iterations=1)
            image = cv2.dilate(image.astype(np.uint8), kernel, iterations=1)

            processed.append(image)

        return processed

    @torch.no_grad
    def infer(
        self, floor_batch: torch.Tensor, masking: bool = True, buffer: bool = True, pil: bool = True
    ) -> Union[Tuple[np.ndarray], Tuple[Image.Image]]:
        """_summary_

        Args:
            floor_batch (torch.Tensor): _description_
            masking (bool, optional): _description_. Defaults to True.
            buffer (bool, optional): _description_. Defaults to True.

        Returns:
            Tuple[torch.Tensor]: _description_
        """

        self.eval()

        generated_walls = self.wall_generator(floor_batch)
        allocated_rooms = self.room_allocator(generated_walls)

        if masking:
            generated_walls = self.mask(generated_walls, floor_batch)
            allocated_rooms = self.mask(allocated_rooms, floor_batch)

        generated_walls_inferred = (generated_walls.detach().cpu().numpy() > 0.5).astype(int)
        generated_walls_inferred = np.where(generated_walls_inferred == 0, Colors.WHITE.value[0], Colors.BLACK.value[0])

        allocated_rooms_inferred = torch.argmax(allocated_rooms, dim=1)
        allocated_rooms_inferred = allocated_rooms_inferred.detach().cpu().numpy()

        if buffer:
            generated_walls_inferred = self.erode_and_dilate(
                generated_walls_inferred, self.configuration.WALL_EROSION_DILATION_KERNEL_SIZE
            )
            allocated_rooms_inferred = self.erode_and_dilate(
                allocated_rooms_inferred, self.configuration.ROOM_EROSION_DILATION_KERNEL_SIZE
            )

        assert len(generated_walls_inferred) == len(allocated_rooms_inferred)

        self.train()

        if pil:
            generated_walls_images = []
            allocated_rooms_images = []
            for generated_wall, allocated_room in zip(generated_walls_inferred, allocated_rooms_inferred):
                generated_wall = generated_wall.squeeze(0)
                generated_wall_image = Image.fromarray(generated_wall)

                # Create an empty RGB image
                allocated_room_channel_3 = np.zeros((*allocated_room.shape, 3), dtype=np.uint8)
                allocated_room_channel_3 += Colors.WHITE.value[0]

                # Map each label to its corresponding color
                for label, color in Colors.COLOR_MAP_NEW.value.items():
                    mask = allocated_room == label
                    allocated_room_channel_3[:, :, 0][mask] = color[0]
                    allocated_room_channel_3[:, :, 1][mask] = color[1]
                    allocated_room_channel_3[:, :, 2][mask] = color[2]

                allocated_room_image = Image.fromarray(allocated_room_channel_3)

                generated_walls_images.append(generated_wall_image)
                allocated_rooms_images.append(allocated_room_image)

            return generated_walls_images, allocated_rooms_images

        else:
            return generated_walls_inferred, allocated_rooms_inferred
