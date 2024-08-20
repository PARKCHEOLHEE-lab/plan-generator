import os
import sys
import torch
import pytorch_lightning as pl

from torch import nn
from typing import List
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

if os.path.abspath(os.path.join(__file__, "../../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

from plan_generator.src.config import Configuration
from plan_generator.src.data import PlanDataset, PlanDataLoader

torch.set_float32_matmul_precision("medium")


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
                            ↓                                    ▲
                            512 → 1024 [16x16]  ───>  1024 [32x32]

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
        self.decoer_blocks = self._create_unet_decoder_blocks()

        self.to(Configuration.DEVICE)

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
            # nn.Conv2d(self.channels_step[i], self.channels_step[i - 1], kernel_size=3, padding=1)
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
            x = self.decoer_blocks[uci](x)
            decoded.append(x)

        x = self.decoer_blocks[-1](x)

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

        self.to(Configuration.DEVICE)

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

        self.to(Configuration.DEVICE)

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


class PlanGenerator(pl.LightningModule):
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

        self.wall_generator_loss_function = nn.BCELoss()
        self.room_allocator_loss_function = nn.CrossEntropyLoss()

        self.automatic_optimization = False

    def forward(self, floor: torch.Tensor, walls: torch.Tensor):
        generated_walls = self.wall_generator(floor)
        allocated_rooms = self.room_allocator(walls)

        return generated_walls, allocated_rooms

    def training_step(self, batch, _):
        wall_generator_optimizer, room_allocator_optimizer = self.optimizers()
        wall_generator_optimizer.zero_grad()
        room_allocator_optimizer.zero_grad()

        floor_batch, walls_batch, rooms_batch = batch

        generated_walls, allocated_rooms = self(floor_batch, walls_batch)

        masked_generated_walls = generated_walls.clone()
        masked_generated_walls[floor_batch == 0] = 0

        masked_allocated_rooms = allocated_rooms.clone()
        masked_allocated_rooms[floor_batch.expand_as(allocated_rooms) == 0] = 0

        wall_generator_loss = self.wall_generator_loss_function(masked_generated_walls, walls_batch)
        room_allocator_loss = self.room_allocator_loss_function(masked_allocated_rooms, rooms_batch.squeeze(1))

        wall_generator_loss.backward()
        room_allocator_loss.backward()

        wall_generator_optimizer.step()
        room_allocator_optimizer.step()

        self.log("train_wall_generator_loss", wall_generator_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_room_allocator_loss", room_allocator_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "train_total_loss", wall_generator_loss + room_allocator_loss, prog_bar=True, on_step=True, on_epoch=True
        )

        return {
            "wall_generator_loss": wall_generator_loss,
            "room_allocator_loss": room_allocator_loss,
            "train_total_loss": wall_generator_loss + room_allocator_loss,
        }

    def on_train_epoch_end(self):
        print("train_epoch_end")

    def validation_step(self, batch, _):
        floor_batch, walls_batch, rooms_batch = batch

        generated_walls, allocated_rooms = self(floor_batch, walls_batch)

        masked_generated_walls = generated_walls.clone()
        masked_generated_walls[floor_batch == 0] = 0

        masked_allocated_rooms = allocated_rooms.clone()
        masked_allocated_rooms[floor_batch.expand_as(allocated_rooms) == 0] = 0

        wall_generator_loss = self.wall_generator_loss_function(masked_generated_walls, walls_batch)
        room_allocator_loss = self.room_allocator_loss_function(masked_allocated_rooms, rooms_batch.squeeze(1))

        self.log("val_wall_generator_loss", wall_generator_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_room_allocator_loss", room_allocator_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "val_total_loss", wall_generator_loss + room_allocator_loss, prog_bar=True, on_step=True, on_epoch=True
        )

        return {
            "val_wall_generator_loss": wall_generator_loss,
            "val_room_allocator_loss": room_allocator_loss,
            "val_total_loss": wall_generator_loss + room_allocator_loss,
        }

    def on_validation_epoch_end(self):
        print("validation_epoch_end")

    def test_step(self):
        pass

    def configure_optimizers(self):
        wall_generator_optimizer = torch.optim.Adam(
            self.wall_generator.parameters(), lr=self.configuration.WALL_GENERATOR_LEARNING_RATE
        )

        wall_generator_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            wall_generator_optimizer, factor=0.1, patience=10, verbose=True
        )

        room_allocator_optimizer = torch.optim.Adam(
            self.room_allocator.parameters(), lr=self.configuration.ROOM_ALLOCATOR_LEARNING_RATE
        )

        room_allocator_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            room_allocator_optimizer, factor=0.1, patience=10, verbose=True
        )

        return [
            {
                "optimizer": wall_generator_optimizer,
                "lr_scheduler": wall_generator_scheduler,
                "monitor": "val_wall_generator_loss",
            },
            {
                "optimizer": room_allocator_optimizer,
                "lr_scheduler": room_allocator_scheduler,
                "monitor": "val_room_allocator_loss",
            },
        ]

    @torch.no_grad()
    def infer(self):
        self.eval()

        # TODO:

        self.train()


if __name__ == "__main__":

    def train():
        config = Configuration()
        model = PlanGenerator(config)

        # Create data loaders
        plan_dataset = PlanDataset()
        plan_dataloader = PlanDataLoader(plan_dataset)

        # Create logger
        logger = TensorBoardLogger(save_dir=config.LOG_DIR)

        # Create checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(logger.log_dir, "checkpoints"),
            filename="{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            monitor="val_loss",
        )

        # Create trainer with specific logger and checkpoint callback
        trainer = pl.Trainer(
            logger=logger,
            callbacks=[checkpoint_callback],
            max_epochs=1,
            accelerator="auto",
            num_nodes=1,
            num_sanity_val_steps=1,
            devices=torch.cuda.device_count(),
            accumulate_grad_batches=8,
        )

        # Fit the model
        trainer.fit(model, plan_dataloader.train_loader, plan_dataloader.validation_loader)

    train()
