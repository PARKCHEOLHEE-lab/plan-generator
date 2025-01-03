import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import sys
import pytz
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import List, Tuple, Optional, Union
from IPython.display import clear_output

from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

if os.path.abspath(os.path.join(__file__, "../../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

from plan_generator.src.utils import runtime_calculator
from plan_generator.src.enums import Colors
from plan_generator.src.config import Configuration
from plan_generator.src.data import PlanDataLoader, PlanDataset
from plan_generator.src.models import PlanGenerator, WallGenerator, RoomAllocator


class PlanGeneratorTrainer:
    """Trainer for the `PlanGenerator`"""

    def __init__(
        self,
        configuration: Configuration,
        plan_generator: PlanGenerator,
        plan_dataset: PlanDataset,
        existing_log_dir: Optional[str] = None,
        sanity_checking: bool = False,
        train_loader_subset_count: int = 1,
    ):
        self.configuration = configuration
        self.plan_generator = plan_generator
        self.plan_dataset = plan_dataset
        self.existing_log_dir = existing_log_dir
        self.sanity_checking = sanity_checking
        self.train_loader_subset_count = train_loader_subset_count

        if self.sanity_checking:
            self.plan_dataset.use_transform = False

        self.configuration.set_seed()

        self.is_multi_gpus = not self.sanity_checking and torch.cuda.device_count() > 1
        if self.is_multi_gpus:
            self.plan_generator = nn.DataParallel(self.plan_generator)

        self.plan_generator = self.plan_generator.to(self.configuration.DEVICE)

        self.plan_dataloader = PlanDataLoader(self.plan_dataset)
        self.train_loader_subsets = self._get_train_loader_subsets(train_loader_subset_count, self.plan_dataloader)

        self.summary_writer = None
        self.states = None

        if not self.sanity_checking:
            # Set summary writer
            self.summary_writer = self._get_summary_writer(self.configuration, self.existing_log_dir)

            # Set states of PlanGenerator
            self.states = self._get_states(self.configuration, self.log_dir)  # FIXME

        # Set optimizers
        self.wall_generator_optimizer, self.room_allocator_optimizer = self._get_optimizers(
            self.plan_generator.module.wall_generator if self.is_multi_gpus else self.plan_generator.wall_generator,
            self.plan_generator.module.room_allocator if self.is_multi_gpus else self.plan_generator.room_allocator,
            self.configuration,
            self.states,
        )

        # Set schedulers
        self.wall_generator_scheduler, self.room_allocator_scheduler = self._get_lr_schedulers(
            self.wall_generator_optimizer, self.room_allocator_optimizer, self.configuration, self.states
        )

        # Set loss functions
        self.wall_generator_loss_function, self.room_allocator_loss_function = self._get_loss_functions()

    @property
    def train_loader(self):
        return self.plan_dataloader.train_loader

    @property
    def validation_loader(self):
        return self.plan_dataloader.validation_loader

    @property
    def log_dir(self):
        return self.summary_writer.log_dir

    def _get_summary_writer(self, configuration: Configuration, existing_log_dir: Union[str, None]) -> SummaryWriter:
        """Create tensorboard SummaryWriter. If existing_log_dir is vliad, use it

        Args:
            configuration (Configuration): preset configuration
            existing_log_dir (Union[str, None]): existing log dir created by SummaryWriter

        Returns:
            SummaryWriter: tensorboard SummaryWriter
        """

        log_dir = os.path.join(
            configuration.LOG_DIR, datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m-%d-%Y__%H-%M-%S")
        )

        if existing_log_dir is not None and configuration.STATES_PT in os.listdir(existing_log_dir):
            log_dir = existing_log_dir

        summary_writer = SummaryWriter(log_dir=log_dir)

        return summary_writer

    def _get_states(self, configuration: Configuration, log_dir: str) -> dict:
        """Define states dict

        Args:
            configuration (Configuration): preset configuration
            log_dir (str): directory path for logging

        Returns:
            dict: states to store all results
        """

        if configuration.STATES_PT in os.listdir(log_dir):
            return torch.load(os.path.join(log_dir, configuration.STATES_PT))

        return {
            "epoch": 1,
            "configuration": self.configuration.to_dict(),
            "wall_generator_states": {
                "wall_generator_state_dict": None,
                "wall_generator_optimizer_state_dict": None,
                "wall_generator_scheduler_state_dict": None,
                "wall_generator_loss_avg_train": None,
                "wall_generator_loss_avg_validation": None,
            },
            "room_allocator_states": {
                "room_allocator_state_dict": None,
                "room_allocator_optimizer_state_dict": None,
                "room_allocator_scheduler_state_dict": None,
                "room_allocator_loss_avg_train": None,
                "room_allocator_loss_avg_validation": None,
            },
        }

    def _get_optimizers(
        self, wall_generator: WallGenerator, room_allocator: RoomAllocator, configuration: Configuration, states: dict
    ) -> Tuple[torch.optim.Optimizer]:
        """Define optimizers. If the given states have optimizers' state_dict, they are overwritten.

        Args:
            wall_generator (WallGenerator): wall_generator model
            room_allocator (RoomAllocator): room_allocator model
            configuration (Configuration): preset configuration
            states (dict): states dict

        Returns:
            Tuple[torch.optim.Optimizer]: optimizers
        """

        wall_generator_optimizer = torch.optim.Adam(
            wall_generator.parameters(), lr=configuration.WALL_GENERATOR_LEARNING_RATE
        )

        room_allocator_optimizer = torch.optim.Adam(
            room_allocator.parameters(), lr=configuration.ROOM_ALLOCATOR_LEARNING_RATE
        )

        if states["wall_generator_states"]["wall_generator_optimizer_state_dict"] is not None:
            wall_generator_optimizer.load_state_dict(
                states["wall_generator_states"]["wall_generator_optimizer_state_dict"]
            )

        if states["room_allocator_states"]["room_allocator_optimizer_state_dict"] is not None:
            room_allocator_optimizer.load_state_dict(
                states["room_allocator_states"]["room_allocator_optimizer_state_dict"]
            )

        return wall_generator_optimizer, room_allocator_optimizer

    def _get_lr_schedulers(
        self,
        wall_generator_optimizer: torch.optim.Optimizer,
        room_allocator_optimizer: torch.optim.Optimizer,
        configuration: Configuration,
        states: dict,
    ) -> Tuple[ReduceLROnPlateau]:
        """Define schedulers

        Args:
            wall_generator_optimizer (torch.optim.Optimizer): wall_generator optimizer
            room_allocator_optimizer (torch.optim.Optimizer): room_allocator optimizer
            configuration (Configuration): preset configuration
            states (dict): states dict

        Returns:
            Tuple[ReduceLROnPlateau]: schedulers
        """

        wall_generator_scheduler = ReduceLROnPlateau(
            wall_generator_optimizer,
            factor=configuration.WALL_GENERATOR_LEARNING_RATE_DECAY_FACTOR,
            patience=configuration.WALL_GENERATOR_LEARNING_RATE_DECAY_PATIENCE,
            verbose=True,
        )

        room_allocator_scheduler = ReduceLROnPlateau(
            room_allocator_optimizer,
            factor=configuration.ROOM_ALLOCATOR_LEARNING_RATE_DECAY_FACTOR,
            patience=configuration.ROOM_ALLOCATOR_LEARNING_RATE_DECAY_PATIENCE,
            verbose=True,
        )

        if states["wall_generator_states"]["wall_generator_scheduler_state_dict"] is not None:
            wall_generator_scheduler.load_state_dict(
                states["wall_generator_states"]["wall_generator_scheduler_state_dict"]
            )

        if states["room_allocator_states"]["room_allocator_scheduler_state_dict"] is not None:
            room_allocator_scheduler.load_state_dict(
                states["room_allocator_states"]["room_allocator_scheduler_state_dict"]
            )

        return wall_generator_scheduler, room_allocator_scheduler

    def _get_loss_functions(self) -> Tuple[nn.Module]:
        """Define loss functions to train the model

        Returns:
            Tuple[nn.Module]: loss functions
        """

        wall_generator_loss_function = nn.BCELoss()
        room_allocator_loss_function = nn.CrossEntropyLoss()

        return wall_generator_loss_function, room_allocator_loss_function

    def _get_train_loader_subsets(
        self, train_loader_subset_count: int, plan_dataloader: PlanDataLoader
    ) -> List[Subset]:
        """Divide the training dataloader to subsets with the number of `train_loader_subset_count`

        Args:
            train_loader_subset_count (int): count to divide
            plan_dataloader (PlanDataLoader): dataloader

        Returns:
            List[Subset]: subsets
        """

        train_loader_subsets = [plan_dataloader.train_loader]
        if train_loader_subset_count > 1:
            train_loader_dataset_size = len(plan_dataloader.train_loader.dataset)
            train_loader_indices = list(range(train_loader_dataset_size))
            np.random.shuffle(train_loader_indices)

            subset_divider = train_loader_dataset_size // train_loader_subset_count
            train_loader_subsets = []
            for subset_count in range(train_loader_subset_count):
                subset_start = subset_count * subset_divider
                subset_end = (subset_count + 1) * subset_divider

                if subset_count == train_loader_subset_count - 1:
                    subset_end = train_loader_dataset_size

                train_loader_subset = Subset(
                    plan_dataloader.train_loader.dataset, train_loader_indices[subset_start:subset_end]
                )

                train_loader = DataLoader(
                    dataset=train_loader_subset,
                    batch_size=plan_dataloader.train_loader.batch_size,
                    num_workers=int(os.cpu_count() * 0.7),
                    shuffle=True,
                    drop_last=True,
                    persistent_workers=True,
                )

                train_loader_subsets.append(train_loader)

        return train_loader_subsets

    def _train(
        self,
        configuration: Configuration,
        plan_generator: PlanGenerator,
        wall_generator_loss_function: nn.Module,
        room_allocator_loss_function: nn.Module,
        wall_generator_optimizer: torch.optim.Optimizer,
        room_allocator_optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
    ) -> Tuple[float]:
        """Function to train the model at each epoch

        Args:
            configuration (Configuration): preset configuration
            plan_generator (PlanGenerator): model
            wall_generator_loss_function (nn.Module): wall_generator loss function
            room_allocator_loss_function (nn.Module): room_allocator loss function
            wall_generator_optimizer (torch.optim.Optimizer): wall_generator optimizer
            room_allocator_optimizer (torch.optim.Optimizer): room_allocator optimizer
            train_loader (DataLoader): traininig dataloader

        Returns:
            Tuple[float]: training losses
        """

        wall_generator_loss_sum_train = 0
        room_allocator_loss_sum_train = 0

        accumulation_step = configuration.GRADIENT_ACCUMULATION_STEP

        tqdm_iterator = tqdm(enumerate(train_loader), desc="training...", total=len(train_loader))

        for batch_index, (floor_batch, walls_batch, rooms_batch) in tqdm_iterator:
            # Forward propagation
            generated_walls_masked, allocated_rooms_masked = plan_generator(floor_batch, walls_batch, masking=False)

            # Compute wall generator loss
            wall_generator_loss = wall_generator_loss_function(generated_walls_masked, walls_batch)
            wall_generator_loss /= accumulation_step
            wall_generator_loss_sum_train += wall_generator_loss.item()
            wall_generator_loss.backward()

            # Compute room allocator loss
            room_allocator_loss = room_allocator_loss_function(allocated_rooms_masked, rooms_batch.squeeze(1))
            room_allocator_loss /= accumulation_step
            room_allocator_loss_sum_train += room_allocator_loss.item()
            room_allocator_loss.backward()

            # Accumulate gradient
            if (batch_index + 1) % accumulation_step == 0:
                wall_generator_optimizer.step()
                wall_generator_optimizer.zero_grad()
                room_allocator_optimizer.step()
                room_allocator_optimizer.zero_grad()

        wall_generator_loss_avg_train = wall_generator_loss_sum_train / len(train_loader)
        room_allocator_loss_avg_train = room_allocator_loss_sum_train / len(train_loader)

        return wall_generator_loss_avg_train, room_allocator_loss_avg_train

    @torch.no_grad
    def _validate(
        self,
        plan_generator: PlanGenerator,
        wall_generator_loss_function: nn.Module,
        room_allocator_loss_function: nn.Module,
        validation_loader: DataLoader,
    ) -> Tuple[float]:
        """Function to validate the model at each epoch

        Args:
            plan_generator (PlanGenerator): model
            wall_generator_loss_function (nn.Module): wall_generator loss function
            room_allocator_loss_function (nn.Module): room_allocator loss function
            validation_loader (DataLoader): validation dataloader

        Returns:
            Tuple[float]: validation losses
        """

        # Set mode to eval()
        plan_generator.eval()

        wall_generator_loss_sum_validation = 0
        room_allocator_loss_sum_validation = 0

        for floor_batch, walls_batch, rooms_batch in tqdm(validation_loader, desc="validating..."):
            generated_walls_masked, allocated_rooms_masked = plan_generator(floor_batch, walls_batch, masking=False)

            wall_generator_loss = wall_generator_loss_function(generated_walls_masked, walls_batch)
            wall_generator_loss_sum_validation += wall_generator_loss.item()

            room_allocator_loss = room_allocator_loss_function(allocated_rooms_masked, rooms_batch.squeeze(1))
            room_allocator_loss_sum_validation += room_allocator_loss.item()

        wall_generator_loss_avg_validation = wall_generator_loss_sum_validation / len(validation_loader)
        room_allocator_loss_avg_validation = room_allocator_loss_sum_validation / len(validation_loader)

        # Re-set mode to train()
        plan_generator.train()

        return wall_generator_loss_avg_validation, room_allocator_loss_avg_validation

    @torch.no_grad
    def _write(
        self,
        epoch: int,
        configuration: Configuration,
        plan_generator: PlanGenerator,
        summary_writer: SummaryWriter,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        wall_generator_loss_avg_train: float,
        room_allocator_loss_avg_train: float,
        wall_generator_loss_avg_validation: float,
        room_allocator_loss_avg_validation: float,
        num_to_visualize: int = 4,
    ) -> None:
        """_summary_

        Args:
            plan_generator (PlanGenerator): _description_
            summary_writer (SummaryWriter): _description_
            train_loader (DataLoader): _description_
            validation_loader (DataLoader): _description_
            wall_generator_loss_avg_train (float): _description_
            room_allocator_loss_avg_train (float): _description_
            wall_generator_loss_avg_validation (float): _description_
            room_allocator_loss_avg_validation (float): _description_
            num_to_visualize (int, optional): _description_. Defaults to 2.
        """

        # Log losses
        summary_writer.add_scalar("wall_generator_loss_avg_train", wall_generator_loss_avg_train, epoch)
        summary_writer.add_scalar("room_allocator_loss_avg_train", room_allocator_loss_avg_train, epoch)
        summary_writer.add_scalar("wall_generator_loss_avg_validation", wall_generator_loss_avg_validation, epoch)
        summary_writer.add_scalar("room_allocator_loss_avg_validation", room_allocator_loss_avg_validation, epoch)

        plan_generator.eval()

        train_samples_indices = torch.randperm(len(train_loader.dataset))[:num_to_visualize]
        train_samples = [train_loader.dataset[ti] for ti in train_samples_indices]

        validation_samples_indices = torch.randperm(len(validation_loader.dataset))[:num_to_visualize]
        validation_samples = [validation_loader.dataset[vi] for vi in validation_samples_indices]

        floor_batch = torch.tensor([]).to(configuration.DEVICE)
        walls_batch = torch.tensor([]).to(configuration.DEVICE)
        rooms_batch = torch.tensor([]).to(configuration.DEVICE)

        for train_sample, validation_sample in zip(train_samples, validation_samples):
            train_floor, train_walls, train_rooms = train_sample
            validation_floor, validation_walls, validation_rooms = validation_sample

            floor_batch = torch.cat([floor_batch, train_floor.unsqueeze(0), validation_floor.unsqueeze(0)])
            walls_batch = torch.cat([walls_batch, train_walls.unsqueeze(0), validation_walls.unsqueeze(0)])
            rooms_batch = torch.cat([rooms_batch, train_rooms.unsqueeze(0), validation_rooms.unsqueeze(0)])

        inferrer = plan_generator
        if isinstance(plan_generator, nn.DataParallel):
            inferrer = plan_generator.module

        size = 4
        row = 2
        col = num_to_visualize
        fig, axes = plt.subplots(row, col * 2, figsize=(col * size, row * size / 2))
        samples = ["train", "validation"] * num_to_visualize

        for ax in axes.flatten():
            ax.axis("off")

        generated_walls_images, allocated_rooms_images = inferrer.infer(floor_batch)
        for idx, (generated_wall_image, allocated_room_image) in enumerate(
            zip(generated_walls_images, allocated_rooms_images)
        ):
            axes[0][idx].imshow(generated_wall_image, cmap="gray")
            axes[0][idx].set_title(samples[idx])
            axes[1][idx].imshow(allocated_room_image)
            axes[1][idx].set_title(samples[idx])

        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        summary_writer.add_image("Generated Plans", image_from_plot, epoch, dataformats="HWC")

        plt.close(fig)

        plan_generator.train()

    def _visualize_one(
        self,
        walls: torch.Tensor,
        rooms: torch.Tensor,
        floor_batch: torch.Tensor,
        generated_walls: torch.Tensor,
        allocated_rooms: torch.Tensor,
    ) -> Tuple[np.ndarray]:
        """Function that visualizes the result
        that is compared to ground truth and generated data for qualitative evaluation

        Args:
            walls (torch.Tensor): ground truth for walls
            rooms (torch.Tensor): ground truth for rooms
            floor_batch (torch.Tensor): floor boundary
            generated_walls (torch.Tensor): generated walls by the model
            allocated_rooms (torch.Tensor): generated rooms by the model

        Returns:
            Tuple[np.ndarray]: images
        """

        # Mask cells of `generated_walls` where the cells of the floor_batch are 0
        generated_walls_masked = generated_walls.clone()
        generated_walls_masked[floor_batch == 0] = 0

        # Mask cells of `allocated_rooms` where the cells of the floor_batch are 0
        allocated_rooms_masked = allocated_rooms.clone()
        allocated_rooms_masked[floor_batch.expand_as(allocated_rooms) == 0] = 0

        walls_to_visualize = generated_walls_masked.squeeze(0).squeeze(0)
        walls_to_visualize = (walls_to_visualize.detach().cpu().numpy() > 0.5).astype(int)
        walls_to_visualize = np.where(walls_to_visualize == 0, Colors.WHITE.value[0], Colors.BLACK.value[0])
        walls_to_visualize = self.plan_generator.erode_and_dilate(
            [walls_to_visualize], self.configuration.WALL_EROSION_DILATION_KERNEL_SIZE
        )[0]

        rooms_to_visualize = torch.argmax(allocated_rooms_masked, dim=1).squeeze(0)
        rooms_to_visualize = rooms_to_visualize.detach().cpu().numpy()
        rooms_to_visualize = self.plan_generator.erode_and_dilate(
            [rooms_to_visualize], self.configuration.ROOM_EROSION_DILATION_KERNEL_SIZE
        )[0]

        # Create an empty RGB image
        rooms_channel_3 = np.zeros((*rooms_to_visualize.shape, 3), dtype=np.uint8)
        rooms_channel_3 += Colors.WHITE.value[0]

        # Map each label to its corresponding color
        for label, color in Colors.COLOR_MAP_NEW.value.items():
            mask = rooms_to_visualize == label
            rooms_channel_3[:, :, 0][mask] = color[0]
            rooms_channel_3[:, :, 1][mask] = color[1]
            rooms_channel_3[:, :, 2][mask] = color[2]

        # Image that combines walls and rooms
        walls_and_rooms = rooms_channel_3.copy()
        walls_and_rooms[:, :, 0][walls_to_visualize == 0] = Colors.BLACK.value[0]
        walls_and_rooms[:, :, 1][walls_to_visualize == 0] = Colors.BLACK.value[1]
        walls_and_rooms[:, :, 2][walls_to_visualize == 0] = Colors.BLACK.value[2]

        # Create to visualize the target plan
        walls_np = walls.detach().cpu().numpy().squeeze(0)
        rooms_np = rooms.detach().cpu().numpy().squeeze(0)

        original_walls_and_rooms_channel_3 = np.zeros((*rooms_to_visualize.shape, 3), dtype=np.uint8)
        original_walls_and_rooms_channel_3 += Colors.WHITE.value[0]

        original_walls_and_rooms_channel_3[:, :, 0][walls_np == 1] = Colors.BLACK.value[0]
        original_walls_and_rooms_channel_3[:, :, 1][walls_np == 1] = Colors.BLACK.value[1]
        original_walls_and_rooms_channel_3[:, :, 2][walls_np == 1] = Colors.BLACK.value[2]

        # Map each label to its corresponding color
        for label, color in Colors.COLOR_MAP_NEW.value.items():
            mask = rooms_np == label
            original_walls_and_rooms_channel_3[:, :, 0][mask] = color[0]
            original_walls_and_rooms_channel_3[:, :, 1][mask] = color[1]
            original_walls_and_rooms_channel_3[:, :, 2][mask] = color[2]

        return walls_to_visualize, rooms_channel_3, walls_and_rooms, original_walls_and_rooms_channel_3

    @runtime_calculator
    def sanity_check(self, index: int = 77, epochs: int = 200, visualize: bool = False) -> None:
        """Check sanity that whether the model creates a valid result with a single data

        Args:
            index (int): index to train for sanity checking. Defaults to 77.
            epochs (int): training epochs. Defaults to 200.
            visualize (bool): whether to visualize. Defaults to False.

        """

        floor: torch.Tensor
        walls: torch.Tensor
        rooms: torch.Tensor

        floor, walls, rooms = self.plan_dataset[index]

        floor_batch = floor.unsqueeze(0)
        walls_batch = walls.unsqueeze(0)

        # To check whether the model is overfitted
        wall_generator_loss_final = None
        room_allocator_loss_final = None

        interval = max(1, epochs // 20)

        for epoch in range(1, epochs + 1):
            generated_walls, allocated_rooms = self.plan_generator(floor_batch, walls_batch, masking=False)

            wall_generator_loss = self.wall_generator_loss_function(generated_walls, walls_batch)
            room_allocator_loss = self.room_allocator_loss_function(allocated_rooms, rooms)

            wall_generator_loss.backward()
            self.wall_generator_optimizer.step()
            self.wall_generator_optimizer.zero_grad()

            room_allocator_loss.backward()
            self.room_allocator_optimizer.step()
            self.room_allocator_optimizer.zero_grad()

            if visualize and (epoch % interval == 0 or epoch == 1):
                (
                    walls_to_visualize,
                    rooms_channel_3,
                    walls_and_rooms,
                    original_walls_and_rooms_channel_3,
                ) = self._visualize_one(walls, rooms, floor_batch, generated_walls, allocated_rooms)

                _, axes = plt.subplots(1, 4, figsize=(21, 7))
                ax_1, ax_2, ax_3, ax_4 = axes.flatten()
                ax_1.imshow(walls_to_visualize, cmap="gray")
                ax_1.axis("off")
                ax_1.set_title(f"epoch: {epoch}, loss: {wall_generator_loss.item()} \n", fontsize=10)

                ax_2.imshow(rooms_channel_3)
                ax_2.axis("off")
                ax_2.set_title(f"epoch: {epoch}, loss: {room_allocator_loss.item()} \n", fontsize=10)

                ax_3.imshow(walls_and_rooms)
                ax_3.axis("off")
                ax_3.set_title(f"epoch: {epoch}, merged \n", fontsize=10)

                ax_4.imshow(original_walls_and_rooms_channel_3)
                ax_4.axis("off")
                ax_4.set_title("ground truth \n", fontsize=10)

                plt.show()

            if epoch == epochs:
                wall_generator_loss_final = wall_generator_loss.item()
                room_allocator_loss_final = room_allocator_loss.item()

        # If all losses are close to 0, it worked as intended

        status = f"""
        wall_generator_loss_final: {wall_generator_loss_final}
        room_allocator_loss_final: {room_allocator_loss_final}
        """

        print(status)

    def fit(self) -> None:
        """main method to train model"""

        epoch_start = self.states["epoch"]
        epoch_end = self.configuration.EPOCHS + 1

        wall_generator_current_loss = torch.inf
        room_allocator_current_loss = torch.inf
        wall_generator_loss_avg_validation = torch.inf
        room_allocator_loss_avg_validation = torch.inf

        for epoch in range(epoch_start, epoch_end):
            train_loader_subset_index = (epoch - 1) % len(self.train_loader_subsets)
            train_loader_subset = self.train_loader_subsets[train_loader_subset_index]
            print(f"train_loader_subset_index: {train_loader_subset_index}/{len(self.train_loader_subsets) - 1}")

            # Train
            wall_generator_loss_avg_train, room_allocator_loss_avg_train = self._train(
                self.configuration,
                self.plan_generator,
                self.wall_generator_loss_function,
                self.room_allocator_loss_function,
                self.wall_generator_optimizer,
                self.room_allocator_optimizer,
                train_loader_subset,
            )

            # Validate
            wall_generator_loss_avg_validation, room_allocator_loss_avg_validation = self._validate(
                self.plan_generator,
                self.wall_generator_loss_function,
                self.room_allocator_loss_function,
                self.validation_loader,
            )

            # Update schedulers by validation losses
            self.wall_generator_scheduler.step(wall_generator_loss_avg_validation)
            self.room_allocator_scheduler.step(room_allocator_loss_avg_validation)

            # Update states of `wall_generator` if validation loss is decreased
            is_wall_generator_improved = wall_generator_loss_avg_validation < wall_generator_current_loss
            if is_wall_generator_improved:
                wall_generator_current_loss = wall_generator_loss_avg_validation
                wall_generator = (
                    self.plan_generator.module.wall_generator
                    if self.is_multi_gpus
                    else self.plan_generator.wall_generator
                )

                w_states = self.states["wall_generator_states"]
                w_states["wall_generator_loss_avg_train"] = wall_generator_loss_avg_train
                w_states["wall_generator_loss_avg_validation"] = wall_generator_loss_avg_validation
                w_states["wall_generator_state_dict"] = wall_generator.state_dict()
                w_states["wall_generator_optimizer_state_dict"] = self.wall_generator_optimizer.state_dict()
                w_states["wall_generator_scheduler_state_dict"] = self.wall_generator_scheduler.state_dict()

            # Update states of `room_allocator` if validation loss is decreased
            is_room_allocator_improved = room_allocator_loss_avg_validation < room_allocator_current_loss
            if is_room_allocator_improved:
                room_allocator_current_loss = room_allocator_loss_avg_validation
                room_allocator = (
                    self.plan_generator.module.room_allocator
                    if self.is_multi_gpus
                    else self.plan_generator.room_allocator
                )

                r_states = self.states["room_allocator_states"]
                r_states["room_allocator_loss_avg_train"] = room_allocator_loss_avg_train
                r_states["room_allocator_loss_avg_validation"] = room_allocator_loss_avg_validation
                r_states["room_allocator_state_dict"] = room_allocator.state_dict()
                r_states["room_allocator_optimizer_state_dict"] = self.room_allocator_optimizer.state_dict()
                r_states["room_allocator_scheduler_state_dict"] = self.room_allocator_scheduler.state_dict()

            # Save states if any validation losses have decreased
            if is_wall_generator_improved or is_room_allocator_improved:
                torch.save(self.states, os.path.join(self.log_dir, self.configuration.STATES_PT))
            else:
                self.states = torch.load(os.path.join(self.log_dir, self.configuration.STATES_PT))
                self.states["epoch"] = epoch
                torch.save(self.states, os.path.join(self.log_dir, self.configuration.STATES_PT))

            # Log
            self._write(
                epoch,
                self.configuration,
                self.plan_generator,
                self.summary_writer,
                self.train_loader,
                self.validation_loader,
                wall_generator_loss_avg_train,
                room_allocator_loss_avg_train,
                wall_generator_loss_avg_validation,
                room_allocator_loss_avg_validation,
            )

            clear_output(wait=True)


if __name__ == "__main__":
    configuration = Configuration()

    plan_dataset = PlanDataset(configuration=configuration)
    plan_generator = PlanGenerator(configuration=configuration)

    plan_generator_trainer = PlanGeneratorTrainer(
        configuration=configuration,
        plan_generator=plan_generator,
        plan_dataset=plan_dataset,
        train_loader_subset_count=20,
    )

    print(f"Devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")

    plan_generator_trainer.fit()
