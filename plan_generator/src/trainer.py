import os
import sys
import torch
import datetime

from typing import Tuple
from tqdm import tqdm
from IPython.display import clear_output

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

if os.path.abspath(os.path.join(__file__, "../../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))


from plan_generator.src.config import Configuration
from plan_generator.src.data import PlanDataLoader
from plan_generator.src.models import PlanGenerator, WallGenerator, RoomAllocator


class PlanGeneratorTrainer:
    """Trainer for the `PlanGenerator`"""

    def __init__(self, plan_generator: PlanGenerator, plan_dataloader: PlanDataLoader, existing_log_dir: str = None):
        self.plan_generator = plan_generator
        self.plan_dataloader = plan_dataloader
        self.existing_log_dir = existing_log_dir

        # Set summary writer
        self.summary_writer = self._get_summary_writer(self.configuration, self.existing_log_dir)

        # Set states of PlanGenerator
        self.states = self._get_states(self.log_dir)

        # Set optimizers
        self.wall_generator_optimizer, self.room_allocator_optimizer = self._get_optimizers(
            self.plan_generator.wall_generator, self.plan_generator.room_allocator, self.configuration
        )

        # Set schedulers
        self.wall_generator_scheduler, self.room_allocator_scheduler = self._get_lr_schedulers(
            self.wall_generator_optimizer, self.room_allocator_optimizer, self.configuration
        )

        # Set loss functions
        self.wall_generator_loss_function, self.room_allocator_loss_function = self._get_loss_functions()

    @property
    def configuration(self):
        return self.plan_generator.configuration

    @property
    def train_loader(self):
        return self.plan_dataloader.train_loader

    @property
    def validation_loader(self):
        return self.plan_dataloader.validation_loader

    @property
    def log_dir(self):
        return self.summary_writer.log_dir

    def _get_summary_writer(self, configuration: Configuration, existing_log_dir: str) -> SummaryWriter:
        """Create tensorboard SummaryWriter

        Args:
            configuration (Configuration): _description_
            existing_log_dir (str): _description_

        Returns:
            SummaryWriter: _description_
        """

        log_dir = os.path.join(configuration.LOG_DIR, datetime.datetime.now().strftime("%m-%d-%Y__%H-%M-%S"))

        if existing_log_dir is not None:
            log_dir = existing_log_dir

        summary_writer = SummaryWriter(log_dir=log_dir)

        return summary_writer

    def _get_states(self, log_dir: str):
        print(log_dir)  # FIXME: Use log_dir to load states dict of model trained
        states = {}
        return states

    def _get_optimizers(
        self, wall_generator: WallGenerator, room_allocator: RoomAllocator, configuration: Configuration
    ) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        wall_generator_optimizer = torch.optim.Adam(
            wall_generator.parameters(), lr=configuration.WALL_GENERATOR_LEARNING_RATE
        )

        room_allocator_optimizer = torch.optim.Adam(
            room_allocator.parameters(), lr=configuration.ROOM_ALLOCATOR_LEARNING_RATE
        )

        return wall_generator_optimizer, room_allocator_optimizer

    def _get_lr_schedulers(
        self,
        wall_generator_optimizer: torch.optim.Optimizer,
        room_allocator_optimizer: torch.optim.Optimizer,
        configuration: Configuration,
    ) -> Tuple[ReduceLROnPlateau, ReduceLROnPlateau]:
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

        return wall_generator_scheduler, room_allocator_scheduler

    def _get_loss_functions(self) -> Tuple[nn.Module, nn.Module]:
        wall_generator_loss_function = nn.BCELoss()
        room_allocator_loss_function = nn.CrossEntropyLoss()

        return wall_generator_loss_function, room_allocator_loss_function

    def _train(
        self,
        configuration: Configuration,
        plan_generator: PlanGenerator,
        wall_generator_loss_function: nn.Module,
        room_allocator_loss_function: nn.Module,
        wall_generator_optimizer: torch.optim.Optimizer,
        room_allocator_optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        sanity_checking: bool,
    ):
        wall_generator_loss_sum_train = 0
        room_allocator_loss_sum_train = 0

        accumulation_step = configuration.GRADIENT_ACCUMULATION_STEP
        if sanity_checking is True:
            accumulation_step = 1

        tqdm_iterator = tqdm(enumerate(train_loader), desc="training...", total=len(train_loader))

        for batch_index, (floor_batch, walls_batch, rooms_batch) in tqdm_iterator:
            # Forward propagation
            generated_walls_masked, allocated_rooms_masked = plan_generator(floor_batch, walls_batch, masking=True)

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
    ):
        # Set mode to eval()
        self.plan_generator.eval()

        wall_generator_loss_sum_validation = 0
        room_allocator_loss_sum_validation = 0

        for floor_batch, walls_batch, rooms_batch in tqdm(validation_loader, desc="validating..."):
            generated_walls_masked, allocated_rooms_masked = plan_generator(floor_batch, walls_batch, masking=True)

            wall_generator_loss = wall_generator_loss_function(generated_walls_masked, walls_batch)
            wall_generator_loss_sum_validation += wall_generator_loss.item()

            room_allocator_loss = room_allocator_loss_function(allocated_rooms_masked, rooms_batch.squeeze(1))
            room_allocator_loss_sum_validation += room_allocator_loss.item()

        wall_generator_loss_avg_validation = wall_generator_loss_sum_validation / len(validation_loader)
        room_allocator_loss_avg_validation = room_allocator_loss_sum_validation / len(validation_loader)

        # Re-set mode to train()
        self.plan_generator.train()

        return wall_generator_loss_avg_validation, room_allocator_loss_avg_validation

    def fit(self, sanity_checking: bool = False):
        # `wall_generator`-related states
        wall_generator_states = {
            "wall_generator_state_dict": None,
            "wall_generator_optimizer_state_dict": None,
            "wall_generator_scheduler_state_dict": None,
            "wall_generator_loss_avg_train": None,
            "wall_generator_loss_avg_validation": None,
        }

        # `room_allocator`-related states
        room_allocator_states = {
            "room_allocator_state_dict": None,
            "room_allocator_optimizer_state_dict": None,
            "room_allocator_scheduler_state_dict": None,
            "room_allocator_loss_avg_train": None,
            "room_allocator_loss_avg_validation": None,
        }

        # states merged
        states = {
            "epoch": 1,
            "configuration": self.configuration.to_dict(),
            "wall_generator_states": wall_generator_states,
            "room_allocator_states": room_allocator_states,
        }

        # epoch_start = 1
        # epoch_end = self.configuration.EPOCHS + 1

        wall_generator_initial_loss = torch.inf
        room_allocator_initial_loss = torch.inf
        wall_generator_loss_avg_validation = torch.inf
        room_allocator_loss_avg_validation = torch.inf

        for epoch in range(1, self.configuration.EPOCHS + 1):
            # Train
            wall_generator_loss_avg_train, room_allocator_loss_avg_train = self._train(
                self.configuration,
                self.plan_generator,
                self.wall_generator_loss_function,
                self.room_allocator_loss_function,
                self.wall_generator_optimizer,
                self.room_allocator_optimizer,
                self.train_loader,
                sanity_checking,
            )

            if sanity_checking is True:
                # Validate
                wall_generator_loss_avg_validation, room_allocator_loss_avg_validation = self._validate(
                    self.plan_generator,
                    self.wall_generator_loss_function,
                    self.room_allocator_loss_function,
                    self.validation_loader,
                )

            # Save states of `wall_generator`
            if wall_generator_loss_avg_validation < wall_generator_initial_loss:
                w_states = states["wall_generator_states"]
                w_states["wall_generator_loss_avg_train"] = wall_generator_loss_avg_train
                w_states["wall_generator_loss_avg_validation"] = wall_generator_loss_avg_validation
                w_states["wall_generator_state_dict"] = self.plan_generator.wall_generator.state_dict()
                w_states["wall_generator_optimizer_state_dict"] = self.wall_generator_optimizer.state_dict()
                w_states["wall_generator_scheduler_state_dict"] = self.wall_generator_scheduler.state_dict()

            # Save states of `room_allocator`
            if room_allocator_loss_avg_validation < room_allocator_initial_loss:
                r_states = states["room_allocator_states"]
                r_states["room_allocator_loss_avg_train"] = room_allocator_loss_avg_train
                r_states["room_allocator_loss_avg_validation"] = room_allocator_loss_avg_validation
                r_states["room_allocator_state_dict"] = self.plan_generator.room_allocator.state_dict()
                r_states["room_allocator_optimizer_state_dict"] = self.wall_generator_optimizer.state_dict()
                r_states["room_allocator_scheduler_state_dict"] = self.room_allocator_scheduler.state_dict()

            states["epoch"] = epoch

            torch.save(states, os.path.join(self.log_dir, self.configuration.STATES_PT))

            clear_output(wait=True)


if __name__ == "__main__":
    from plan_generator.src.data import PlanDataset

    plan_dataset = PlanDataset(slicer=1)
    plan_dataloader = PlanDataLoader(plan_dataset, batch_size=1)

    configuration = Configuration()
    plan_generator = PlanGenerator(configuration=configuration)
    plan_generator_trainer = PlanGeneratorTrainer(plan_generator=plan_generator, plan_dataloader=plan_dataloader)

    plan_generator_trainer.fit()
