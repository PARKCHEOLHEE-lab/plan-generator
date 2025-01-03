import os
import sys
import cv2
import random
import pickle
import torch
import numpy as np
import multiprocessing

from typing import Tuple
from torch.utils.data import Dataset, DataLoader, random_split

if os.path.abspath(os.path.join(__file__, "../../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

from plan_generator.src.config import Configuration
from plan_generator.src.enums import LabelsNew, LabelsOld, Colors
from plan_generator.src.utils import runtime_calculator
from plan_generator.src.transforms import TransformMirroring, TransformRotating


class PlanDataCreatorHelper:
    """Plan dataset creator helper"""

    @staticmethod
    def process_data(image_path: str, index: int) -> torch.Tensor:
        """Process a single image data

        Args:
            original_image (np.ndarray): original image data

        Returns:
            torch.Tensor: processed data
        """

        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Extract the exterior walls
        exterior_walls_mask = (
            (original_image[:, :, 0] == LabelsOld.EXTERIOR_WALL_1.value)
            | (original_image[:, :, 0] == LabelsOld.MAIN_ENTRANCE.value)
            | (original_image[:, :, 1] == LabelsOld.EXTERIOR_WALL_2.value)
        )

        exterior_walls = np.zeros_like(original_image) + Colors.WHITE.value[0]
        exterior_walls[exterior_walls_mask] = Colors.BLACK.value

        # Extract the interior walls
        interior_walls = np.zeros_like(original_image) + Colors.WHITE.value[0]
        interior_walls_mask = (
            (original_image[:, :, 1] == LabelsOld.INTERIOR_WALL.value)
            | (original_image[:, :, 1] == LabelsOld.INTERIOR_DOOR.value)
            | (original_image[:, :, 1] == LabelsOld.WALL_IN.value)
        )
        interior_walls[interior_walls_mask] = Colors.BLACK.value

        # Merge all walls into one
        walls = np.zeros_like(original_image) + Colors.WHITE.value[0]
        walls[exterior_walls_mask] = Colors.BLACK.value
        walls[interior_walls_mask] = Colors.BLACK.value

        # Convert the walls to the binary-shaped data
        binary_walls = np.zeros((walls.shape[0], walls.shape[1], 1))
        binary_walls[walls[:, :, 0] == 0] = 1

        # Extract the floor boundary
        floor = np.zeros_like(original_image) + Colors.WHITE.value[0]
        floor[exterior_walls_mask] = Colors.BLACK.value
        floor = cv2.floodFill(floor, mask=None, seedPoint=(0, 0), newVal=(0, 0, 0))[1]
        floor = abs(Colors.WHITE.value[0] - floor)
        floor[exterior_walls_mask] = Colors.BLACK.value

        # Convert the floor boundary to the binary-shaped data
        binary_floor = np.zeros((floor.shape[0], floor.shape[1], 1))
        binary_floor[floor[:, :, 0] == 0] = 1

        # Extract rooms
        rooms = np.zeros(shape=(original_image.shape[0], original_image.shape[1], 1))
        room_types = np.unique(original_image[:, :, 1])
        for room_type in room_types:
            room_color = Colors.COLOR_MAP_OLD.value.get(room_type)
            if room_color is not None:
                room_mask = original_image[:, :, 1] == room_type
                rooms[room_mask] = LabelsNew.CLASS_MAP.value[room_type]

        # Permute data shape as (n, c, h, w) to fit the Pytorch
        permuted_binary_floor = torch.FloatTensor(binary_floor).permute(2, 0, 1)
        permuted_binary_walls = torch.FloatTensor(binary_walls).permute(2, 0, 1)
        permuted_rooms = torch.LongTensor(rooms).permute(2, 0, 1)

        processed = torch.vstack([permuted_binary_floor, permuted_binary_walls, permuted_rooms])

        # Save preprocessed data
        torch.save(
            obj=processed,
            f=os.path.join(Configuration.DATA_SAVE_DIR, f"{index}.pt"),
            pickle_protocol=pickle.HIGHEST_PROTOCOL,
        )


class PlanDataCreator(PlanDataCreatorHelper):
    """Plan dataset creator"""

    def __init__(self, slicer: int = int(1e10)):
        self.slicer = slicer

    @runtime_calculator
    def create(self) -> None:
        """Create dataset"""

        os.makedirs(Configuration.DATA_SAVE_DIR, exist_ok=True)

        files = os.listdir(Configuration.DATA_PATH)[: self.slicer]
        tasks = [(os.path.join(Configuration.DATA_PATH, file), index) for index, file in enumerate(files)]

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            print("Processing data ...")
            pool.starmap(PlanDataCreatorHelper.process_data, tasks)
            print("Processing is done !")


class PlanDataset(Dataset):
    """Plan dataset"""

    def __init__(self, configuration: Configuration, slicer=int(1e10), use_transform: bool = True):
        self.configuration = configuration
        self.slicer = slicer
        self.use_transform = use_transform

        files = os.listdir(Configuration.DATA_SAVE_DIR)[: self.slicer]
        self.dataset_paths = [os.path.join(Configuration.DATA_SAVE_DIR, name) for name in files]

        # No fixed seed
        self.local_random = random.Random()
        self.transform_mirroring = TransformMirroring()
        self.transform_rotating = TransformRotating()

        torch.multiprocessing.set_start_method("spawn", force=True)

    def __len__(self) -> int:
        return len(self.dataset_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        """Using On-the-fly data augmentation method for the memory efficiency and the generalization

        - Mirroring
            0: none
            1: horizontal
            2: vertical

        - Rotating
            0: none
            1: 90 degree
            2: 180 degree
            3: 270 degree
        """

        floor: torch.Tensor
        walls: torch.Tensor
        rooms: torch.Tensor

        floor, walls, rooms = torch.load(self.dataset_paths[index])

        floor = floor.unsqueeze(0)
        walls = walls.unsqueeze(0)
        rooms = rooms.unsqueeze(0)

        # mirroring data
        mirroring_dimension = self.local_random.choice((0, 1, 2))
        if self.use_transform and mirroring_dimension in (1, 2):
            floor, walls, rooms = self.transform_mirroring((floor, walls, rooms), mirroring_dimension)

        # rotating data
        rotation_multiplier = self.local_random.choice((0, 1, 2, 3))
        if self.use_transform and rotation_multiplier in (1, 2, 3):
            floor, walls, rooms = self.transform_rotating((floor, walls, rooms), rotation_multiplier)

        return (
            floor.to(self.configuration.DEVICE),
            walls.to(self.configuration.DEVICE),
            rooms.long().to(self.configuration.DEVICE),
        )


class PlanDataLoader:
    def __init__(self, plan_dataset: PlanDataset, batch_size: int = Configuration.BATCH_SIZE):
        self.plan_dataset = plan_dataset
        self.batch_size = batch_size

        # split dataset
        self.train_dataset, self.validation_dataset, self.test_dataset = random_split(
            self.plan_dataset, [Configuration.TRAIN_SIZE, Configuration.VALIDATION_SIZE, Configuration.TEST_SIZE]
        )

        # assign data loaders
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=int(os.cpu_count() * 0.7),
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
        )

        self.validation_loader = DataLoader(
            dataset=self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=int(os.cpu_count() * 0.2),
            shuffle=False,
            drop_last=True,
            persistent_workers=True,
        )

        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=int(os.cpu_count() * 0.1),
            shuffle=False,
            drop_last=True,
            persistent_workers=True,
        )
