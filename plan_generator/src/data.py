import os
import sys
import cv2
import pickle
import torch
import numpy as np
import multiprocessing

from typing import List, Tuple
from torch.utils.data import Dataset

if os.path.abspath(os.path.join(__file__, "../../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

from plan_generator.src.config import Configuration
from plan_generator.src.enums import LabelsNew, LabelsOld, Colors
from plan_generator.src.utils import runtime_calculator


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

    @staticmethod
    def process_mirroring(data: torch.Tensor) -> List[torch.Tensor]:
        """Mirror data horzontally and vertically

        Args:
            data (torch.Tensor): processed data

        Returns:
            List[torch.Tensor]: mirrored data
        """

        mirrored = []

        floor = data[0].unsqueeze(0)
        walls = data[1].unsqueeze(0)
        rooms = data[2].unsqueeze(0)

        # Mirror data vertically
        mirrored_floor_vertically = floor.flip(2)
        mirrored_walls_vertically = walls.flip(2)
        mirrored_rooms_vertically = rooms.flip(2)
        mirrored_vertically = torch.vstack(
            [mirrored_floor_vertically, mirrored_walls_vertically, mirrored_rooms_vertically]
        )

        # Mirror data horizontally
        mirrored_floor_horizontally = floor.flip(1)
        mirrored_walls_horizontally = walls.flip(1)
        mirrored_rooms_horizontally = rooms.flip(1)
        mirrored_horizontally = torch.vstack(
            [mirrored_floor_horizontally, mirrored_walls_horizontally, mirrored_rooms_horizontally]
        )

        mirrored = [mirrored_vertically, mirrored_horizontally]

        return mirrored

    @staticmethod
    def process_rotating(data: torch.Tensor) -> List[torch.Tensor]:
        """Rotate data 90, 180, and 270 degrees

        Args:
            data (torch.Tensor): processed data

        Returns:
            List[torch.Tensor]: rotated data
        """

        rotated = []

        floor = data[0].unsqueeze(0)
        walls = data[1].unsqueeze(0)
        rooms = data[2].unsqueeze(0)

        # Rotate data by 90 x `time`
        for time in range(1, 4):
            rotated_floor = torch.rot90(floor, time, dims=(1, 2))
            rotated_walls = torch.rot90(walls, time, dims=(1, 2))
            rotated_rooms = torch.rot90(rooms, time, dims=(1, 2))
            rotated_each = torch.vstack([rotated_floor, rotated_walls, rotated_rooms])

            rotated.append(rotated_each)

        return rotated


class PlanDataCreator(PlanDataCreatorHelper):
    """Plan dataset creator"""

    def __init__(self, mirroring: bool = False, rotating: bool = False, slicer: int = int(1e10)):
        self.mirroring = mirroring
        self.rotating = rotating
        self.slicer = slicer

    @runtime_calculator
    def create(self) -> None:
        """Create dataset"""

        os.makedirs(Configuration.DATA_SAVE_DIR, exist_ok=True)

        files = os.listdir(Configuration.DATA_PATH)
        tasks = [(os.path.join(Configuration.DATA_PATH, file), index) for index, file in enumerate(files)][
            : self.slicer
        ]

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            print("Processing data ...")
            pool.starmap(PlanDataCreatorHelper.process_data, tasks)
            print("Processing is done !")


class PlanDataset(Dataset):
    """Plan dataset"""

    def __init__(self):
        self.dataset_paths = [
            os.path.join(Configuration.DATA_SAVE_DIR, name) for name in os.listdir(Configuration.DATA_SAVE_DIR)
        ]

    def __len__(self) -> int:
        return len(self.dataset_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        floor, walls, rooms = torch.load(self.dataset_paths[index])

        return floor.to(Configuration.DEVICE), walls.to(Configuration.DEVICE), rooms.to(Configuration.DEVICE)
