import os
import sys
import cv2
import torch
import traceback
import numpy as np
import multiprocessing

from tqdm import tqdm
from typing import List, Dict
from torch.utils.data import Dataset

if os.path.abspath(os.path.join(__file__, "../../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

from plan_generator.src.config import Configuration
from plan_generator.src.enums import LabelsNew, LabelsOld, Colors
from plan_generator.src.utils import runtime_calculator


class PlanDataCreatorHelper:
    """Plan dataset creator helper"""

    @staticmethod
    def read_original_image(image_path: str) -> np.ndarray:
        """Read raw image data

        Args:
            image_path (str): image path

        Returns:
            np.ndarray: original image data
        """

        original_image = None

        try:
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        except:
            print(traceback.format_exc(), image_path)

        return original_image

    @staticmethod
    def process_data(original_image: np.ndarray) -> dict:
        """Process a single image data

        Args:
            original_image (np.ndarray): original image data

        Returns:
            dict: processed data
        """

        processed = {
            "floor": None,
            "walls": None,
            "rooms": None,
        }

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

        # Save the processed data
        processed["floor"] = permuted_binary_floor
        processed["walls"] = permuted_binary_walls
        processed["rooms"] = permuted_rooms

        assert processed["floor"] is not None
        assert processed["walls"] is not None
        assert processed["rooms"] is not None

        return processed

    @staticmethod
    def process_mirroring(data: dict) -> List[Dict]:
        """Mirror data horzontally and vertically

        Args:
            data (dict): processed data

        Returns:
            List[Dict]: mirrored data
        """

        mirrored = []

        floor = data["floor"]
        walls = data["walls"]
        rooms = data["rooms"]

        mirrored_floor_vertically = floor.flip(2)
        mirrored_walls_vertically = walls.flip(2)
        mirrored_rooms_vertically = rooms.flip(2)
        mirrored.append(
            {
                "floor": mirrored_floor_vertically,
                "walls": mirrored_walls_vertically,
                "rooms": mirrored_rooms_vertically,
            }
        )

        mirrored_floor_horizontally = floor.flip(1)
        mirrored_walls_horizontally = walls.flip(1)
        mirrored_rooms_horizontally = rooms.flip(1)
        mirrored.append(
            {
                "floor": mirrored_floor_horizontally,
                "walls": mirrored_walls_horizontally,
                "rooms": mirrored_rooms_horizontally,
            }
        )

        return mirrored

    @staticmethod
    def process_rotating(data: dict) -> List[Dict]:
        """Rotate data 90, 180, and 270 degrees

        Args:
            data (dict): processed data

        Returns:
            List[Dict]: rotated data
        """

        rotated = []

        floor = data["floor"]
        walls = data["walls"]
        rooms = data["rooms"]

        for time in range(1, 4):
            rotated_floor = torch.rot90(floor, time, dims=(1, 2))
            rotated_walls = torch.rot90(walls, time, dims=(1, 2))
            rotated_rooms = torch.rot90(rooms, time, dims=(1, 2))

            rotated.append(
                {
                    "floor": rotated_floor,
                    "walls": rotated_walls,
                    "rooms": rotated_rooms,
                }
            )

        return rotated

    @runtime_calculator
    def _create_dataset(self, mirroring: bool, rotating: bool, slicer: int) -> List[Dict]:
        """Create dataset

        Returns:
            List[Dict]: dataset
        """

        image_names = os.listdir(Configuration.DATA_PATH)
        image_paths = [os.path.join(Configuration.DATA_PATH, image_name) for image_name in image_names][:slicer]

        original_images = []
        plan_dataset = []
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            with tqdm(total=len(image_paths), desc="Reading images") as pbar:
                for original_image in pool.imap(PlanDataCreatorHelper.read_original_image, image_paths, chunksize=10):
                    if original_image is not None:
                        original_images.append(original_image)
                    pbar.update()

            with tqdm(total=len(original_images), desc="Processing images") as pbar:
                for data in pool.imap(PlanDataCreatorHelper.process_data, original_images, chunksize=10):
                    plan_dataset.append(data)
                    pbar.update()

            if mirroring:
                mirrored = []
                with tqdm(total=len(plan_dataset), desc="Mirroring images") as pbar:
                    for data in pool.imap(PlanDataCreatorHelper.process_mirroring, plan_dataset, chunksize=10):
                        mirrored += data
                        pbar.update()

                plan_dataset += mirrored

            if rotating:
                rotated = []
                with tqdm(total=len(plan_dataset), desc="Rotating images") as pbar:
                    for data in pool.imap(PlanDataCreatorHelper.process_rotating, plan_dataset, chunksize=10):
                        rotated += data
                        pbar.update()

                plan_dataset += rotated

        return plan_dataset


class PlanDataCreator(PlanDataCreatorHelper):
    """Plan dataset creator"""

    def __init__(self, mirroring: bool, rotating: bool, slicer: int = int(1e10)):
        self.mirroring = mirroring
        self.rotating = rotating
        self.slicer = slicer

        self.plan_dataset = []

    def create(self) -> None:
        """Create dataset"""

        self.plan_dataset = self._create_dataset(self.mirroring, self.rotating, self.slicer)


class PlanDataset(Dataset):
    def __init__(self, dataset: List[Dict]):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        floor = self.dataset[index]["floor"]
        walls = self.dataset[index]["walls"]
        rooms = self.dataset[index]["rooms"]

        return floor.to(Configuration.DEVICE), walls.to(Configuration.DEVICE), rooms.to(Configuration.DEVICE)


if __name__ == "__main__":
    plan_data_creator = PlanDataCreator(mirroring=True, rotating=True, slicer=500)
    plan_data_creator.create()
