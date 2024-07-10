import os
import sys
import cv2
import torch
import numpy as np

from typing import List, Dict
from torch.utils.data import Dataset

if os.path.abspath(os.path.join(__file__, "../../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

from plan_generator.src.config import Configuration
from plan_generator.src.enums import LabelsNew, LabelsOld, Colors


class PlanDataCreatorHelper:
    """Plan dataset creator helper"""

    def create_dataset(self) -> List[Dict]:
        """Create dataset

        Returns:
            List[Dict]: dataset
        """

        plan_dataset = []

        image_names = os.listdir(Configuration.DATA_PATH)
        for image_name in enumerate(image_names):
            image_path = os.path.join(Configuration.DATA_PATH, image_name)

            # Read the original image data
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

            plan_dataset.append(
                {"floor": permuted_binary_floor, "walls": permuted_binary_walls, "rooms": permuted_rooms}
            )

        return plan_dataset

    def mirror_dataset(self, plan_dataset: List[Dict]) -> List[Dict]:
        """Augment dataset by mirroring

        Args:
            plan_dataset (List[Dict]): dataset

        Returns:
            List[Dict]: mirrored dataset
        """

        mirrored_dataset = []
        for data in plan_dataset:
            floor = data["floor"]
            walls = data["walls"]
            rooms = data["rooms"]

            mirrored_floor_vertically = floor[:, :, ::-1]
            mirrored_walls_vertically = walls[:, :, ::-1]
            mirrored_rooms_vertically = rooms[:, :, ::-1]
            mirrored_dataset.append(
                {
                    "floor": mirrored_floor_vertically,
                    "walls": mirrored_walls_vertically,
                    "rooms": mirrored_rooms_vertically,
                }
            )

            mirrored_floor_horizontally = floor[:, ::-1, :]
            mirrored_walls_horizontally = walls[:, ::-1, :]
            mirrored_rooms_horizontally = rooms[:, ::-1, :]
            mirrored_dataset.append(
                {
                    "floor": mirrored_floor_horizontally,
                    "walls": mirrored_walls_horizontally,
                    "rooms": mirrored_rooms_horizontally,
                }
            )

        return mirrored_dataset

    def rotate_dataset(self, plan_dataset: List[Dict]) -> List[Dict]:
        """Augment dataset by rotating

        Args:
            plan_dataset (List[Dict]): dataset

        Returns:
            List[Dict]: rotated dataset
        """

        rotated_dataset = []
        for data in plan_dataset:
            floor = data["floor"]
            walls = data["walls"]
            rooms = data["rooms"]

            for time in range(1, 4):
                rotated_floor = torch.rot90(floor, time, dims=(1, 2))
                rotated_walls = torch.rot90(walls, time, dims=(1, 2))
                rotated_rooms = torch.rot90(rooms, time, dims=(1, 2))

                rotated_dataset.append(
                    {
                        "floor": rotated_floor,
                        "walls": rotated_walls,
                        "rooms": rotated_rooms,
                    }
                )

        return rotated_dataset


class PlanDataCreator(PlanDataCreatorHelper):
    """Plan dataset creator"""

    def __init__(self, mirroring: bool, rotating: bool):
        self.mirroring = mirroring
        self.rotating = rotating
        self.plan_dataset = []

    def create(self) -> None:
        """Create dataset"""

        plan_dataset = self.create_dataset()

        if self.mirroring:
            mirrored_dataset = self.mirror_dataset(plan_dataset)
            plan_dataset += mirrored_dataset

        if self.rotating:
            rotated_dataset = self.rotate_dataset(plan_dataset)
            plan_dataset += rotated_dataset

        self.plan_dataset = plan_dataset


class PlanDataset(Dataset):
    def __init__(self):
        pass
