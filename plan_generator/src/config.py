import os
import sys
import torch
import random
import numpy as np

if os.path.abspath(os.path.join(__file__, "../../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

from plan_generator.src.enums import LabelsNew


class DataConfiguration:
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    DATA_PATH = os.path.abspath(os.path.join(DATA_DIR, "plan"))
    DATA_SAVE_DIR = os.path.abspath(os.path.join(DATA_DIR, "processed"))

    IMAGE_SIZE = 256


class ModelConfiguration:
    """Configuration for the model"""

    CHANNELS_STEP = [64, 128, 256, 512, 1024]
    REPEAT = 5

    EROSION_AND_DILATION_KERNEL_SIZE = (7, 7)

    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"

    SEED = 777

    TRAIN_SIZE = 0.80
    VALIDATION_SIZE = 0.15
    TEST_SIZE = 0.05

    assert np.isclose(TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE, 1.00)

    BATCH_SIZE = 64
    BATCH_SIZE_WITH_DEVICE_COUNT = BATCH_SIZE * torch.cuda.device_count()

    WALL_GENERATOR_IN_CHANNELS = 1
    WALL_GENERATOR_OUT_CHANNELS = 1
    WALL_GENERATOR_CHANNELS_STEP = [64, 128, 256, 512, 1024]
    WALL_GENERATOR_REPEAT = 5
    WALL_GENERATOR_LEARNING_RATE = 0.0002

    ROOM_ALLOCATOR_IN_CHANNELS = 1
    ROOM_ALLOCATOR_OUT_CHANNELS = LabelsNew.NUM_CLASSES.value
    ROOM_ALLOCATOR_CHANNELS_STEP = [64, 128, 256, 512, 768, 896, 1024]
    ROOM_ALLOCATOR_REPEAT = 5
    ROOM_ALLOCATOR_LEARNING_RATE = 0.0001


class Configuration(DataConfiguration, ModelConfiguration):
    """Configuration for the plan generator"""

    LOG_DIR = os.path.abspath(os.path.join(__file__, "../../runs"))

    @staticmethod
    def set_seed(seed: int = ModelConfiguration.SEED):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

        print("CUDA status")
        print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"  DEVICE: {Configuration.DEVICE} \n")

        print("Seeds status:")
        print(f"  Seeds set for torch        : {torch.initial_seed()}")
        print(f"  Seeds set for torch on GPU : {torch.cuda.initial_seed()}")
        print(f"  Seeds set for numpy        : {seed}")
        print(f"  Seeds set for random       : {seed} \n")

        Configuration.SEED = seed
