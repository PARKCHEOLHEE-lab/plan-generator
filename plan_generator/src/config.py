import os
import torch
import random
import numpy as np


class DataConfiguration:
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    DATA_PATH = os.path.abspath(os.path.join(DATA_DIR, "plan"))

    IMAGE_SIZE = 256


class ModelConfiguration:
    """Configuration for the model"""

    STEPS = [64, 128, 256, 512, 1024]
    EROSION_AND_DILATION_KERNEL_SIZE = (7, 7)

    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"

    SEED = 777


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
