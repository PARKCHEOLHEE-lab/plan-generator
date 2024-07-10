import os


class DataConfiguration:
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    DATA_PATH = os.path.abspath(os.path.join(DATA_DIR, "plan"))

    IMAGE_SIZE = 256


class ModelConfiguration:
    """Configuration for the model"""

    STEPS = [64, 128, 256, 512, 1024]
    EROSION_AND_DILATION_KERNEL_SIZE = (7, 7)


class Configuration(DataConfiguration, ModelConfiguration):
    """Configuration for the plan generator"""
