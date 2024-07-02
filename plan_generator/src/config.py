import os


class DataConfiguration:
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    DATA_PATH = os.path.abspath(os.path.join(DATA_DIR, "plan"))


class Configuration(DataConfiguration):
    """Configuration for the plan generator"""
