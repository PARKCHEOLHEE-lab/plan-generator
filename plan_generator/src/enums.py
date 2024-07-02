from enum import Enum


class Labels(Enum):
    """Channel labels"""

    EXTERIOR_WALL_1 = 127
    MAIN_ENTRANCE = 255
    OTHER = 0

    LIVING_ROOM = 0
    MASTER_ROOM = 1
    KITCHEN = 2
    BATHROOM = 3
    DINING_ROOM = 4
    CHILD_ROOM = 5
    STUDY_ROOM = 6
    SECOND_ROOM = 7
    GUEST_ROOM = 8
    BALCONY = 9
    ENTRANCE = 10
    STORAGE = 11
    WALL_IN = 12
    EXTERNAL_AREA = 13
    EXTERIOR_WALL_2 = 14
    FRONT_DOOR = 15
    INTERIOR_WALL = 16
    INTERIOR_DOOR = 17


class Colors(Enum):
    """Colors for rgb"""

    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    ORANGE = (255, 165, 0)
    PURPLE = (128, 0, 128)
    LIME = (0, 255, 0)
    PINK = (255, 192, 203)
    TEAL = (0, 128, 128)
    LAVENDER = (230, 230, 250)
    BROWN = (165, 42, 42)
    MAROON = (128, 0, 0)
    OLIVE = (128, 128, 0)
    NAVY = (0, 0, 128)
    GRAY = (128, 128, 128)
