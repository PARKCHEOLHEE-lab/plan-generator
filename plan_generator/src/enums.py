from enum import Enum, auto


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


class LabelsNew(Enum):
    LIVING_ROOM = 1
    MASTER_ROOM = auto()
    KITCHEN = auto()
    BATHROOM = auto()
    DINING_ROOM = auto()
    CHILD_ROOM = auto()
    STUDY_ROOM = auto()
    SECOND_ROOM = auto()
    GUEST_ROOM = auto()
    BALCONY = auto()
    STORAGE = auto()
    WALL = auto()

    NUM_CLASSES = WALL

    CLASS_MAP = {
        Labels.LIVING_ROOM.value: LIVING_ROOM,
        Labels.MASTER_ROOM.value: MASTER_ROOM,
        Labels.KITCHEN.value: KITCHEN,
        Labels.BATHROOM.value: BATHROOM,
        Labels.DINING_ROOM.value: DINING_ROOM,
        Labels.CHILD_ROOM.value: CHILD_ROOM,
        Labels.STUDY_ROOM.value: STUDY_ROOM,
        Labels.SECOND_ROOM.value: SECOND_ROOM,
        Labels.GUEST_ROOM.value: GUEST_ROOM,
        Labels.BALCONY.value: BALCONY,
        Labels.STORAGE.value: STORAGE,
    }


class Colors(Enum):
    """Colors for rgb"""

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    ORANGE = (255, 165, 0)
    PURPLE = (128, 0, 128)
    PINK = (255, 192, 203)
    BROWN = (165, 42, 42)
    GRAY = (128, 128, 128)

    COLOR_MAP = {
        Labels.LIVING_ROOM.value: GREEN,
        Labels.MASTER_ROOM.value: BLUE,
        Labels.KITCHEN.value: RED,
        Labels.BATHROOM.value: PURPLE,
        Labels.DINING_ROOM.value: YELLOW,
        Labels.CHILD_ROOM.value: ORANGE,
        Labels.STUDY_ROOM.value: MAGENTA,
        Labels.SECOND_ROOM.value: BROWN,
        Labels.GUEST_ROOM.value: BLUE,
        Labels.BALCONY.value: PINK,
        Labels.STORAGE.value: GRAY,
    }
