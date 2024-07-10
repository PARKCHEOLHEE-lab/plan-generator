from enum import Enum, auto


class LabelsOld(Enum):
    """Channel labels, from `plan_generator/data/label.xlsx`"""

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
        LabelsOld.LIVING_ROOM.value: LIVING_ROOM,
        LabelsOld.MASTER_ROOM.value: MASTER_ROOM,
        LabelsOld.KITCHEN.value: KITCHEN,
        LabelsOld.BATHROOM.value: BATHROOM,
        LabelsOld.DINING_ROOM.value: DINING_ROOM,
        LabelsOld.CHILD_ROOM.value: CHILD_ROOM,
        LabelsOld.STUDY_ROOM.value: STUDY_ROOM,
        LabelsOld.SECOND_ROOM.value: SECOND_ROOM,
        LabelsOld.GUEST_ROOM.value: GUEST_ROOM,
        LabelsOld.BALCONY.value: BALCONY,
        LabelsOld.STORAGE.value: STORAGE,
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

    COLOR_MAP_OLD = {
        LabelsOld.LIVING_ROOM.value: GREEN,
        LabelsOld.MASTER_ROOM.value: BLUE,
        LabelsOld.KITCHEN.value: RED,
        LabelsOld.BATHROOM.value: PURPLE,
        LabelsOld.DINING_ROOM.value: YELLOW,
        LabelsOld.CHILD_ROOM.value: ORANGE,
        LabelsOld.STUDY_ROOM.value: MAGENTA,
        LabelsOld.SECOND_ROOM.value: BROWN,
        LabelsOld.GUEST_ROOM.value: BLUE,
        LabelsOld.BALCONY.value: PINK,
        LabelsOld.STORAGE.value: GRAY,
    }

    COLOR_MAP_NEW = {
        LabelsNew.LIVING_ROOM.value: GREEN,
        LabelsNew.MASTER_ROOM.value: BLUE,
        LabelsNew.KITCHEN.value: RED,
        LabelsNew.BATHROOM.value: PURPLE,
        LabelsNew.DINING_ROOM.value: YELLOW,
        LabelsNew.CHILD_ROOM.value: ORANGE,
        LabelsNew.STUDY_ROOM.value: MAGENTA,
        LabelsNew.SECOND_ROOM.value: BROWN,
        LabelsNew.GUEST_ROOM.value: BLUE,
        LabelsNew.BALCONY.value: PINK,
        LabelsNew.STORAGE.value: GRAY,
    }
