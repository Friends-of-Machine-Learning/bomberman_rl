from enum import Enum
from enum import IntEnum


class DirectionEnum(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class ActionNames(Enum):
    UP = "UP"
    RIGHT = "RIGHT"
    DOWN = "DOWN"
    LEFT = "LEFT"
    WAIT = "WAIT"
    BOMB = "BOMB"


ACTION_TO_INDEX = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3, "WAIT": 4, "BOMB": 5}

DIRECTION_MAP = {
    DirectionEnum.UP: (0, -1),
    DirectionEnum.RIGHT: (1, 0),
    DirectionEnum.DOWN: (0, 1),
    DirectionEnum.LEFT: (-1, 0),
}
OPPOSITE_DIRECTION = {
    "UP": "DOWN",
    "DOWN": "UP",
    "LEFT": "RIGHT",
    "RIGHT": "LEFT",
    "BOMB": "BOMB",
    "WAIT": "NONE",
}
