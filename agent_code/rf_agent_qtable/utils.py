from enum import Enum
from enum import IntEnum
from typing import Tuple

import numpy as np


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


def BFS(
    self_pos: Tuple[int, int], field: np.ndarray, goal: int = 1, ignore: int = -100
) -> Tuple[int, int]:
    """
    Breath First Search Algorithm to find a given value in an 2 dimensional np.ndarray.
    Takes the start position, the field and the goal value to search. Returns (X, Y) direction Tuple.
    :param self_pos: The start position to search from.
    :param field: The 2dimensional np.ndarray. Can move only on 0 value fields. Any other value will be handled as wall.
    :param goal: The value we search in the given field.
    :return:  Tuple[int, int] with X and Y coordinate as next best direction.
    """
    queue = [self_pos]
    parents = np.ones((*field.shape, 2), dtype=int) * -1
    current_pos = queue.pop(0)

    while field[current_pos[0], current_pos[1]] != goal:
        for i, j in DIRECTION_MAP.values():
            neighbor = current_pos + np.array([i, j])
            n_x, n_y = neighbor

            # Skip invalid fields
            # Only 0 and goal is valid to move to.
            if not (
                field[n_x, n_y] == 0
                or field[n_x, n_y] == goal
                or field[n_x, n_y] == ignore
            ):
                continue

            # no parent yet
            if parents[n_x, n_y][0] == -1:
                parents[n_x, n_y] = current_pos
            else:
                continue

            # add to queue
            queue.append(neighbor)
        if len(queue) == 0:
            break
        else:
            current_pos = queue.pop(0)

    # Goal could not be found
    if field[current_pos[0], current_pos[1]] != goal:
        return (0, 0)

    # We stand on the goal, don't move
    if np.all(current_pos == self_pos):
        return (0, 0)

    while np.any(parents[current_pos[0], current_pos[1]] != self_pos):
        current_pos = parents[current_pos[0], current_pos[1]]

    diff = current_pos - self_pos

    # X coordinate to the left
    if diff[0] < 0:
        return DIRECTION_MAP[DirectionEnum.LEFT]

    # X coordinate to the right
    if diff[0] > 0:
        return DIRECTION_MAP[DirectionEnum.RIGHT]

    # Y coordinate Up
    if diff[1] < 0:
        return DIRECTION_MAP[DirectionEnum.UP]

    # Y coordinate Down
    if diff[1] > 0:
        return DIRECTION_MAP[DirectionEnum.DOWN]
