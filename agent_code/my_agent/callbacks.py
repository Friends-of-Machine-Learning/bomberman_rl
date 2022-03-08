import os
import pickle
import random
from enum import Enum
from enum import IntEnum

import numpy as np

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]


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


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")

        self.model = np.zeros((6, 4))
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = 0.1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])

    features: np.ndarray = state_to_features(game_state)

    self.logger.debug("Querying model for action.")

    return ACTIONS[np.argmax(features @ self.model.T)]


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # self x and self y coordinates
    pos = game_state["self"][3]
    sx, sy = pos

    coins = game_state["coins"]
    walls = game_state["field"]

    # Use the coins absolute position to determine wich direction to go
    coins_up = sum(y < sy for _, y in coins)
    coins_right = sum(x > sx for x, _ in coins)
    coins_down = sum(y > sy for _, y in coins)
    coins_left = sum(x < sx for x, _ in coins)
    coins_directions = np.array([coins_up, coins_right, coins_down, coins_left])

    # Encode ['UP', 'RIGHT', 'DOWN', 'LEFT'] as ints, either 1 or 0
    direction_feature = np.zeros(4)
    direction: list = np.argsort(coins_directions).tolist()[::-1]

    d: int = 0
    while direction:
        d = direction.pop(0)  # get the next best direction
        # check if the optimal direction has a wall
        # We need to reverse the np.add because Walls is
        # Y, X indexed
        _y, _x = np.add(pos, DIRECTION_MAP[d])[::-1]
        is_wall = walls[_y, _x] != 0
        if not is_wall:
            break

    direction_feature[d] = 1

    return direction_feature
