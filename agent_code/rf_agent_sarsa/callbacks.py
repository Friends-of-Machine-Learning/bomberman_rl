import os
import pickle
import random

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from . import features

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]


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
    self.features_used = [
        features.WallInDirectionFeature(self),
        # features.BFSCoinFeature(self),
        # features.BFSCrateFeature(self),
        features.CanPlaceBombFeature(self),
        # features.BombCrateFeature(self),
        features.NextToCrate(self),
        # eatures.BombDistanceDirectionsFeature(self),
        # features.RunawayDirectionFeature(self),
        # features.BombViewFeature(self),
        features.InstantDeathDirections(self),
        features.OmegaMovementFeature(self),
    ]

    KEEP_MODEL: bool = False

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")

        feature_size = sum(f.get_feature_size() for f in self.features_used)

        # Following 2 lines only if previously trained by turn-based mode, else comment out
        if KEEP_MODEL:
            with open("my-saved-model.pt", "rb") as file:
                self.model = pickle.load(file)
        else:
            self.model = [RandomForestRegressor(n_estimators=50) for _ in ACTIONS]
            for tree in self.model:
                tree.fit(np.zeros((1, feature_size)), [0])

    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    self.last_action = np.zeros(6)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    PLAYER_INPUT: bool = False

    random_prob = 0.08
    if not PLAYER_INPUT and self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])

    features: np.ndarray = state_to_features(self, game_state)
    self.logger.debug("Querying model for action.")

    self.last_action = np.zeros(6)
    action_index = np.argmax([tree.predict([features]) for tree in self.model])
    next_action = ACTIONS[action_index]
    self.last_action[action_index] = 1

    # Uncomment the following 2 lines to train the agent by playing yourself in turn-based mode.
    if self.train and PLAYER_INPUT:
        return game_state["user_input"]

    return next_action


def state_to_features(self, game_state: dict) -> np.array:
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

    x_feature = []

    feature: features.BaseFeature
    for feature in self.features_used:
        x = feature.state_to_feature(self, game_state)
        x_feature.append(x)

    return np.concatenate(x_feature)
