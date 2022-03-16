import os
import pickle
import random

import numpy as np

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
        # features.CoinForceFeature(self),
        # features.WallInDirectionFeature(self),
        # features.ClosestCoinFeature(self),
        # features.BFSCoinFeature(self),
        # features.BFSCrateFeature(self),
        # features.BombCrateFeature(self),
        # features.AvoidBombFeature(self),
        # features.CanPlaceBombFeature(self),
        features.InstantDeathDirections(self),
        features.OmegaMovementFeature(self),
        features.NextToCrate(self),
        features.CanPlaceBombFeature(self),
    ]

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")

        feature_size = sum(f.get_feature_size() for f in self.features_used)
        self.model = np.zeros((6, feature_size))
        self.means = np.zeros(6)
        self.n_mean_instances = np.zeros(6)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model, self.means = pickle.load(file)
    self.last_action = np.zeros(6)


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

    features: np.ndarray = state_to_features(self, game_state)
    self.logger.debug("Querying model for action.")

    self.last_action = np.zeros(6)
    action_index = np.argmax(features @ self.model.T + self.means)
    next_action = ACTIONS[action_index]
    self.last_action[action_index] = 1

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
