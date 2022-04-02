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
    self.feature_print: bool = False  # type: ignore

    # This agents features
    self.features_used = [
        features.BFSCoinFeature(self),
        features.BFSCrateFeature(self),
        features.BombCrateFeature(self),
        features.CanPlaceBombFeature(self),
        features.ClosestSafeSpaceDirection(self),
        features.InstantDeathDirectionsFeatures(self),
        features.BombIsSuicideFeature(self),
    ]

    self.keep_model: bool = False  # type: ignore

    # Create new model if it does not exist or in training mode
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")

        if self.keep_model:
            with open("my-saved-model.pt", "rb") as file:
                self.model = pickle.load(file)
        else:
            feature_dims: list = []
            for f in self.features_used:
                for _ in range(f.get_feature_size()):
                    feature_dims.append(f.get_feature_dims())
            feature_dims.append(len(ACTIONS))
            self.model = np.zeros(tuple(feature_dims))

    # Load model if it exists and not in training mode
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

    random_prob = 0.1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])

    features: np.ndarray = state_to_features(self, game_state, True)
    self.logger.debug("Querying model for action.")

    # Get the action with the highest probability
    action_index = np.argmax(self.model[tuple(features)])
    next_action = ACTIONS[action_index]
    return next_action


def state_to_features(self, game_state: dict, temp: bool = False) -> np.ndarray:
    """
    Converts the game state to a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :param temp: If true, self.print_features will be used to determine if features should be printed.
    :return: The feature vector as a numpy array.
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    x_feature = []
    feature: features.BaseFeature
    # Iterate over all features and get the feature vector, print it if in debug mode
    if self.feature_print and temp:
        feature_debug_text: list = []
        for feature in self.features_used:
            x = feature.state_to_feature(self, game_state)
            feature_debug_text.append(feature.feature_to_readable_name(x))
            x_feature.append(x)
        print(", ".join(feature_debug_text))

    # Else just get the feature vector
    else:
        for feature in self.features_used:
            x_feature.append(feature.state_to_feature(self, game_state))

    return np.concatenate(x_feature).astype(int)
