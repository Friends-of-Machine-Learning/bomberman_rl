from abc import ABC
from abc import abstractmethod
from types import SimpleNamespace
from typing import Tuple

import numpy as np

from .utils import DIRECTION_MAP
from .utils import DirectionEnum


class BaseFeature(ABC):
    """
    Base Class for all features.
    """

    def __init__(self, agent: SimpleNamespace, feature_size: int = 1):
        self.agent: SimpleNamespace = agent
        self.feature_size: int = feature_size

    def get_feature_size(self) -> int:
        return self.feature_size

    @abstractmethod
    def state_to_feature(self, agent: SimpleNamespace, game_state: dict) -> np.ndarray:
        """
        Takes the agent and current game_state to calculate a new feature array.

        Parameters
        ----------
        agent : SimpleNamespace
            The SipleNamespace 'self' of the running agent.
        game_state : dict
            The current game_state.
        Returns
        -------
        extracted_feature : np.ndarray
        """
        raise NotImplementedError()


class CoinForceFeature(BaseFeature):
    """
    Applies force to all Coins and determines the absolute force of all Coins onto the agent.
    """

    def __init__(self, agent):
        super().__init__(agent, 2)

    def state_to_feature(self, agent: SimpleNamespace, game_state: dict) -> np.ndarray:
        pos = game_state["self"][3]

        coins = game_state["coins"]

        if coins:
            coin_connection = np.subtract(coins, pos)
            # coin_dist = np.sum(np.abs(coin_connection), axis=1)

            coin_dist = np.linalg.norm(coin_connection, axis=1)

            # Filter our distance of 0, 0 means we already picked up the coin
            valid_coins = coin_dist != 0
            coin_dist = coin_dist[valid_coins]
            coin_connection = coin_connection[valid_coins]

            force = np.sum(coin_connection / np.power(coin_dist, 3)[:, None], axis=0)
        else:
            force = np.zeros(2)
        return force


class WallInDirectionFeature(BaseFeature):
    def __init__(self, agent):
        super().__init__(agent, 4)

    def state_to_feature(self, agent: SimpleNamespace, game_state: dict) -> np.ndarray:
        walls = game_state["field"]
        pos = game_state["self"][3]

        # 1 if wall in that direction, else 0
        return np.array(
            (
                WallInDirectionFeature._is_wall_in_direction(
                    pos, DirectionEnum.UP, walls
                ),
                WallInDirectionFeature._is_wall_in_direction(
                    pos, DirectionEnum.RIGHT, walls
                ),
                WallInDirectionFeature._is_wall_in_direction(
                    pos, DirectionEnum.DOWN, walls
                ),
                WallInDirectionFeature._is_wall_in_direction(
                    pos, DirectionEnum.LEFT, walls
                ),
            )
        )

    @staticmethod
    def _is_wall_in_direction(
        agent_pos: Tuple[int, int], direction: DirectionEnum, walls: np.ndarray
    ) -> int:
        _y, _x = np.add(agent_pos, DIRECTION_MAP[direction])[::-1]
        return int(walls[_y, _x] != 0)


class RandomFeature(BaseFeature):
    def __init__(self, agent: SimpleNamespace):
        super().__init__(agent, 1)

    def state_to_feature(self, agent: SimpleNamespace, game_state: dict) -> np.ndarray:
        return np.array([np.random.randint(-10, 10)])
