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
                WallInDirectionFeature.is_wall_in_direction(
                    pos, DirectionEnum.UP, walls
                ),
                WallInDirectionFeature.is_wall_in_direction(
                    pos, DirectionEnum.RIGHT, walls
                ),
                WallInDirectionFeature.is_wall_in_direction(
                    pos, DirectionEnum.DOWN, walls
                ),
                WallInDirectionFeature.is_wall_in_direction(
                    pos, DirectionEnum.LEFT, walls
                ),
            )
        )

    @staticmethod
    def is_wall_in_direction(
        agent_pos: Tuple[int, int], direction: DirectionEnum, walls: np.ndarray
    ) -> int:
        _y, _x = np.add(agent_pos, DIRECTION_MAP[direction])[::-1]
        return int(walls[_y, _x] != 0)


class ClosestCoinFeature(BaseFeature):
    """
    Finds the closest coin to the agent and one hot encodes the direction as a Feature.
    """

    def __init__(self, agent: SimpleNamespace):
        super().__init__(agent, 4)

    def state_to_feature(self, agent: SimpleNamespace, game_state: dict) -> np.ndarray:
        coins = game_state["coins"]

        if not coins:
            return np.array([0, 0, 0, 0])

        pos = game_state["self"][3]
        sx, sy = pos

        walls = game_state["field"]

        coin_connection = np.subtract(coins, pos)
        coin_dist = np.linalg.norm(coin_connection, axis=1)

        closest_coin_index = np.argmin(coin_dist)

        closest_coin_pos = coins[closest_coin_index]

        cx, cy = closest_coin_pos
        up = int(cy < sy)
        right = int(cx > sx)
        down = int(cy > sy)
        left = int(cx < sx)

        up = (
            up
            if not WallInDirectionFeature.is_wall_in_direction(
                pos, DirectionEnum.UP, walls
            )
            else 0
        )
        down = (
            down
            if not WallInDirectionFeature.is_wall_in_direction(
                pos, DirectionEnum.DOWN, walls
            )
            else 0
        )
        left = (
            left
            if not WallInDirectionFeature.is_wall_in_direction(
                pos, DirectionEnum.LEFT, walls
            )
            else 0
        )
        right = (
            right
            if not WallInDirectionFeature.is_wall_in_direction(
                pos, DirectionEnum.RIGHT, walls
            )
            else 0
        )

        dirs = [up, right, down, left]

        if not all([up, right, down, left]):
            rand_dir = np.random.randint(0, 4)
            dirs[rand_dir] = 1

        return np.array(dirs)


class BFSCoinFeature(BaseFeature):
    def __init__(self, agent: SimpleNamespace):
        super().__init__(agent, 4)

    def state_to_feature(self, agent: SimpleNamespace, game_state: dict) -> np.ndarray:
        field = game_state["field"].copy()
        coin_pos = np.array(game_state["coins"])

        if len(coin_pos) == 0:
            # print("no coin pos")
            return np.zeros(self.feature_size)

        field[coin_pos[:, 0], coin_pos[:, 1]] = -2

        self_pos = game_state["self"][3]

        queue = [self_pos]
        parents = np.ones((*field.shape, 2), dtype=int) * -1
        current_pos = queue.pop(0)

        while field[current_pos[0], current_pos[1]] != -2:
            # print(queue)
            for i, j in zip([-1, 1, 0, 0], [0, 0, -1, 1]):
                neighbor = current_pos + np.array([i, j])

                # walls are not valid
                if field[neighbor[0], neighbor[1]] == -1:
                    continue

                # no parent yet
                if parents[neighbor[0], neighbor[1]][0] == -1:
                    parents[neighbor[0], neighbor[1]] = current_pos
                else:
                    continue

                # add to queue
                queue.append(neighbor)
                # print(queue)
            if len(queue) == 0:
                break
            else:
                current_pos = queue.pop(0)

        # no coin found
        if field[current_pos[0], current_pos[1]] != -2:
            return [0, 0, 0, 0]

        # we already stand on it
        # no coin found
        if np.all(current_pos == self_pos):
            return [0, 0, 0, 0]

        while np.any(parents[current_pos[0], current_pos[1]] != self_pos):
            current_pos = parents[current_pos[0], current_pos[1]]

        res = []
        diff = current_pos - self_pos

        if diff[0] < 0:
            # print([1, 0, 0, 0])
            return np.array([1, 0, 0, 0])

        if diff[0] > 0:
            # print([0, 1, 0, 0])
            return np.array([0, 1, 0, 0])

        if diff[1] < 0:
            # print([0, 0, 1, 0])
            return np.array([0, 0, 1, 0])

        if diff[1] > 0:
            # print([0, 0, 0, 1])
            return np.array([0, 0, 0, 1])
