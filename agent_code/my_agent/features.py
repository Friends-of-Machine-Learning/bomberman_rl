from abc import ABC
from abc import abstractmethod
from types import SimpleNamespace
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

import settings
import settings as s
from .utils import DIRECTION_MAP
from .utils import DirectionEnum

FeatureSpace = Union[np.ndarray, List[Union[float, int]]]


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
    def state_to_feature(
        self, agent: SimpleNamespace, game_state: dict
    ) -> FeatureSpace:
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

    def state_to_feature(
        self, agent: SimpleNamespace, game_state: dict
    ) -> FeatureSpace:
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

    def state_to_feature(
        self, agent: SimpleNamespace, game_state: dict
    ) -> FeatureSpace:
        walls = game_state["field"]
        pos = game_state["self"][3]

        # 1 if wall in that direction, else 0
        walls = np.array(
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
        return walls

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

    def state_to_feature(
        self, agent: SimpleNamespace, game_state: dict
    ) -> FeatureSpace:
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

    def state_to_feature(
        self, agent: SimpleNamespace, game_state: dict
    ) -> FeatureSpace:
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

                # walls and crates are not valid
                if (
                    field[neighbor[0], neighbor[1]] == -1
                    or field[neighbor[0], neighbor[1]] == 1
                ):
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
            # print([0, 0, 0, 0])
            return [0, 0, 0, 0]

        # we already stand on it
        # no coin found
        if np.all(current_pos == self_pos):
            # print([0, 0, 0, 0])
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


class BFSCrateFeature(BaseFeature):
    """
    Find the Closest Crate direction.
    """

    def __init__(self, agent: SimpleNamespace):
        super().__init__(agent, 4)

    def state_to_feature(
        self, agent: SimpleNamespace, game_state: dict
    ) -> FeatureSpace:
        field = game_state["field"].copy()

        self_pos = game_state["self"][3]

        queue = [self_pos]
        parents = np.ones((*field.shape, 2), dtype=int) * -1
        current_pos = queue.pop(0)

        while field[current_pos[0], current_pos[1]] != 1:
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
        if field[current_pos[0], current_pos[1]] != 1:
            # print([0, 0, 0, 0])
            return [0, 0, 0, 0]

        # we already stand on it
        # no coin found
        if np.all(current_pos == self_pos):
            # print([0, 0, 0, 0])
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


class BombCrateFeature(BaseFeature):
    """
    Check if a crate is in bomb range, if so return 1 else 0
    """

    def __init__(self, agent: SimpleNamespace):
        super().__init__(agent, 1)
        self.can_place = np.array([1])

    def state_to_feature(
        self, agent: SimpleNamespace, game_state: dict
    ) -> FeatureSpace:
        field = game_state["field"]
        pos = game_state["self"][3]
        sy, sx = pos

        if 1 in field[sy, sx : sx + s.BOMB_POWER]:
            return self.can_place
        if 1 in field[sy, sx - s.BOMB_POWER - 1 : sx]:  # -1 to fix OBO
            return self.can_place
        if 1 in field[sy : sy + s.BOMB_POWER, sx]:
            return self.can_place
        if 1 in field[sy - s.BOMB_POWER - 1 : sy, sx]:  # -1 to fix OBO
            return self.can_place

        return np.array([0])


class AvoidBombFeature(BaseFeature):
    """
    (bad) return the distance to the bomb in every direction
    """

    def __init__(self, agent: SimpleNamespace):
        super().__init__(agent, 4)
        self.bomb_val = -3  # represents the bomb in the field

    def state_to_feature(
        self, agent: SimpleNamespace, game_state: dict
    ) -> FeatureSpace:
        bombs = game_state["bombs"]
        if not bombs:
            return [s.BOMB_POWER + 1] * 4

        field = game_state["field"].copy()
        for (x, y), _ in bombs:
            field[y, x] = self.bomb_val

        pos = game_state["self"][3]
        sx, sy = pos

        upper = max(sy - s.BOMB_POWER, 0)
        lower = min(sy + s.BOMB_POWER + 1, s.ROWS)
        left = max(sx - s.BOMB_POWER, 0)
        right = min(sx + s.BOMB_POWER + 1, s.COLS)

        relevant_y_up = field[upper : sy + 1, sx][::-1]
        relevant_y_down = field[sy:lower, sx]
        relevant_x_left = field[sy, left : sx + 1][::-1]
        relevant_x_right = field[sy, sx:right]

        ret = [
            np.where(relevant_y_up == self.bomb_val)[0],
            np.where(relevant_y_down == self.bomb_val)[0],
            np.where(relevant_x_left == self.bomb_val)[0],
            np.where(relevant_x_right == self.bomb_val)[0],
        ]
        result = np.array(
            [dist[0] if len(dist) > 0 else s.BOMB_POWER + 1 for dist in ret]
        )
        result -= 2
        return result


class BombViewFeature(BaseFeature):
    def __init__(self, agent: SimpleNamespace):
        super().__init__(agent, 5)
        self.bomb_val = -3  # represents the bomb in the field

    def state_to_feature(
        self, agent: SimpleNamespace, game_state: dict
    ) -> FeatureSpace:

        bombs = game_state["bombs"]
        if not bombs:
            return [0] * 5

        field = game_state["field"].copy()
        for (x, y), _ in bombs:
            field[y, x] = self.bomb_val

        pos = game_state["self"][3]
        sx, sy = pos

        upper = max(sy - s.BOMB_POWER, 0)
        lower = min(sy + s.BOMB_POWER + 1, s.ROWS)
        left = max(sx - s.BOMB_POWER, 0)
        right = min(sx + s.BOMB_POWER + 1, s.COLS)

        relevant_y_up = field[upper:sy, sx][::-1]
        relevant_y_down = field[sy + 1 : lower, sx]
        relevant_x_left = field[sy, left:sx][::-1]
        relevant_x_right = field[sy, sx + 1 : right]

        ret = [
            np.any(relevant_y_up == self.bomb_val),
            np.any(relevant_y_down == self.bomb_val),
            np.any(relevant_x_left == self.bomb_val),
            np.any(relevant_x_right == self.bomb_val),
        ]

        if field[sx, sy] == self.bomb_val:
            ret.append(1)
        else:
            ret.append(0)

        return np.array(ret, dtype=int)


class CanPlaceBombFeature(BaseFeature):
    def __init__(self, agent: SimpleNamespace):
        super().__init__(agent, 1)

    def state_to_feature(
        self, agent: SimpleNamespace, game_state: dict
    ) -> FeatureSpace:
        return [int(game_state["self"][2])]


class ClosestSafeSpaceDirection(BaseFeature):
    def __init__(self, agent: SimpleNamespace):
        super().__init__(agent, 4)
        self.bomb_val = -3  # represents the bomb in the field

    def state_to_feature(
        self, agent: SimpleNamespace, game_state: dict
    ) -> FeatureSpace:
        bombs = game_state["bombs"]
        if not bombs:
            return [0, 0, 0, 0]

        field = game_state["field"].copy()
        for (x, y), _ in bombs:
            field[y, x] = self.bomb_val

        pos = game_state["self"][3]
        sx, sy = pos

        explosionmask = field.copy()
        for (x, y), _ in bombs:
            explosionmask[y, x] = self.bomb_val

            # up
            for i in range(settings.BOMB_POWER + 1):
                if explosionmask[y + i, x] != -1:
                    explosionmask[y + i, x] = -4
                else:
                    break
            # down
            for i in range(settings.BOMB_POWER + 1):
                if explosionmask[y - i, x] != -1:
                    explosionmask[y - i, x] = -4
                else:
                    break
            # left
            for i in range(settings.BOMB_POWER + 1):
                if explosionmask[y, x - i] != -1:
                    explosionmask[y, x - i] = -4
                else:
                    break
            # right
            for i in range(settings.BOMB_POWER + 1):
                if explosionmask[y, x + i] != -1:
                    explosionmask[y, x + i] = -4
                else:
                    break

        if explosionmask[sy, sx] == 0:
            return [0, 0, 0, 0]

        queue = [pos]
        parents = np.ones((*explosionmask.shape, 2), dtype=int) * -1
        current_pos = queue.pop(0)

        while explosionmask[current_pos[0], current_pos[1]] != 0:
            # print(queue)
            for i, j in zip([-1, 1, 0, 0], [0, 0, -1, 1]):
                neighbor = current_pos + np.array([i, j])

                # walls are not valid
                if explosionmask[neighbor[0], neighbor[1]] == -1:
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
        if explosionmask[current_pos[0], current_pos[1]] != 0:
            return [0, 0, 0, 0]

        # we already stand on it
        # no coin found
        if np.all(current_pos == pos):
            return [0, 0, 0, 0]

        while np.any(parents[current_pos[0], current_pos[1]] != pos):
            current_pos = parents[current_pos[0], current_pos[1]]

        res = []
        diff = current_pos - pos

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


class RunawayDirection(BaseFeature):
    def __init__(self, agent: SimpleNamespace):
        super().__init__(agent, 4)
        self.bomb_val = -3  # represents the bomb in the field

    def state_to_feature(
        self, agent: SimpleNamespace, game_state: dict
    ) -> FeatureSpace:
        bombs = game_state["bombs"]
        if not bombs:
            return [1, 1, 1, 1]

        field = game_state["field"].copy()
        for (x, y), _ in bombs:
            field[y, x] = self.bomb_val

        pos = game_state["self"][3]
        sx, sy = pos

        old_bombs = [(x, y) for (x, y), _ in bombs]
        con_old = np.subtract(old_bombs, pos)
        dist_old = np.linalg.norm(con_old, axis=1)
        old_mean = np.mean(dist_old)

        res = [0, 0, 0, 0]

        for index, (i, j) in enumerate(zip([-1, 1, 0, 0], [0, 0, -1, 1])):
            neighbor = pos + np.array([i, j])

            # walls are not valid
            if field[neighbor[0], neighbor[1]] == -1:
                continue

            new_bombs = [(x, y) for (x, y), _ in bombs]
            con_new = np.subtract(new_bombs, neighbor)
            dist_new = np.linalg.norm(con_new, axis=1)
            new_mean = np.mean(dist_new)

            if new_mean > old_mean:
                res[index] = 1

        return res


class NextToCrate(BaseFeature):
    """
    Check if a crate is next to the agent, if so return 1 else 0
    """

    def __init__(self, agent: SimpleNamespace):
        super().__init__(agent, 1)

    def state_to_feature(
        self, agent: SimpleNamespace, game_state: dict
    ) -> FeatureSpace:
        field = game_state["field"]
        pos = game_state["self"][3]
        sy, sx = pos

        place_bomb = np.array([1])

        if 1 in field[sy, sx + 1 : sx + 2]:  # +1 and +2 because right side is exclusive
            return place_bomb
        if 1 in field[sy, sx - 1 : sx]:
            return place_bomb
        if 1 in field[sy + 1 : sy + 2, sx]:
            return place_bomb
        if 1 in field[sy - 1 : sy, sx]:
            return place_bomb

        return np.array([0])


class InstantDeathDirections(BaseFeature):
    """
    Check for every direction if a step would be lethal
    """

    def __init__(self, agent: SimpleNamespace):
        super().__init__(agent, 5)

    def state_to_feature(
        self, agent: SimpleNamespace, game_state: dict
    ) -> FeatureSpace:

        field = game_state["field"]
        next_explosion_map = game_state["explosion_map"].copy() - 1
        next_explosion_map[next_explosion_map < 0] = 0

        for (x, y), t in game_state["bombs"]:

            # skip bombs not about to explode
            if t != 0:
                continue

            # up
            for i in range(settings.BOMB_POWER + 1):
                if field[y + i, x] != -1:
                    next_explosion_map[y + i, x] = 4
                else:
                    break
            # down
            for i in range(settings.BOMB_POWER + 1):
                if field[y - i, x] != -1:
                    next_explosion_map[y - i, x] = 4
                else:
                    break
            # left
            for i in range(settings.BOMB_POWER + 1):
                if field[y, x - i] != -1:
                    next_explosion_map[y, x - i] = 4
                else:
                    break
            # right
            for i in range(settings.BOMB_POWER + 1):
                if field[y, x + i] != -1:
                    next_explosion_map[y, x + i] = 4
                else:
                    break

        pos = game_state["self"][3]
        sy, sx = pos

        res = []
        # all directions
        for (i, j) in zip([-1, 1, 0, 0], [0, 0, -1, 1]):

            # check if explosion will be present in next step
            if next_explosion_map[sy + i, sx + j] > 1:
                res.append(1)
                continue

            res.append(0)

        # own position
        if next_explosion_map[sy, sx] > 1:
            res.append(1)
        else:
            res.append(0)

        return res
