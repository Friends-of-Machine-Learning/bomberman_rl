from abc import ABC
from abc import abstractmethod
from types import SimpleNamespace
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

import settings as s
from .utils import BFS
from .utils import DIRECTION_MAP
from .utils import DirectionEnum

FeatureSpace = Union[np.ndarray, List[Union[float, int]], Tuple]


class BaseFeature(ABC):
    """
    Base Class for all features.
    """

    def __init__(
        self, agent: SimpleNamespace, feature_size: int = 1, feature_names: dict = None
    ):
        self.agent: SimpleNamespace = agent
        self.feature_size: int = feature_size
        self.feature_names: dict = feature_names if feature_names else {}

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

    def feature_to_readable_name(self, features: FeatureSpace) -> str:
        return f"{self.__class__.__name__}: {self.feature_names.get(tuple(features), str(features))}"


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
        _x, _y = np.add(agent_pos, DIRECTION_MAP[direction])
        return int(walls[_x, _y] != 0)


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
        super().__init__(agent, 2)
        self.coin_val = -2

    def state_to_feature(
        self, agent: SimpleNamespace, game_state: dict
    ) -> FeatureSpace:
        field = game_state["field"].copy()
        coin_pos = np.array(game_state["coins"])

        if len(coin_pos) == 0:
            return np.zeros(self.feature_size)

        field[coin_pos[:, 0], coin_pos[:, 1]] = self.coin_val
        self_pos = game_state["self"][3]
        return BFS(self_pos, field, self.coin_val)


class BFSCrateFeature(BaseFeature):
    """
    Find the Closest Crate direction.
    """

    def __init__(self, agent: SimpleNamespace):
        super().__init__(agent, 2)

    def state_to_feature(
        self, agent: SimpleNamespace, game_state: dict
    ) -> FeatureSpace:
        field = game_state["field"].copy()

        self_pos = game_state["self"][3]
        return BFS(self_pos, field, 1)


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
        sx, sy = pos

        if 1 in field[sx, sy : sy + s.BOMB_POWER]:
            return self.can_place
        if 1 in field[sx, sy - s.BOMB_POWER - 1 : sy]:  # -1 to fix OBO
            return self.can_place
        if 1 in field[sx : sx + s.BOMB_POWER, sy]:
            return self.can_place
        if 1 in field[sx - s.BOMB_POWER - 1 : sx, sy]:  # -1 to fix OBO
            return self.can_place

        return np.array([0])


class BombDistanceDirectionsFeature(BaseFeature):
    """
    Return the distance in every direction to the closest bomb.
    5 means no bomb in proximity.
    0 means standing on bomb.
    """

    def __init__(self, agent: SimpleNamespace):
        super().__init__(agent, 4)
        self.bomb_val = -3  # represents the bomb in the field

    def feature_to_readable_name(self, features: FeatureSpace) -> str:
        up, down, left, right = features
        return f"{self.__class__.__name__}: (U={up}, D={down}, L={left}, R={right})"

    def state_to_feature(
        self, agent: SimpleNamespace, game_state: dict
    ) -> FeatureSpace:
        bombs = game_state["bombs"]
        if not bombs:
            return [s.BOMB_POWER + 1] * 4

        field = game_state["field"].copy()
        for (x, y), _ in bombs:
            field[x, y] = self.bomb_val

        pos = game_state["self"][3]
        sx, sy = pos

        upper = max(sy - s.BOMB_POWER, 0)
        lower = min(sy + s.BOMB_POWER + 1, s.ROWS)
        left = max(sx - s.BOMB_POWER, 0)
        right = min(sx + s.BOMB_POWER + 1, s.COLS)

        relevant_y_up = field[sx, upper : sy + 1][::-1]
        relevant_y_down = field[sx, sy:lower]
        relevant_x_left = field[left : sx + 1, sy][::-1]
        relevant_x_right = field[sx:right, sy]

        ret = [
            np.where(relevant_y_up == self.bomb_val)[0],
            np.where(relevant_y_down == self.bomb_val)[0],
            np.where(relevant_x_left == self.bomb_val)[0],
            np.where(relevant_x_right == self.bomb_val)[0],
        ]
        result = np.array(
            [dist[0] if len(dist) > 0 else s.BOMB_POWER + 1 for dist in ret]
        )
        return result


class BombViewFeature(BaseFeature):
    """
    4 directions + Standing on bomb as feature.
    1 If bomb is present in that direction, else 0.
    """

    def __init__(self, agent: SimpleNamespace):
        super().__init__(agent, 5)
        self.bomb_val = -3  # represents the bomb in the field

    def feature_to_readable_name(self, features: FeatureSpace) -> str:
        u, d, l, r, s = features
        return f"{self.__class__.__name__}: (U={u}, D={d}, L={l}, R={r}, S={s})"

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

        relevant_y_up = field[sx, upper:sy][::-1]
        relevant_y_down = field[sx, sy + 1 : lower]
        relevant_x_left = field[left:sx, sy][::-1]
        relevant_x_right = field[sx + 1 : right, sy]

        ret = [
            np.any(relevant_y_up == self.bomb_val),  # 1 if bomb up
            np.any(relevant_y_down == self.bomb_val),  # 1 if bomb down
            np.any(relevant_x_left == self.bomb_val),  # 1 if bomb left
            np.any(relevant_x_right == self.bomb_val),  # 1 if bomb right
        ]

        # 1 if stand on bomb
        if field[sx, sy] == self.bomb_val:
            ret.append(1)
        else:
            ret.append(0)

        return np.array(ret, dtype=int)


class CanPlaceBombFeature(BaseFeature):
    """
    1 if agent can place bomb, else 0.
    """

    def __init__(self, agent: SimpleNamespace):
        super().__init__(agent, 1)

    def state_to_feature(
        self, agent: SimpleNamespace, game_state: dict
    ) -> FeatureSpace:
        return [int(game_state["self"][2])]


class ClosestSafeSpaceDirection(BaseFeature):
    """
    Uses BFS to find the closest safe space, returns the direction if safe space found.
    """

    _feature_names = {
        DIRECTION_MAP[DirectionEnum.UP]: "Up",
        DIRECTION_MAP[DirectionEnum.DOWN]: "Down",
        DIRECTION_MAP[DirectionEnum.LEFT]: "Left",
        DIRECTION_MAP[DirectionEnum.RIGHT]: "Right",
        (0, 0): "Wait WTF",
    }

    def __init__(self, agent: SimpleNamespace):
        super().__init__(agent, 4, self._feature_names)
        self.bomb_val = -3  # represents the bomb in the field

    def state_to_feature(
        self, agent: SimpleNamespace, game_state: dict
    ) -> FeatureSpace:
        bombs = game_state["bombs"]
        if not bombs:
            return 0, 0

        field = game_state["field"].copy()
        for (x, y), _ in bombs:
            field[x, y] = self.bomb_val

        pos = game_state["self"][3]
        sx, sy = pos

        explosionmask = field.copy()
        for (x, y), _ in bombs:
            explosionmask[x, y] = self.bomb_val

            # up
            for i in range(s.BOMB_POWER + 1):
                if explosionmask[x, y + i] != -1:
                    explosionmask[x, y + i] = -4
                else:
                    break
            # down
            for i in range(s.BOMB_POWER + 1):
                if explosionmask[x, y - i] != -1:
                    explosionmask[x, y - i] = -4
                else:
                    break
            # left
            for i in range(s.BOMB_POWER + 1):
                if explosionmask[x - i, y] != -1:
                    explosionmask[x - i, y] = -4
                else:
                    break
            # right
            for i in range(s.BOMB_POWER + 1):
                if explosionmask[x + i, y] != -1:
                    explosionmask[x + i, y] = -4
                else:
                    break

        if explosionmask[sx, sy] == 0:
            return 0, 0

        return BFS(pos, explosionmask, 0)


class RunawayDirectionFeature(BaseFeature):
    """
    4 directions as feature. Each direction is 1 if it leads away from bomb in average.
    """

    def __init__(self, agent: SimpleNamespace):
        super().__init__(agent, 4)
        self.bomb_val = -3  # represents the bomb in the field

    def feature_to_readable_name(self, features: FeatureSpace) -> str:
        return f"{self.__class__.__name__}: "

    def state_to_feature(
        self, agent: SimpleNamespace, game_state: dict
    ) -> FeatureSpace:
        bombs = game_state["bombs"]
        if not bombs:
            return [1, 1, 1, 1]

        field = game_state["field"].copy()
        for (x, y), _ in bombs:
            field[x, y] = self.bomb_val

        pos = game_state["self"][3]

        old_bombs = [(x, y) for (x, y), _ in bombs]
        con_old = np.subtract(old_bombs, pos)
        dist_old = np.linalg.norm(con_old, axis=1)
        old_mean = np.mean(dist_old)

        res = [0, 0, 0, 0]

        for index, (i, j) in enumerate(DIRECTION_MAP.values()):
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

    _feature_names = {(1,): "True", (0,): "False"}

    def __init__(self, agent: SimpleNamespace):
        super().__init__(agent, 1, self._feature_names)

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
            for i in range(s.BOMB_POWER + 1):
                if field[x, y + i] != -1:
                    next_explosion_map[x, y + i] = 4
                else:
                    break
            # down
            for i in range(s.BOMB_POWER + 1):
                if field[x, y - i] != -1:
                    next_explosion_map[x, y - i] = 4
                else:
                    break
            # left
            for i in range(s.BOMB_POWER + 1):
                if field[x - i, y] != -1:
                    next_explosion_map[x - i, y] = 4
                else:
                    break
            # right
            for i in range(s.BOMB_POWER + 1):
                if field[x + i, y] != -1:
                    next_explosion_map[x + i, y] = 4
                else:
                    break

        pos = game_state["self"][3]
        sy, sx = pos

        res = []
        # all directions
        for (i, j) in zip(DIRECTION_MAP.values()):

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


class OmegaMovementFeature(BaseFeature):
    """
    Manages multiple movement related features.
    """

    _feature_names = {
        DIRECTION_MAP[DirectionEnum.UP]: "Up",
        DIRECTION_MAP[DirectionEnum.DOWN]: "Down",
        DIRECTION_MAP[DirectionEnum.LEFT]: "Left",
        DIRECTION_MAP[DirectionEnum.RIGHT]: "Right",
        (0, 0): "Wait WTF",
    }

    def __init__(self, agent: SimpleNamespace):
        super().__init__(agent, 2, self._feature_names)
        self.coin_feature = BFSCoinFeature(agent)
        self.crate_feature = BFSCrateFeature(agent)
        self.runaway_feature = ClosestSafeSpaceDirection(agent)

    def state_to_feature(
        self, agent: SimpleNamespace, game_state: dict
    ) -> FeatureSpace:
        c_x, c_y = self.coin_feature.state_to_feature(agent, game_state)
        cr_x, cr_y = self.crate_feature.state_to_feature(agent, game_state)
        r_x, r_y = self.runaway_feature.state_to_feature(agent, game_state)

        # Most important, do not die
        if r_x or r_y:
            return r_x, r_y
        if c_x or c_y:
            return c_x, c_y
        return cr_x, cr_y
