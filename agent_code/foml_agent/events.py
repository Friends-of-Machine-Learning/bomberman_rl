from abc import ABC
from abc import abstractmethod
from typing import List

import numpy as np

import events as e
from . import features
from .utils import DIRECTION_MAPSTR


class BaseEvent(ABC):
    """
    Base Class for all Custom Events.
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return self.__class__.__name__

    @abstractmethod
    def game_events_occurred(
        self,
        old_game_state: dict,
        self_action: str,
        new_game_state: dict,
        events: List[str],
    ) -> None:
        """
        Takes the old and new game state to compute new custom Events.
        New events might be appended to the passed in events list.

        Parameters
        ----------
        old_game_state : dict
            The last game_state.
        self_action : str
            The last action performed.
        new_game_state : dict
            The following game_state.
        events : List[str]
            List of events this step.
        """
        raise NotImplementedError()


class UselessBombEvent(BaseEvent):
    """
    Event for when the agent placed a bomb although the bomb can't destroy anything in that place.
    """

    def __init__(self):
        super().__init__()

    def game_events_occurred(
        self,
        old_game_state: dict,
        self_action: str,
        new_game_state: dict,
        events: List[str],
    ) -> None:
        bomb_feature = features.BombCrateFeature(None)

        if (
            not bool(bomb_feature.state_to_feature(None, old_game_state)[0])
            and self_action == "BOMB"
        ):
            events.append(str(self))


class AvoidDeathEvent(BaseEvent):
    def __init__(self):
        super().__init__()

    def game_events_occurred(
        self,
        old_game_state: dict,
        self_action: str,
        new_game_state: dict,
        events: List[str],
    ) -> None:
        death_feature = features.InstantDeathDirectionsFeatures(None)

        deadly_moves = death_feature.state_to_feature(None, old_game_state)
        u, d, l, r, s = deadly_moves

        action_to_move = {
            "UP": u,
            "DOWN": d,
            "LEFT": l,
            "RIGHT": r,
            "WAIT": s,
            "BOMB": s,
        }
        # If the agent made a move that is not 1 in InstantDeathDirectionsFeatures, append the event.
        if not bool(action_to_move[self_action]) and 1 in deadly_moves:
            events.append(str(self))


class PlacedGoodBombEvent(BaseEvent):
    """
    Add event if the agent placed a bomb that might hit a crate or enemy.
    """

    def __init__(self):
        super().__init__()

    def game_events_occurred(
        self,
        old_game_state: dict,
        self_action: str,
        new_game_state: dict,
        events: List[str],
    ) -> None:
        bomb_feature = features.BombCrateFeature(None)

        if (
            bool(bomb_feature.state_to_feature(None, old_game_state)[0])
            and self_action == "BOMB"
        ):
            events.append(str(self))


class NewFieldEvent(BaseEvent):
    """
    Add event if agent moved to a new field.
    """

    def __init__(self):
        super().__init__()
        self.visited = []

    def game_events_occurred(
        self,
        old_game_state: dict,
        self_action: str,
        new_game_state: dict,
        events: List[str],
    ) -> None:
        if not new_game_state:
            return
        new_pos = new_game_state["self"][3]
        if new_pos in self.visited:
            return

        self.visited.append(new_pos)
        events.append(str(self))


class DestroyedAnyCrate(BaseEvent):
    def game_events_occurred(
        self,
        old_game_state: dict,
        self_action: str,
        new_game_state: dict,
        events: List[str],
    ) -> None:
        if e.CRATE_DESTROYED in events:
            events.append(str(self))


class SmartBombEvent(BaseEvent):
    def __init__(self):
        super().__init__()

    def game_events_occurred(
        self,
        old_game_state: dict,
        self_action: str,
        new_game_state: dict,
        events: List[str],
    ) -> None:
        bomb_feature = features.BombCrateFeature(None)
        bomb_suicide = features.BombIsSuicideFeature(None)

        if (
            self_action == "BOMB"
            and bool(bomb_feature.state_to_feature(None, old_game_state)[0])
            and not bool(bomb_suicide.state_to_feature(None, old_game_state)[0])
        ):
            events.append(str(self))


class FollowOmegaEvent(BaseEvent):
    """
    Add event if agent moved towards the closest reachable coin.
    """

    def __init__(self):
        super().__init__()

    def game_events_occurred(
        self,
        old_game_state: dict,
        self_action: str,
        new_game_state: dict,
        events: List[str],
    ) -> None:
        if not new_game_state:
            return

        wished_direction = np.array(
            features.OmegaMovementFeature(None).state_to_feature(None, old_game_state)
        )
        moved = DIRECTION_MAPSTR.get(self_action, (0, 0))

        if wished_direction[0] == moved[0] and wished_direction[1] == moved[1]:
            events.append(str(self))
