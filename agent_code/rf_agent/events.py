from abc import ABC
from abc import abstractmethod
from types import SimpleNamespace
from typing import List

import numpy as np

import events as e


class BaseEvent(ABC):
    """
    Base Class for all Custom Events.
    """

    E: str = "BaseEvent"  # The unique Event Name

    def __init__(self, event_name="BaseEvent"):
        BaseEvent.E = event_name
        super().__init__()

    @abstractmethod
    def game_events_occurred(
        self: SimpleNamespace,
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
        self : SimpleNamespace
            The SimpleNamespace 'self' of the running agent.
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
    Append the UselessBombEvent to the events list when the bomb destroyed a crate or killed an enemy.
    """

    def __init__(self, event_name="UselessBombEvent"):
        super().__init__(event_name)

    def game_events_occurred(
        self: SimpleNamespace,
        old_game_state: dict,
        self_action: str,
        new_game_state: dict,
        events: List[str],
    ) -> None:

        if e.BOMB_EXPLODED in events and not (
            e.CRATE_DESTROYED in events or e.KILLED_OPPONENT in events
        ):
            events.append(self.E)
