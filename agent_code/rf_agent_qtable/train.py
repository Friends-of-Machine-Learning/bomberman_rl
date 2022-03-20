import pickle
from collections import deque
from collections import namedtuple
from typing import List

import numpy as np

import events as e
from . import events as ev
from .callbacks import ACTIONS
from .callbacks import state_to_features
from .utils import ACTION_TO_INDEX
from .utils import OPPOSITE_DIRECTION

# This is only an example!
Transition = namedtuple(
    "Transition", ("action", "feature", "next_feature", "steps", "rounds", "reward")
)

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 800 * 4  # keep only ... last transitions
END_TRANSITION_HISTORY_SIZE = 20 * 4  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
BACKTRACK_EVENT = "BACKTRACK"
RUNAWAY_EVENT = "RUNAWAY"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    # self.begin_transition = []
    self.custom_events = [ev.UselessBombEvent()]
    self.transitions = []
    self.end_transitions = []


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(
        f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}'
    )

    if old_game_state is None:
        return

    for custom_event in self.custom_events:
        custom_event.game_events_occurred(
            old_game_state, self_action, new_game_state, events
        )

    old_state = state_to_features(self, old_game_state)
    new_state = state_to_features(self, new_game_state)
    rewards = (reward_from_events(self, events),)

    update_q_table(self, old_state, new_state, rewards, ACTION_TO_INDEX[self_action])


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(
        f'Encountered event(s) {", ".join(map(repr, events))} in final step'
    )

    for custom_event in self.custom_events:
        custom_event.game_events_occurred(last_game_state, last_action, None, events)

    end_round_q_table(
        self,
        state_to_features(self, last_game_state),
        reward_from_events(self, events),
        ACTION_TO_INDEX[last_action],
    )
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        # GOOD
        e.COIN_COLLECTED: 5,
        e.CRATE_DESTROYED: 2,
        e.MOVED_UP: 1,
        e.MOVED_DOWN: 1,
        e.MOVED_LEFT: 1,
        e.MOVED_RIGHT: 1,
        e.BOMB_DROPPED: 0.5,
        # BAD
        e.KILLED_SELF: -10,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def update_q_table(
    self,
    old_state,
    new_state,
    reward,
    action_index: int,
    lr: float = 0.7,
    gamma: float = 0.175,
) -> None:
    Q = self.model

    q_old_index = tuple(list(old_state) + [action_index])
    Q[q_old_index] = Q[q_old_index] + lr * (
        reward + gamma * np.max(Q[tuple(list(new_state))]) - Q[q_old_index]
    )

    self.model = Q


def end_round_q_table(
    self,
    old_state,
    reward,
    action_index: int,
    lr: float = 0.8,
) -> None:
    Q = self.model

    q_old_index = tuple(list(old_state) + [action_index])
    Q[q_old_index] = Q[q_old_index] + lr * (reward - Q[q_old_index])

    self.model = Q
