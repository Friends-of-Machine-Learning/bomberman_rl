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

# This is only an example!
Transition = namedtuple(
    "Transition", ("action", "feature", "next_feature", "steps", "rounds", "reward")
)

# Hyper parameters
TRANSITION_HISTORY_SIZE = 400 * 3  # keep only ... last transitions
END_TRANSITION_HISTORY_SIZE = (
    TRANSITION_HISTORY_SIZE // 400
)  # keep only ... last transitions


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.custom_events = [
        ev.UselessBombEvent(),
        ev.PlacedGoodBombEvent(),
        ev.AvoidDeathEvent(),
        ev.NewFieldEvent(),
        ev.DestroyedAnyCrate(),
        ev.SmartBombEvent(),
        ev.FollowOmegaEvent(),
    ]
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.end_transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


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

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(
        f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}'
    )

    if old_game_state is None or new_game_state is None:
        return
    for custom_event in self.custom_events:
        custom_event.game_events_occurred(
            old_game_state, self_action, new_game_state, events
        )
    # state_to_features is defined in callbacks.py
    self.transitions.append(
        Transition(
            self_action,
            state_to_features(self, old_game_state),
            state_to_features(self, new_game_state),
            old_game_state["step"],
            old_game_state["round"],
            reward_from_events(self, events),
        )
    )


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

    self.end_transitions.append(
        Transition(
            last_action,
            state_to_features(self, last_game_state),
            None,
            last_game_state["step"],
            last_game_state["round"],
            reward_from_events(self, events),
        )
    )

    # Call Q-Function for each action, and its transitions
    end_transition_for_action = {}
    for transition in self.end_transitions:
        end_transition_for_action.setdefault(transition.action, []).append(transition)
    transition_for_action = {}
    for transition in self.transitions:
        transition_for_action.setdefault(transition.action, []).append(transition)

    for action in ACTIONS:
        q_function_train(
            self,
            transition_for_action.get(action, []),
            end_transition_for_action.get(action, []),
            ACTION_TO_INDEX[action],
        )

    if last_game_state["round"] % 20 == 0:
        # Store the model
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump((self.model, self.means), file)


# cache in global space, no need to construct each time
game_rewards = {
    # GOOD
    e.COIN_COLLECTED: 5,
    e.CRATE_DESTROYED: 2,
    e.MOVED_UP: 0.5,
    e.MOVED_DOWN: 0.5,
    e.MOVED_LEFT: 0.5,
    e.MOVED_RIGHT: 0.5,
    e.BOMB_DROPPED: 0.5,
    # BAD
    e.KILLED_SELF: -6,
    e.INVALID_ACTION: -1,
    e.WAITED: -0.1,
}


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    global game_rewards

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def q_function_train(
    self,
    transitions: List[Transition],
    end_transitions: List[Transition],
    action_index: int,
    gamma: float = 0.9,
    alpha: float = 0.1,
) -> None:
    model = self.model
    if not transitions:
        return

    rewards = np.array([x.reward for x in transitions])
    states_old = np.array([x.feature for x in transitions])
    states_new = np.array([x.next_feature for x in transitions])
    states_end = np.array([x.feature for x in end_transitions])
    q_vals_end = np.array([x.reward for x in end_transitions])

    q_vals = rewards + gamma * q_func(model, states_new, self.means)

    if len(states_end.shape) > 1:
        q_vals = np.concatenate((q_vals, q_vals_end))
        states_old = np.concatenate((states_old, states_end))

    self.means[action_index] = (
        self.n_mean_instances[action_index] * self.means[action_index] + np.sum(q_vals)
    ) / (self.n_mean_instances[action_index] + len(q_vals))

    self.n_mean_instances[action_index] += len(q_vals)

    q_vals -= self.means[action_index]

    beta = model[action_index]

    model[action_index] = beta + alpha * np.mean(
        states_old * (q_vals - states_old @ beta)[:, None], axis=0
    )

    self.model = model


def q_func(model: np.ndarray, state, means) -> np.ndarray:
    return np.max(state @ model.T + means, axis=1)
