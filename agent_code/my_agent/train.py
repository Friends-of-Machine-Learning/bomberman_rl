import pickle
from collections import deque
from collections import namedtuple
from typing import List

import numpy as np
from sklearn.linear_model import LinearRegression

import events as e
from .callbacks import ACTION_TO_INDEX
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple(
    "Transition", ("action", "feature", "next_feature", "steps", "rounds", "reward")
)

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1000000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.LR = LinearRegression()


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

    # Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)

    if old_game_state is None or new_game_state is None:
        return
    # state_to_features is defined in callbacks.py
    self.transitions.append(
        Transition(
            self_action,
            state_to_features(old_game_state),
            state_to_features(new_game_state),
            old_game_state["step"] if old_game_state else 0,
            old_game_state["round"] if old_game_state else 0,
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

    transition_for_action = {}
    for transition in self.transitions:
        transition_for_action.setdefault(transition.action, []).append(transition)
    for action, trans in transition_for_action.items():
        self.model = q_function_train(self.model, trans, ACTION_TO_INDEX[action])

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
        e.COIN_COLLECTED: 1,
        e.KILLED_SELF: -5,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def q_function_train(
    model: np.ndarray,
    transitions: List[Transition],
    action_index: int,
    gamma: float = 0.8,
    alpha: float = 0.1,
) -> np.ndarray:
    rewards = np.array([x.reward for x in transitions])
    states_old = np.array([x.feature for x in transitions])

    states_new = np.array([x.next_feature for x in transitions])

    q_vals = rewards + gamma * q_func(model, states_new)
    q_vals = q_vals - np.mean(
        q_vals
    )  # ToDo: Use these means to determine the self.means

    beta = model[action_index]
    model[action_index] = beta + alpha * np.mean(
        states_old * (q_vals - states_old @ beta)[:, None], axis=0
    )

    return model


def q_func(model: np.ndarray, state) -> np.ndarray:
    return np.max(state @ model.T, axis=1)
