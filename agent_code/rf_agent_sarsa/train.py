import pickle
from collections import namedtuple
from types import SimpleNamespace
from typing import List

import numpy as np
from sklearn.ensemble import RandomForestRegressor

import events as e
from .callbacks import ACTIONS
from .callbacks import state_to_features
from .utils import ACTION_TO_INDEX

# This is only an example!
Transition = namedtuple(
    "Transition", ("action", "feature", "next_feature", "steps", "rounds", "reward")
)

SARSA = namedtuple(
    "SARSA", ("old_features", "old_action", "reward", "new_features", "new_action")
)

# Custom Events
BACKTRACK_EVENT = "BACKTRACK"
RUNAWAY_EVENT = "RUNAWAY"


def setup_training(self: SimpleNamespace) -> None:
    """
    Initialise self for training purpose.
    This is called after `setup` in callbacks.py.

    :param self: This SimpleNamespace Object is passed to all callbacks and you can set arbitrary values.
    """

    # Used to keep track of Transitions on per Round basis to create SARSA Pairs. Will be cleared after each Round.
    self.last_game_transitions = []
    # Special Case, only needed for last Transition on Round.
    self.last_game_end_transition = None

    # Stores all SARSA Transitions.
    self.sarsa_transitions = []
    # Stores all SARSA Transitions from end Round Transitions
    self.sarsa_end_transitions = []


def game_events_occurred(
    self: SimpleNamespace,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
) -> None:
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events, and your knowledge of the (new) game state.

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

    # if (
    #    len(self.sarsa_transitions) > 2
    #    and OPPOSITE_DIRECTION[self_action] == self.sarsa_transitions[-2].action
    # ):
    #    events.append(BACKTRACK_EVENT)

    # Compare Bomb distance to agent from old and new game state
    # If agent moved away from bomb add event
    if old_game_state["bombs"] and new_game_state["bombs"]:
        old_bombs = [(x, y) for (x, y), _ in old_game_state["bombs"]]
        con_old = np.subtract(old_bombs, old_game_state["self"][3])
        dist_old = np.linalg.norm(con_old, axis=1)
        old_mean = np.mean(dist_old)

        new_bombs = [(x, y) for (x, y), _ in new_game_state["bombs"]]
        con_new = np.subtract(new_bombs, new_game_state["self"][3])
        dist_new = np.linalg.norm(con_new, axis=1)
        new_mean = np.mean(dist_new)

        if new_mean > old_mean:
            events.append(RUNAWAY_EVENT)

    self.last_game_transitions.append(
        Transition(
            self_action,
            state_to_features(self, old_game_state),
            state_to_features(self, new_game_state),
            old_game_state["step"],
            old_game_state["round"],
            reward_from_events(self, events),
        )
    )


def end_of_round(
    self: SimpleNamespace, last_game_state: dict, last_action: str, events: List[str]
) -> None:
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    :param last_game_state: The last game_state of this round.
    :param last_action: The last action in this round.
    :param events: All events in this round.
    """
    self.logger.debug(
        f'Encountered event(s) {", ".join(map(repr, events))} in final step'
    )

    self.last_game_end_transition = Transition(
        last_action,
        state_to_features(self, last_game_state),
        None,
        last_game_state["step"],
        last_game_state["round"],
        reward_from_events(self, events),
    )

    # Add the last game_end_transition to include last transition
    self.last_game_transitions.append(self.last_game_end_transition)
    # create SARSA tuples from the last game
    for actio, reactio in zip(
        self.last_game_transitions[:-1], self.last_game_transitions[1:]
    ):
        old_features = actio.feature
        old_action = actio.action
        reward = actio.reward
        new_features = actio.next_feature
        new_action = reactio.action
        self.sarsa_transitions.append(
            SARSA(old_features, old_action, reward, new_features, new_action)
        )

    # for the end
    old_features = self.last_game_end_transition.feature
    old_action = self.last_game_end_transition.action
    reward = self.last_game_end_transition.reward
    new_features = None
    new_action = None
    self.sarsa_end_transitions.append(
        SARSA(old_features, old_action, reward, new_features, new_action)
    )

    self.last_game_transitions = []

    # Call Q-Function for each action, and its transitions
    sarsa_end_for_action = {}
    for sarsa in self.sarsa_end_transitions:
        sarsa_end_for_action.setdefault(sarsa.old_action, []).append(sarsa)

    sarsa_for_action = {}
    for sarsa in self.sarsa_transitions:
        sarsa_for_action.setdefault(sarsa.old_action, []).append(sarsa)

    for action in ACTIONS:
        q_function_train(
            self,
            sarsa_for_action.get(action, []),
            sarsa_end_for_action.get(action, []),
            ACTION_TO_INDEX[action],
        )

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self: SimpleNamespace, events: List[str]) -> int:
    """
    En/discourage certain behavior of the agent by giving rewards for certain events.
    """

    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.CRATE_DESTROYED: 0.5,
        e.BOMB_DROPPED: 0.5,
        e.KILLED_SELF: -4,
        e.INVALID_ACTION: -2,
        e.WAITED: -0.1,
        e.MOVED_DOWN: -0.05,
        e.MOVED_UP: -0.05,
        e.MOVED_LEFT: -0.05,
        e.MOVED_RIGHT: -0.05,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def q_function_train(
    self: SimpleNamespace,
    sarsa_transitions: List[SARSA],
    sarsa_end_transitions: List[SARSA],
    action_index: int,
    gamma: float = 0.7,
) -> None:
    model = self.model
    if not sarsa_transitions:
        return

    rewards = np.array([x.reward for x in sarsa_transitions])
    states_old = np.array([x.old_features for x in sarsa_transitions])
    states_new = np.array([x.new_features for x in sarsa_transitions])
    action_indices = np.array(
        [ACTION_TO_INDEX[x.new_action] for x in sarsa_transitions]
    )

    states_end = np.array([x.old_features for x in sarsa_end_transitions])
    q_vals_end = np.array([x.reward for x in sarsa_end_transitions])

    q_vals = rewards + gamma * q_func(model, states_new, action_indices)

    if len(states_end.shape) > 1:
        q_vals = np.concatenate((q_vals, q_vals_end))
        states_old = np.concatenate((states_old, states_end))

    model[action_index].fit(states_old, q_vals)

    self.model = model


def q_func(
    model: List[RandomForestRegressor], state: np.ndarray, action_indices: np.ndarray
) -> np.ndarray:
    res = np.zeros_like(action_indices)

    for act_idx in np.unique(action_indices):
        action_mask = action_indices == act_idx
        res[action_mask] = model[act_idx].predict(state[action_mask])

    return res
