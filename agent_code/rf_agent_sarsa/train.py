import pickle
from collections import deque
from collections import namedtuple
from typing import List

import numpy as np
from sklearn.ensemble import RandomForestRegressor

import events as e
from .callbacks import ACTIONS
from .callbacks import state_to_features
from .utils import ACTION_TO_INDEX
from .utils import OPPOSITE_DIRECTION

# This is only an example!
Transition = namedtuple(
    "Transition", ("action", "feature", "next_feature", "steps", "rounds", "reward")
)

SARSA = namedtuple(
    "SARSA", ("old_features", "old_action", "reward", "new_features", "new_action")
)

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 800  # keep only ... last transitions
END_TRANSITION_HISTORY_SIZE = 20  # keep only ... last transitions
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
    self.last_game_transitions = []
    self.last_game_end_transition = None

    self.sarsa_transitions = []
    self.sarsa_end_transitions = []


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

    self.last_game_end_transition = Transition(
        last_action,
        state_to_features(self, last_game_state),
        None,
        last_game_state["step"],
        last_game_state["round"],
        reward_from_events(self, events),
    )

    # create sarsa tuples from the last game
    for i in range(len(self.last_game_transitions) - 1):

        old_features = self.last_game_transitions[i].feature
        old_action = self.last_game_transitions[i].action
        reward = self.last_game_transitions[i].reward
        new_features = self.last_game_transitions[i].next_feature
        new_action = self.last_game_transitions[i + 1].action
        self.sarsa_transitions.append(
            SARSA(old_features, old_action, reward, new_features, new_action)
        )

    # for the last transition
    old_features = self.last_game_transitions[-1].feature
    old_action = self.last_game_transitions[-1].action
    reward = self.last_game_transitions[-1].reward
    new_features = self.last_game_transitions[-1].next_feature
    new_action = self.last_game_end_transition.action
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


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """

    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.CRATE_DESTROYED: 0.5,
        # RUNAWAY_EVENT: 0.5,
        e.KILLED_SELF: -7,
        e.BOMB_DROPPED: 0.5,
        e.INVALID_ACTION: -0.1,
        # e.WAITED: -0.2,
        BACKTRACK_EVENT: -0.0,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def q_function_train(
    self,
    sarsa_transitions: List[SARSA],
    sarsa_end_transitions: List[SARSA],
    action_index: int,
    gamma: float = 0.7,
    alpha: float = 0.075,
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


def q_func(model, state) -> np.ndarray:
    return np.max([forest.predict(state) for forest in model], axis=0)


def q_func(model, state, action_indices) -> np.ndarray:

    res = np.zeros_like(action_indices)

    for act_idx in range(len(ACTIONS)):
        action_mask = action_indices == act_idx

        if np.any(action_mask):
            res[action_mask] = model[act_idx].predict(state[action_mask])

    return res
