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
    self.transitions_for_action = {action: [] for action in ACTIONS}
    self.end_transitions_for_action = {action: [] for action in ACTIONS}

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

    self.transitions_for_action[self_action].append(
        Transition(
            self_action,
            state_to_features(self, old_game_state),
            state_to_features(self, new_game_state),
            old_game_state["step"],
            old_game_state["round"],
            reward_from_events(self, events),
        )
    )

    if (
        len(self.transitions) > 3
        and OPPOSITE_DIRECTION.get(self_action, None) == self.transitions[-2].action
    ):
        events.append(BACKTRACK_EVENT)

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

    self.end_transitions_for_action[last_action].append(
        Transition(
            last_action,
            state_to_features(self, last_game_state),
            None,
            last_game_state["step"],
            last_game_state["round"],
            reward_from_events(self, events),
        )
    )

    if last_game_state["round"] % 10 == 0:
        feature_size = sum(f.get_feature_size() for f in self.features_used)
        for tree in self.model:
            tree.fit(np.zeros((1, feature_size)), [0])
        # try to converge the forest to the q function
        for _ in range(10):
            for action in ACTIONS:
                q_function_train(
                    self,
                    self.transitions_for_action[action],
                    self.end_transitions_for_action[action],
                    ACTION_TO_INDEX[action],
                )

        # Store the model
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.model, file)


# cache in global space, no need to construct each time
game_rewards = {
    # GOOD
    e.COIN_COLLECTED: 5,
    e.CRATE_DESTROYED: 1,
    str(ev.AvoidDeathEvent()): 0.5,
    str(ev.PlacedGoodBombEvent()): 0.5,
    str(ev.NewFieldEvent()): 0.5,
    # BAD
    e.KILLED_SELF: -10,
    e.INVALID_ACTION: -1,
    str(ev.UselessBombEvent()): -3,
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
    alpha: float = 0.075,
) -> None:
    model = self.model
    if not transitions:
        return

    rewards = np.array([x.reward for x in transitions])
    states_old = np.array([x.feature for x in transitions])
    states_new = np.array([x.next_feature for x in transitions])
    states_end = np.array([x.feature for x in end_transitions])
    q_vals_end = np.array([x.reward for x in end_transitions])

    q_vals = rewards + gamma * q_func(model, states_new)

    if len(states_end.shape) > 1:
        q_vals = np.concatenate((q_vals, q_vals_end))
        states_old = np.concatenate((states_old, states_end))

    model[action_index].fit(states_old, q_vals)

    self.model = model


def q_func(model, state) -> np.ndarray:
    return np.max([forest.predict(state) for forest in model], axis=0)
