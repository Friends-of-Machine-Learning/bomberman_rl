import pickle
from collections import deque
from collections import namedtuple
from typing import Deque
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
TRANSITION_HISTORY_SIZE = 800  # keep only ... last transitions
END_TRANSITION_HISTORY_SIZE = 200  # keep only ... last transitions
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
    self.custom_events = [
        ev.UselessBombEvent(),
        ev.PlacedGoodBombEvent(),
        ev.AvoidDeathEvent(),
        ev.NewFieldEvent(),
    ]
    self.transitions = []
    self.memory = deque(maxlen=500)


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

    if (
        len(self.transitions) > 2
        and OPPOSITE_DIRECTION[self_action] == self.transitions[-2].action
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

    self.transitions.append(
        Transition(
            last_action,
            state_to_features(self, last_game_state),
            None,
            last_game_state["step"],
            last_game_state["round"],
            reward_from_events(self, events),
        )
    )
    sum = 0
    for trans in self.transitions:
        sum += trans.reward
    with open("game_rewards.txt", "a") as file:
        file.write(f"{sum}\n")

    ############# training #################

    gamma = 0.8
    alpha = 0.0001

    rewards = [x.reward for x in self.transitions]

    rewards = np.array(rewards)
    q_vals = []

    weighting = np.array([1, gamma, gamma ** 2])
    for i in range(len(rewards) - 2):
        q_vals.append(np.sum(rewards[i : i + 3] * weighting))

    q_vals.append(np.sum(rewards[-2:] * weighting[0:2]))
    q_vals.append(rewards[-1])

    for i in range(len(rewards) - 3):
        q_vals[i] += q_func(self.model, self.means, [self.transitions[i + 3].feature])[
            0
        ]

    # Call Q-Function for each action, and its transitions
    transition_for_action = {}
    for transition, q_val in zip(self.transitions, q_vals):
        transition_for_action.setdefault(transition.action, []).append(
            (transition.feature, q_val)
        )

    for action in ACTIONS:
        q_function_train(
            self,
            transition_for_action.get(action, []),
            ACTION_TO_INDEX[action],
            gamma,
            alpha,
        )

    with open("means.txt", "a") as file:
        file.write(" ".join([str(val) for val in self.means]) + "\n")

    self.transitions = []

    if last_game_state["round"] % 10 == 0:

        # Store the model
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump((self.model, self.means), file)


# cache in global space, no need to construct each time
game_rewards = {
    # GOOD
    e.COIN_COLLECTED: 5,
    str(ev.DestroyedAnyCrate()): 2,
    e.MOVED_UP: 0.5,
    e.MOVED_DOWN: 0.5,
    e.MOVED_LEFT: 0.5,
    e.MOVED_RIGHT: 0.5,
    e.BOMB_DROPPED: 0.5,
    e.KILLED_OPPONENT: 10,
    # BAD
    e.GOT_KILLED: -7,
    e.KILLED_SELF: -6,
    e.INVALID_ACTION: -1,
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
    transitions_qvals,
    action_index: int,
    gamma: float,
    alpha: float,
) -> None:
    model = self.model
    if not transitions_qvals:
        return

    # states = np.array([pair[0] for pair in transitions_qvals])
    # q_vals = np.array([pair[1] for pair in transitions_qvals])

    states = [pair[0] for pair in transitions_qvals]
    q_vals = [pair[1] for pair in transitions_qvals]

    mem_states = [
        t.feature for t in self.memory if ACTION_TO_INDEX[t.action] == action_index
    ]
    if len(mem_states) > 0:
        mem_q_vals = np.array(
            [
                t.reward
                for t in self.memory
                if ACTION_TO_INDEX[t.action] == action_index
            ],
            dtype=float,
        )
        mem_q_vals += q_func(self.model, self.means, np.array(mem_states))
        mem_q_vals = [a for a in mem_q_vals]

        states.extend(mem_states)
        q_vals.extend(mem_q_vals)

    states = np.array(states, dtype=float)
    q_vals = np.array(q_vals, dtype=float)
    # if len(states == 1):
    # states = np.array([states])
    # q_vals = np.array([q_vals])

    # self.means[action_index] = (
    #    self.n_mean_instances[action_index] * self.means[action_index] + np.sum(q_vals)
    # ) / (self.n_mean_instances[action_index] + len(q_vals))

    self.means[action_index] += alpha * (np.mean(q_vals) - self.means[action_index])

    # self.n_mean_instances[action_index] += len(q_vals)

    q_vals = q_vals.astype("float64")
    q_vals -= self.means[action_index]

    beta = model[action_index]
    to_remember = np.argmax(np.abs(q_vals - states @ beta))
    if to_remember < len(transitions_qvals):
        self.memory.append(self.transitions[to_remember])

    model[action_index] = beta + alpha * np.mean(
        states * (q_vals - states @ beta)[:, None], axis=0
    )

    self.model = model


def q_func(model: np.ndarray, means, state) -> np.ndarray:
    return np.max(state @ model.T + means, axis=1)
