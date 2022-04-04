import os

import numpy as np

from ..qtable_agent.train import end_of_round as end_of_round_other
from ..qtable_agent.train import game_events_occurred as game_events_occurred_other
from ..qtable_agent.train import reward_from_events as reward_from_events_other
from ..qtable_agent.train import setup_training as setup_training_other

TOTAL_ROUND_REWARD = 0
AGENT = "qtable_agent_good_less_death_reward"


def setup_training(self):
    setup_training_other(self)

    # create folder AGENT if it does not yet exist.
    if not os.path.exists(AGENT):
        os.makedirs(AGENT)

    with open(f"{AGENT}/qtable_sparseness.txt", "w") as myfile:
        myfile.write("")
    with open(f"{AGENT}/game_rewards.txt", "w") as myfile:
        myfile.write("")


def game_events_occurred(self, old_game_state, old_action, new_game_state, events):
    game_events_occurred_other(self, old_game_state, old_action, new_game_state, events)
    reward_from_events(self, events)


def end_of_round(self, last_game_state, last_action, events):
    global NEW_ROUND
    global TOTAL_ROUND_REWARD
    end_of_round_other(self, last_game_state, last_action, events)
    reward_from_events(self, events)

    with open(f"{AGENT}/qtable_sparseness.txt", "a") as myfile:
        qtable: np.ndarray = self.model
        zeroes = np.count_nonzero(qtable == 0)
        others = np.count_nonzero(qtable != 0)

        myfile.write(f"{zeroes} {others}\n")

    with open(f"{AGENT}/game_rewards.txt", "a") as myfile:
        myfile.write(str(TOTAL_ROUND_REWARD) + "\n")

    # reset the TOTAL_ROUND_REWARD to 0
    TOTAL_ROUND_REWARD = 0


def reward_from_events(self, events):
    global TOTAL_ROUND_REWARD
    reward_for_step = reward_from_events_other(self, events)
    TOTAL_ROUND_REWARD += reward_for_step
