import os

from ..foml_agent.train import end_of_round as end_of_round_other
from ..foml_agent.train import game_events_occurred as game_events_occurred_other
from ..foml_agent.train import reward_from_events as reward_from_events_other
from ..foml_agent.train import setup_training as setup_training_other

TOTAL_ROUND_REWARD = 0
AGENT = "foml_agent"


def setup_training(self):
    setup_training_other(self)

    # create folder AGENT if it does not yet exist.
    if not os.path.exists(AGENT):
        os.makedirs(AGENT)

    with open(f"{AGENT}/means.txt", "w") as myfile:
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

    with open(f"{AGENT}/means.txt", "a") as myfile:
        myfile.write(" ".join([str(val) for val in self.means]) + "\n")

    with open(f"{AGENT}/game_rewards.txt", "a") as myfile:
        myfile.write(str(TOTAL_ROUND_REWARD) + "\n")

    # reset the TOTAL_ROUND_REWARD to 0
    TOTAL_ROUND_REWARD = 0


def reward_from_events(self, events):
    global TOTAL_ROUND_REWARD
    reward_for_step = reward_from_events_other(self, events)
    TOTAL_ROUND_REWARD += reward_for_step
