import time

from ..foml_agent.train import end_of_round as end_of_round_foml
from ..foml_agent.train import game_events_occurred as game_events_occurred_foml
from ..foml_agent.train import reward_from_events as reward_from_events_foml
from ..foml_agent.train import setup_training as setup_training_foml

TOTAL_ROUND_REWARD = 0
FEATURE_SIZE = 0


def setup_training(self):
    global FEATURE_SIZE
    setup_training_foml(self)
    self.total_round_time = 0
    self.average_round_time = 0

    FEATURE_SIZE = sum(f.get_feature_size() for f in self.features_used)
    # create empty file called FEATURE_SIZE.txt
    with open(str(FEATURE_SIZE) + ".txt", "w") as myfile:
        myfile.write("")


def game_events_occurred(self, old_game_state, old_action, new_game_state, events):
    game_events_occurred_foml(self, old_game_state, old_action, new_game_state, events)


def end_of_round(self, last_game_state, last_action, events):
    global NEW_ROUND
    global TOTAL_ROUND_REWARD
    end_of_round_foml(self, last_game_state, last_action, events)
    reward_from_events(self, events)

    # time how long each round takes
    if last_game_state["round"] == 1:
        self.round_start_time = time.time()
        self.round_end_time = time.time()
    else:
        self.round_start_time = self.round_end_time
        self.round_end_time = time.time()

    # update total round time
    self.total_round_time += self.round_end_time - self.round_start_time

    # update average round time
    self.average_round_time = self.total_round_time / last_game_state["round"]

    # open file called FEATURE_SIZE.txt and add TOTAL_ROUND_REWARD | round time.
    with open(str(FEATURE_SIZE) + ".txt", "a") as myfile:
        myfile.write(
            str(TOTAL_ROUND_REWARD) + " | " + str(self.average_round_time) + "\n"
        )

    # reset the TOTAL_ROUND_REWARD to 0
    TOTAL_ROUND_REWARD = 0


def reward_from_events(self, events):
    """Sum all rewards from each step, reset at each end round.

    Parameters
    ----------
    events : List
        List of all events that occurred during the step.
    """

    global TOTAL_ROUND_REWARD
    reward_for_step = reward_from_events_foml(self, events)
    TOTAL_ROUND_REWARD += reward_for_step
