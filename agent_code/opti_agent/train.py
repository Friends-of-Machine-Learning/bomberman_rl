import pickle

from ..qtable_agent.train import end_of_round as end_of_round_foml
from ..qtable_agent.train import game_events_occurred as game_events_occurred_foml
from ..qtable_agent.train import reward_from_events as reward_from_events_foml
from ..qtable_agent.train import setup_training as setup_training_foml

TOTAL_ROUND_REWARD = 0
BEST_ROUND_REWARD = -10e10


def setup_training(self):
    setup_training_foml(self)


def game_events_occurred(self, old_game_state, old_action, new_game_state, events):
    game_events_occurred_foml(self, old_game_state, old_action, new_game_state, events)
    reward_from_events(self, events)


def end_of_round(self, last_game_state, last_action, events):
    global TOTAL_ROUND_REWARD
    global BEST_ROUND_REWARD
    end_of_round_foml(self, last_game_state, last_action, events)
    reward_from_events(self, events)

    if TOTAL_ROUND_REWARD > BEST_ROUND_REWARD:
        print(f"Found Better Model: {TOTAL_ROUND_REWARD} > {BEST_ROUND_REWARD}")
        BEST_ROUND_REWARD = TOTAL_ROUND_REWARD

        # Store the model
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.model, file)
    elif last_game_state["round"] % 100 == 0:
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

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
