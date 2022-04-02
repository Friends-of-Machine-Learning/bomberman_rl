import pickle

from ..qtable_agent.train import end_of_round as end_of_round_other
from ..qtable_agent.train import game_events_occurred as game_events_occurred_other
from ..qtable_agent.train import reward_from_events as reward_from_events_other
from ..qtable_agent.train import setup_training as setup_training_other

# Store the agents round rewards for comparison between rounds
TOTAL_ROUND_REWARD = 0
BEST_ROUND_REWARD = -10e10


def setup_training(self):
    # Setup the other for training.
    setup_training_other(self)


def game_events_occurred(self, old_game_state, old_action, new_game_state, events):
    # Let the other agent know that a game event has occurred.
    game_events_occurred_other(self, old_game_state, old_action, new_game_state, events)

    # Calculate the rewards for the step and add it to the total reward for this round.
    reward_from_events(self, events)


def end_of_round(self, last_game_state, last_action, events):
    global TOTAL_ROUND_REWARD
    global BEST_ROUND_REWARD

    # Let the other agent know that the round has ended.
    end_of_round_other(self, last_game_state, last_action, events)
    # Update the round reward.
    reward_from_events(self, events)

    # Check if the round reward is the best reward so far.
    if TOTAL_ROUND_REWARD > BEST_ROUND_REWARD:
        print(f"Found Better Model: {TOTAL_ROUND_REWARD} > {BEST_ROUND_REWARD}")
        BEST_ROUND_REWARD = TOTAL_ROUND_REWARD

        # Store the model
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.model, file)
    # Rollback to the best model every 100 rounds.
    elif last_game_state["round"] % 100 == 0:
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

    # Reset the TOTAL_ROUND_REWARD to 0.
    TOTAL_ROUND_REWARD = 0


def reward_from_events(self, events):
    global TOTAL_ROUND_REWARD
    # Calculate the rewards for the step and add it to the total reward for this round.
    reward_for_step = reward_from_events_other(self, events)
    TOTAL_ROUND_REWARD += reward_for_step
