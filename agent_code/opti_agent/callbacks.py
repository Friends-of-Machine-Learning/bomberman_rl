from ..qtable_agent.callbacks import act as act_foml
from ..qtable_agent.callbacks import setup as setup_foml
from ..qtable_agent.callbacks import state_to_features as state_to_features_foml


def setup(self):
    setup_foml(self)


def act(self, state):
    return act_foml(self, state)


def state_to_features(self, state):
    return state_to_features_foml(self, state)
