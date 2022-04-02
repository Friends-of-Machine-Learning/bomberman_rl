from ..foml_agent.callbacks import act as act_other
from ..foml_agent.callbacks import setup as setup_other
from ..foml_agent.callbacks import state_to_features as state_to_features_other


def setup(self):
    setup_other(self)


def act(self, state):
    return act_other(self, state)


def state_to_features(self, state):
    return state_to_features_other(self, state)
