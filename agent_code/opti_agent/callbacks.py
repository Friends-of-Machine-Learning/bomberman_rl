"""
This agent is used as an alternative agent runner for other agents.
It it used to check the performance of the agent each round and determine wether or not to update the saved model.
This allows the agent to rollback to a save state any time it starts to loose performance.
To use it for any agent simply change the imported agent to the agent you want to optimize.
"""
from ..qtable_agent.callbacks import act as act_other
from ..qtable_agent.callbacks import setup as setup_other
from ..qtable_agent.callbacks import state_to_features as state_to_features_other


def setup(self):
    setup_other(self)


def act(self, state):
    return act_other(self, state)


def state_to_features(self, state):
    return state_to_features_other(self, state)
