import numpy as np
import math
from agent import LearningAgent, ClickAgent, RandomAgent

params = {
    'learning': {
        'agent_class': LearningAgent,
        'initial_value': 5,
        'epsilon': lambda t: 1./(np.log(t+1) + 1),
        'alpha': lambda t: 1,
        'gamma': lambda t: .95
    },
    'trained': {
        'agent_class': LearningAgent,
        'initial_value': 5,
        'epsilon': lambda t: 0,
        'alpha': lambda t: 0,
        'gamma': lambda t: 0
    },
    'random': {
        'agent_class': RandomAgent,
    },
    'click': {
        'agent_class': ClickAgent
    }
}