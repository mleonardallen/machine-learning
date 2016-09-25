import random
import math
import operator
import numpy as np
import pickle
import os
from pydispatch import dispatcher

class Agent(object):
    """Base class for all agents."""

    def __init__(self, *args, **kwargs):

        self.wins = 0
        self.env = kwargs.get('env', None)
        self.symbol = kwargs.get('symbol', None)
        assert self.env is not None, "Enviroment Required!"

    def reset(self):
        pass

    def update(self, t):
        pass

    def get_state(self, **kwargs):
        mirrored = kwargs.get('mirrored', False)
        state = self.env.sense(self)

        if mirrored:
            state = -state + 0 # multiply by -1 gives apponents perspective

        return tuple(state.flatten())

class RandomAgent(Agent):

    def __init__(self, *args, **kwargs):
        super(RandomAgent, self).__init__(*args, **kwargs)

    def update(self, t):
        actions = self.env.get_valid_actions()
        idx = np.random.choice(len(actions))
        self.env.act(self, actions[idx])

class ClickAgent(Agent):

    def __init__(self, *args, **kwargs):
        super(ClickAgent, self).__init__(*args, **kwargs)
        dispatcher.connect(self.handle_event, signal='board.click', sender=dispatcher.Any)
        self.action = None

    def handle_event(self, sender, action ):
        if self.env.turn != self.symbol:
            return

        actions = self.env.get_valid_actions()
        if action not in actions:
            return

        self.action = action


    def update(self, t):
        if self.action == None:
            return

        reward = self.env.act(self, self.action)
        self.action = None

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    # table to hold all of the q^ values
    # static variable so agents can share
    

    def __init__(self, *args, **kwargs):
        super(LearningAgent, self).__init__(*args, **kwargs)

        self.file = kwargs.get('file', 'qtable.pkl')

        if kwargs.get('clear', False) is True:
            self.clear_qtable()

        if kwargs.get('save', False) is True:
            dispatcher.connect(self.save_qtable, signal='main.complete', sender=dispatcher.Any)

        self.qtable = None
        self.initial_value = kwargs.get('initial_value', 1) # initial q(s,a)
        self.epsilon = kwargs.get('epsilon', lambda t: 1./(t+1)) # exploration
        self.alpha = kwargs.get('alpha', lambda t: 1. / (.8*np.log(t+2))) # learning rate
        self.gamma = kwargs.get('gamma', lambda t: math.pow(0.9, t)) # discount factor


    def clear_qtable(self):
        if os.path.isfile(self.file):
            os.remove(self.file)

    def save_qtable(self):
        with open(self.file, 'wb') as f:
            pickle.dump(self.qtable, f, pickle.HIGHEST_PROTOCOL)

    def load_qtable(self):
        try:
            with open(self.file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            return {}

    def get_qtable(self):
        if self.qtable is not None:
            return self.qtable

        self.qtable = self.load_qtable()
        return self.qtable


    def update(self, t):

        if len(self.env.get_valid_actions()) == 0:
            return

        # Update state
        state = self.get_state()

        # Select action according to your policy
        action = self.getAction(state, t)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Learn policy based on state, action, reward
        self.updateQ_sa(state, action, reward, self.get_state(mirrored=True), t)

    def getAction(self, state, t):
        epsilon = self.epsilon(t)
        explore = np.random.choice([False, True], p=[1-epsilon, epsilon])

        if explore:
            actions = self.env.get_valid_actions()
            idx = np.random.choice(len(actions))
            return actions[idx]

        return self.getPolicy(state)

    def getPolicy(self, state):
        """get optimal action"""
        q_s = self.getQ_s(state)
        actions = [action for action in q_s.keys() if q_s[action] == max(q_s.values())]
        idx = np.random.choice(len(actions))
        return actions[idx]

    def updateQ_sa(self, state, action, reward, state_prime, t):
        """Q(s,a) <- (alpha) reward + omega * max a Q(s',a')"""

        alpha = self.alpha(t) # learning rate
        gamma = self.gamma(t) # discount factor
        q_sa = self.getQ_sa(state, action) # current q^

        # Important: Bellman's equation is modified to account for zero-sum game.
        # In a zero sum game reward is opposite of opponents reward
        # The next move is the apponents, so we need to see that move from the apponents perspective

        q_prime_values = self.getQ_s(state_prime).values() # get q values from apponents perspective
        q_prime = max(q_prime_values) if len(q_prime_values) > 0 else 0
        q_prime = -q_prime + 0

        # Q(s,a) <- (alpha) reward + omega * max a Q(s',a')
        q_sa = (1 - alpha) * q_sa + alpha * (reward + gamma * q_prime)

        # update q^
        self.setQ_sa(state, action, q_sa)

    def getQ_sa(self, state, action):
        """get Q(s,a)"""
        return self.get_qtable().get((state, action), self.initial_value)

    def setQ_sa(self, state, action, value):
        """set Q(s,a)"""
        self.get_qtable()[(state, action)] = value

    def getQ_s(self, state):
        return {action: self.getQ_sa(state, action) for action in self.env.get_valid_actions()}
