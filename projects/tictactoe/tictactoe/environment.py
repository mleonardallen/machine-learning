import time
import random
from simulator import Simulator
import numpy as np
from sys import maxsize

class Environment(object):
    """Environment within which all agents operate."""

    def __init__(self, grid_size=(3,3)):
        
        # Initialize simulation variables
        self.done = False
        self.t = 1

        self.grid_size = grid_size
        self.state = np.zeros(grid_size)
        self.turn = 1
        self.actions = map(tuple, np.transpose(np.where(np.zeros(grid_size) == 0)))

        self.agents = []

    def add_agent(self, **kwargs):
        agent_class = kwargs.get('agent_class')
        assert agent_class is not None, "agent_class is required!"
        agent = agent_class(env=self, **kwargs)
        self.agents.append(agent)

    def reset(self):
        self.state = np.zeros(self.grid_size)
        self.done = False
        self.actions = []
        # self.t = 0

    def get_valid_actions(self):
        empty = np.where(self.state == 0)
        return map(tuple, np.transpose(empty))

    def get_actions(self):
        return self.actions

    def step(self):

        if len(self.get_valid_actions()) == 0:
            return False

        # Update agents
        turn = self.turn
        for agent in self.agents:
            if agent.symbol != turn:
                continue

            agent.update(self.t)

        self.t += 1
        return True

    def sense(self, agent):
        # each agent only sees from their own perspective
        # so no need to have a different state for mirrored placements
        return self.state * agent.symbol + 0


    def won(self, agent):
        haswon = False

        # test row and colum  win conditions
        for i in range(self.grid_size[0]):
            row = np.average(self.state[i])
            col = np.average(self.state[:,i])
            if agent.symbol in [row, col]:
                haswon = True

        # test diagonal win conditions
        diag1 = np.average(np.flipud(self.state).diagonal())
        diag2 = np.average(self.state.diagonal())
        if agent.symbol in [diag1, diag2]:
            haswon = True

        return haswon

    def act(self, agent, action):
        assert agent in self.agents, "Unknown agent!"
        # assert action in self.valid_actions, "Invalid action!"

        self.turn = self.turn * -1
        self.state[action] = agent.symbol
        self.actions.append((action, agent.symbol))

        reward = 0
        if self.won(agent):
            # print "Agent", "X" if agent.symbol == 1 else "O", "Wins!"
            # if agent.symbol == -1:
            #     print 
            #     print self.actions
            agent.wins += 1
            reward = maxsize
            self.done = True

        return reward
