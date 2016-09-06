import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pygame

import math
import operator
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # Initialize any additional variables here
        self.actions = [None, 'forward', 'right', 'left'] # available actions

        self.q_table = {} # table to hold all of the q^ values
        self.total_t = 0 # use in place of t

        # model parameters
        self.params = {
            'initial_value': 1,
            'epsilon': lambda t: 1./(t+1),
            'alpha': lambda t: 1. / (.8*np.log(t+2)),
            'gamma': lambda t: math.pow(0.9, t)
        }

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        deadline = self.env.get_deadline(self)
        inputs = self.env.sense(self)

        # Update state
        self.state = self.getState()

        # Select action according to your policy
        action = self.getPolicy(self.state, self.total_t)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Learn policy based on state, action, reward
        self.updateQ_sa(self.state, action, reward, self.getState(), self.total_t)

        self.total_t += 1

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def getState(self):
        next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)

        # Update state
        # Note: Using tuple so that we have a hashable value for the q table.
        return (next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'])

    def getPolicy(self, state, t):
        q_s = self.getQ_s(state)
        argmax = max(q_s.iteritems(), key=operator.itemgetter(1))[0]
        epsilon = self.params['epsilon'](t)

        # if all values are the same, make a random choice
        if min(q_s.values()) == max(q_s.values()):
            return random.choice(self.actions)

        # take optimal action with some exploration
        return np.random.choice([
            argmax,
            random.choice(self.actions)
        ], p=[1-epsilon, epsilon])

    # Q(s,a) <- (alpha) reward + omega * max a Q(s',a')
    def updateQ_sa(self, state, action, reward, state_prime, t):

        alpha = self.params['alpha'](t) # learning rate
        gamma = self.params['gamma'](t) # discount factor

        q_sa = self.getQ_sa(state, action) # current q^
        q_prime = max(self.getQ_s(state_prime).values()) # next state q^

        # Q(s,a) <- (alpha) reward + omega * max a Q(s',a')
        q_sa = (1 - alpha) * q_sa + alpha * (reward + gamma * q_prime)

        # update q^
        self.setQ_sa(state, action, q_sa)

    # set Q(s,a)
    def setQ_sa(self, state, action, value):
        if state not in self.q_table:
            self.q_table[state] = {}

        self.q_table[state][action] = value

    # get Q(s,a)
    def getQ_sa(self, state, action):
        return self.q_table[state][action] if state in self.q_table and action in self.q_table[state] else self.params['initial_value']

    # get Q(s)
    def getQ_s(self, state):
        qs = {}
        for action in self.actions:
            qs[action] = self.getQ_sa(state, action)

        return qs

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment(num_dummies=3)  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent

    # a.params = i
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials

    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

if __name__ == '__main__':
    run()
