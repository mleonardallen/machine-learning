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

        # TODO: Initialize any additional variables here
        self.q = {} # dictionary to hold all of the q^ values
        self.alpha = 0.5 # learning rate
        self.gamma = 0.9 # discount factor
        self.epsilon = 0.0 # exploration rate
        self.initial_value = 10 # initial value for Q^
        self.t = 0 # interations
        self.actions = [None, 'forward', 'right', 'left'] # available actions

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self.getState()

        # TODO: Select action according to your policy
        action = self.getPolicy(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        self.updateQ_sa(self.state, action, reward, self.getState())
        self.t += 1
        
        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def getState(self):
        next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        return "-".join([
            next_waypoint,
            inputs['light'],
            inputs['oncoming'] if inputs['oncoming'] else 'None',
            inputs['left'] if inputs['left'] else 'None',
            inputs['right'] if inputs['right'] else 'None'
        ])

    def getPolicy(self, state):
        q_s = self.getQ_s(state)
        argmax = max(q_s.iteritems(), key=operator.itemgetter(1))[0]

        # take optimal action with some exploration
        return np.random.choice([
            argmax,
            random.choice(self.actions)
        ], p=[1-self.epsilon, self.epsilon])

    # Q(s,a) <- (alpha) reward + omega * max a Q(s',a')
    def updateQ_sa(self, state, action, reward, state_prime):

        # q^(s,a) = (1 - alpha) * q^(s,a) + alpha * reward
        immediate = (1 - self.alpha) * self.getQ_sa(state, action) + self.alpha * reward

        # omega ^ t
        discountedRate = math.pow(self.gamma, self.t)
        
        # max a Q^(s', a')
        q_prime = max(self.getQ_s(state_prime).values())

        # update q^
        newValue = immediate + discountedRate * q_prime
        self.setQ_sa(state, action, newValue)

    # Q(s,a)
    def setQ_sa(self, state, action, value):
        if state not in self.q:
            self.q[state] = {}

        self.q[state][action] = value

    # Q(s,a)
    def getQ_sa(self, state, action):
        return self.q[state][action] if state in self.q and action in self.q[state] else self.initial_value

    # Q(s)
    def getQ_s(self, state):
        qs = {}
        for action in self.actions:
            qs[action] = self.getQ_sa(state, action)

        return qs

    def stringify(self, state):
        return "-".join(list(state.values()))


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
