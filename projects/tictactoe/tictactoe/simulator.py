import os
import time
import random
from board import Board

class Simulator(object):
    """Simulates agents in a dynamic smartcab environment.

    Uses PyGame to display GUI, if available.
    """

    def __init__(self, env, size=None, update_delay=1.0, display=True):
        self.env = env

        self.display = display

        self.quit = False
        self.start_time = None
        self.current_time = 0.0
        self.last_updated = 0.0
        self.update_delay = update_delay  # duration between each step (in secs)

        if self.display:
            try:
                self.board = Board(grid_size=self.env.grid_size)
                self.paused = False
            except ImportError as e:
                self.display = False
                print "Simulator.__init__(): Unable to import pygame; display disabled.\n{}: {}".format(e.__class__.__name__, e)
            except Exception as e:
                self.display = False
                print "Simulator.__init__(): Error initializing GUI objects; display disabled.\n{}: {}".format(e.__class__.__name__, e)

    def run(self, n_trials=1):
        self.quit = False
        for trial in xrange(n_trials):
            # print "Simulator.run(): Trial {}".format(trial)  # [debug]
            self.env.reset()
            self.current_time = 0.0
            self.last_updated = 0.0
            self.start_time = time.time()

            while True:
                try:
                    # Update current time
                    self.current_time = time.time() - self.start_time
                    # print "Simulator.run(): current_time = {:.3f}".format(self.current_time)

                    # Update environment
                    if self.current_time - self.last_updated >= self.update_delay:

                        self.last_updated = self.current_time

                        if (self.env.step() is False):
                            self.env.reset()
                            break;

                    # Render GUI and sleep
                    if self.display:
                        self.board.render(self.env.state)

                except KeyboardInterrupt:
                    self.quit = True
                finally:
                    if self.quit or self.env.done:
                        break

            if self.quit:
                break
