from environment import Environment
from simulator import Simulator
from pydispatch import dispatcher
import pandas
import optparse

def run():
    """Run the agent for a finite number of trials."""

    options = parseOptions()

    env = Environment()  # create environment (also adds some dummy traffic)
    sim = Simulator(env, update_delay=0, display=options.display) # create simulator (uses pygame when display=True, if available)

    results = {}

    from settings import params
    for agent, symbol in [(options.player1, 1), (options.player2, -1)]:
        kwargs = params[agent]
        env.add_agent(
            symbol=symbol, 
            file=options.file, 
            clear=options.clear,
            save=options.save,
            **kwargs)

    sim.run(n_trials=options.iterations)  # run for a specified number of trials

    for agent in env.agents:
        results["X" if agent.symbol == 1 else 'O'] = agent.wins

    print results

    dispatcher.send(signal='main.complete', sender={})

    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

def parseOptions():
    optParser = optparse.OptionParser()

    optParser.add_option('-i', '--iterations',action='store',
        type='int', dest='iterations', default=10,
        metavar="K", help='Number of rounds of value iteration (default %default)')

    optParser.add_option('-d', '--display', action='store_true',
        dest='display', default=False,
        help='Use GUI display')

    optParser.add_option('-a', '--player1', action='store', metavar="A",
        dest='player1', default="learning",
        choices=('random', 'learning', 'click', 'trained'),
        help="Agent type (options are 'random', 'learning', 'trained' and 'click', default %default)")

    optParser.add_option('-b', '--player2', action='store', metavar="B",
        dest='player2', default="random",
        choices=('random', 'learning', 'click', 'trained'),
        help="Agent type (options are 'random', 'learning', 'trained' and 'click', default %default)")

    optParser.add_option('-c', '--clear', action='store_true',
        dest='clear', default=False,
        help='Clear agent memory')

    optParser.add_option('-f', '--file', action='store',
        dest='file', default='qtable.pkl',
        help='File to load for agents memory')

    optParser.add_option('-s', '--save', action='store_true',
        dest='save', default=False,
        help='Save trial results to memory')

    opts, args = optParser.parse_args()

    # MANAGE CONFLICTS
    if 'click' in [opts.player1, opts.player2]:
        opts.display = True

    from settings import params
    

    return opts

if __name__ == '__main__':
    run()
