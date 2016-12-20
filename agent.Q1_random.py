import random
import sys
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import matplotlib.pyplot as plt

class LearningAgent(Agent):
    """
    An agent that learns to drive in the smartcab world.
    This is the implementation of Q1, choosing actions randomly while successfully simulating.
    """
    @property
    def trials(self):
        return self._trials
        
    @trials.setter
    def trials(self, value):
        self._trials = value

    @property
    def reward_history(self):
        return self._reward_history
        
    @reward_history.setter
    def reward_history(self, value):
        try:
            getattr(self, 'reward_history')         
        except AttributeError:
            self._reward_history = []

        self._reward_history.append(value)

    @property
    def remaining(self):
        return self._remaining

    @remaining.setter
    def remaining(self, value):
        self._remaining = value

    @property
    def reached_destination(self):
        return self._reached_destination

    @reached_destination.setter
    def reached_destination(self, value):
        self._reached_destination = value


    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # TODO: Initialize any additional variables here
        self.reward_balance = 0
        self.trials = 0
        self.remaining = 0
        self.reached_destination = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.reward_balance = 0
        self.trials += 1

        if (self.remaining > 0): # then we must have reached the destination
            self.reached_destination += 1
            print "Reached destination ({})".format(self.trials)
    
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)  # t increments as deadline decrements, but both reset after each trial
        self.remaining = deadline

        # TODO: Update state
        
        # TODO: Select action according to your policy
        rando = ['left', 'right', 'forward', None]
        action = random.choice(rando)

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.reward_balance = self.reward_balance + reward
        self.reward_history = self.reward_balance

        # TODO: Learn policy based on state, action, reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""
    
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent

    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=10)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    print "\n\n--------------------------------------"
    print "{} out of {} reached the destination".format(a.reached_destination, a.trials)
    print "--------------------------------------\n\n"

    plt.hist(a.reward_history, bins='auto')
    plt.title("Reward Balance History Over {} Random Actions".format(a.trials))
    plt.show()

    print 'Done'


if __name__ == '__main__':
    run()
