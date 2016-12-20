import random
import sys
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import matplotlib.pyplot as plt

class LearningAgent(Agent):
    """
    An agent that learns to drive in the smartcab world.
    This is the implementation of Q3, the initial implementation of Q-Learning.
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

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # TODO: Initialize any additional variables here
        self.reward_balance = 0
        self.trials = 0
        self.remaining = 0
        self.reached_destination = 0
        self.state = (None,None)
        self.iterations = 0
        self.Qmatrix = {}
        self.prevState = None
        self.prevAction = None
        self.actions = ['left', 'right', 'forward', 'None']
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.reward_balance = 0
        self.state = (None,None)
        self.prevState = None
        self.prevAction = None

        self.trials += 1
        if (self.remaining > 0): # then we must have reached the destination
            self.reached_destination += 1
            print "Reached destination ({})".format(self.trials)
    
    def makeStateKey(self, inputs):
        '''Make a key for this state, based on the inputs and next waypoint'''
        S = ""
        for key, value in inputs.iteritems():
            S += key + value.__str__() + '|'
        S += self.next_waypoint.__str__()
        return S

    def selectAction(self, newState):
        '''Use Q-Learning to select an action'''
        rando = random.random()
        if rando < self.epsilon:
            action = random.choice(self.actions)
            print "selectAction choosing random action", action
        else:
            allQ = [self.Qmatrix.get((newState, a), 0.0) for a in self.actions]
            maxQ = max(allQ)

            if maxQ > 0.0:
                print "Chose action using Q: ", maxQ

            numMax = allQ.count(maxQ)
            if numMax > 1:
                maxList = [i for i in range(len(self.actions)) if allQ[i] == maxQ]
                idx = random.choice(maxList)
            else:
                idx = allQ.index(maxQ)

            action = self.actions[idx]
        return action

    def updateQmatrix(self, newState, newAction, prevState, prevAction, reward):
        '''Update the state-action matrix with the new Q value'''
        prevQnew = self.Qmatrix.get((newState, newAction), 0.0)
        self.Qmatrix[(newState, newAction)] = prevQnew + reward

        if (prevState is not None) and (prevAction is not None):
            maxQ = max([self.Qmatrix.get((newState, a), 0.0) for a in self.actions])
            utility = reward + (self.gamma * maxQ)
            prevQ = self.Qmatrix.get((prevState, prevAction))
            # IDEA - IF PREVQ IS NEGATIVE, DON'T ADD EXPECTED VALUE
            self.Qmatrix[(prevState, prevAction)] = prevQ + self.alpha * (utility - prevQ)


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)  # t increments as deadline decrements, but both reset after each trial
        self.remaining = deadline

        # TODO: Update state
        newState = self.makeStateKey(inputs)
        
        # TODO: Select action according to your policy
        action = self.selectAction(newState)

        # Execute action and get reward
        if (action == 'None'):
            reward = self.env.act(self, None)
        else:
            reward = self.env.act(self, action)

        self.state = (newState, action)

        self.reward_balance = self.reward_balance + reward
        self.reward_history = self.reward_balance

        # TODO: Learn policy based on state, action, reward
        self.updateQmatrix(newState, action, self.prevState, self.prevAction, reward)
        self.prevState = newState
        self.prevAction = action

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""
    
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent

    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.01, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    print "\n\n--------------------------------------"
    print "{} out of {} reached the destination".format(a.reached_destination, a.trials)
    print "--------------------------------------\n\n"

    plt.hist(a.reward_history, bins='auto')
    plt.title("Reward Balance History Over {} Q-Learning Actions".format(a.trials))
    plt.show()

    print 'Done'


if __name__ == '__main__':
    run()
