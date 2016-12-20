import random
import sys
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import matplotlib.pyplot as plt
import sklearn

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
    def alpha(self):
        return self._alpha
        
    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def gamma(self):
        return self._gamma
        
    @gamma.setter
    def gamma(self, value):
        self._gamma = value

    @property
    def epsilon(self):
        return self._epsilon
        
    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value

    @property
    def Qmatrix(self):
        return self._Qmatrix
        
    @Qmatrix.setter
    def Qmatrix(self, value):
        self._Qmatrix = value

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

    @property
    def total_penalties(self):
        return self._total_penalties

    @total_penalties.setter
    def total_penalties(self, value):
        self._total_penalties = value

    @property
    def penalties(self):
        return self._penalties

    @penalties.setter
    def penalties(self, value):
        self._penalties = value

    @property
    def total_actions(self):
        return self._total_actions

    @total_actions.setter
    def total_actions(self, value):
        self._total_actions = value

    @property
    def exploring(self):
        return self._exploring

    @exploring.setter
    def exploring(self, value):
        self._exploring = value

    @property
    def exploring_mistakes(self):
        return self._exploring_mistakes

    @exploring_mistakes.setter
    def exploring_mistakes(self, value):
        self._exploring_mistakes = value

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
        self.alpha = 0.3
        self.gamma = 0.1
        self.epsilon = 0.1
        self.iterations = 0
        self.penalties = 0
        self.total_actions = 0
        self.exploring = False

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
            #action = random.choice(self.actions) # choose random from all actions

            untried_actions = [] # choose random from untried actions first
            for act in self.actions:
                if self.Qmatrix.get((newState, act), 0.0) == 0.0:
                    untried_actions.append(act)

            if len(untried_actions) == 0: # we've already tried them all at least once
                untried_actions = self.actions

            self.exploring = True
            action = random.choice(untried_actions)
        else:
            allQ = [self.Qmatrix.get((newState, a), 0.0) for a in self.actions]
            maxQ = max(allQ)

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
            self.Qmatrix[(prevState, prevAction)] = prevQ + self.alpha * (utility - prevQ)


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)  # t increments as deadline decrements, but both reset after each trial
        self.remaining = deadline

        # Decay alpha
        self.iterations += 1
        self.alpha = 1.0 / self.iterations  # to encourage convergence

        # TODO: Update state
        newState = self.makeStateKey(inputs)
        
        # TODO: Select action according to your policy
        self.exploring = False
        action = self.selectAction(newState)
        self.total_actions += 1

        # Execute action and get reward
        if (action == 'None'):
            reward = self.env.act(self, None)
        else:
            reward = self.env.act(self, action)

        if reward < 0:
            self.penalties += 1
            self.total_penalties += 1
            if self.exploring:
                self.exploring_mistakes += 1

        self.state = (newState, action)

        self.reward_balance = self.reward_balance + reward
        self.reward_history = self.reward_balance

        # TODO: Learn policy based on state, action, reward
        self.updateQmatrix(newState, action, self.prevState, self.prevAction, reward)
        self.prevState = newState
        self.prevAction = action

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""
    
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    a.total_penalties = 0
    a.exploring_mistakes = 0

    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    # Grid search over hyperparameters
    best_score = {'alpha': 0.0, 'gamma': 0.0, 'epsilon': 0.0, 'score': 0.0}
    lowest_penalties = {'alpha': 0.0, 'gamma': 0.0, 'epsilon': 0.0, 'penalties': float("inf")}
    step_size = 1
    for al in [x / 10.0 for x in range(1, 5, step_size)]:
        for gam in [x / 10.0 for x in range(1, 6, step_size)]:
            for eps in [x / 10.0 for x in range(1, 5, step_size)]:
                a.alpha = al
                a.gamma = gam
                a.epsilon = eps
                a.trials = 0
                a.reached_destination = 0
                a.Qmatrix = {}
                sim.run(n_trials=100)  # run for a specified number of trials
                # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

                success = float(a.reached_destination) / float(a.trials)
                if success > best_score['score']:
                    best_score = {'alpha': al, 'gamma': gam, 'epsilon': eps, 'score': success}

                mistakes = float(a.penalties) / float(a.total_actions)
                if mistakes < lowest_penalties['penalties']:
                    lowest_penalties = {'alpha': al, 'gamma': gam, 'epsilon': eps, 'penalties': mistakes}

    print "\n\n--------------------------------------"
    print "Best score: alpha = {}, gamma = {}, epsilon = {}, score = {}".format(best_score['alpha'], best_score['gamma'], best_score['epsilon'], best_score['score'])
    print "Lowest penalties: alpha = {}, gamma = {}, epsilon = {}, penalties = {}".format(lowest_penalties['alpha'], lowest_penalties['gamma'], lowest_penalties['epsilon'], lowest_penalties['penalties'])
    print "{} out of {} penalties {:.2%} were during exploration".format(a.exploring_mistakes, a.total_penalties, float(a.exploring_mistakes) / a.total_penalties)
    print "--------------------------------------\n\n"

    # Plot reward balance
    plt.hist(a.reward_history, bins='auto')
    plt.title("Reward Balance History Over {} Q-Learning Actions".format(a.trials))
    plt.show()

    print 'Done'


if __name__ == '__main__':
    run()
