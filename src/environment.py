'''
Implementation of a basic RL environment for continuous spaces.
Includes three test problems which were used in generating the figures.
'''

import numpy as np
import gym
import math

#-------------------------------------------------------------------------------


class Environment(object):
    '''General RL environment'''

    def __init__(self):
        pass

    def reset(self):
        pass

    def advance(self, action):
        '''
        Moves one step in the environment.

        Args:
            action

        Returns:
            reward - double - reward
            newState - int - new state
            pContinue - 0/1 - flag for end of the episode
        '''
        return 0, 0, 0


#-------------------------------------------------------------------------------

'''Implementation of a continuous environment using the AI Gym from Google'''
class ContinuousAIGym(Environment):
    def __init__(self, env, epLen):
        '''
            env - AI Gym Environment
            epLen - Number of steps per episode
        '''
        self.env = env
        self.epLen = epLen
        self.timestep = 0
        self.state = self.env.reset()


    def get_epLen(self):
        return self.epLen

    def reset(self):
        '''Reset the environment'''
        self.timestep = 0
        self.state = self.env.reset()

    def advance(self, action):
        '''
        Move one step in the environment

        Args:
        action - int - chosen action
        Returns:
            reward - double - reward
            newState - int - new state
            pContinue - 0/1 - flag for end of the episode
        '''
        newState, reward, terminal, info = self.env.step(action)

        if self.timestep == self.epLen or terminal:
            pContinue = 0
            self.reset()
        else:
            pContinue = 1

        return reward, newState, pContinue

#-------------------------------------------------------------------------------
'''An ambulance environment over [0,1].  An agent interacts through the environment
by picking a location to station the ambulance.  Then a patient arrives and the ambulance
most go and serve the arrival, paying a cost of travel.'''

class AmbulanceEnvironment(Environment):
    def __init__(self, epLen, arrivals, alpha, starting_state):
        '''
        epLen - number of steps
        arrivals - arrival distribution for patients
        alpha - parameter for difference in costs
        starting_state - starting location
        '''
        self.epLen = epLen
        self.arrivals = arrivals
        self.alpha = alpha
        self.state = starting_state
        self.starting_state = starting_state
        self.timestep = 0


    def get_epLen(self):
        return self.epLen

    def reset(self):
        '''Reset the environment'''
        self.timestep = 0
        self.state = self.starting_state

    def advance(self, action):
        '''
        Move one step in the environment

        Args:
        action - int - chosen action
        Returns:
            reward - double - reward
            newState - int - new state
            pContinue - 0/1 - flag for end of the episode
        '''
        old_state = self.state
        # new state is sampled from the arrivals distribution
        newState = self.arrivals(self.timestep)

        # Cost is a linear combination of the distance traveled to the action
        # and the distance served to the pickup
        reward = 1-(self.alpha * np.abs(self.state - action) + (1 - self.alpha) * np.abs(action - newState))

        if self.timestep == self.epLen:
            pContinue = 0
            self.reset()
        else:
            pContinue = 1
        self.state = newState
        self.timestep += 1
        return reward, newState, pContinue
#-------------------------------------------------------------------------------

'''An oil environment also over [0,1].  Here the agent interacts with the environment
by picking a location to travel to, paying a cost of travel, and receiving a reward at the new location.'''
class OilEnvironment(Environment):
    def __init__(self, epLen, oil_prob, starting_state, cost_param, noise_variance):
        '''
        epLen - number of steps
        oil_prob - a function returning a reward in [0,1] for being in a state
        starting_state - the starting state
        '''
        self.epLen = epLen
        self.state = starting_state
        self.starting_state = starting_state
        self.timestep = 0
        self.oil_prob = oil_prob
        self.cost_param = cost_param
        self.noise_variance = noise_variance


    def get_epLen(self):
        return self.epLen

    def reset(self):
        '''Reset the environment'''
        self.timestep = 0
        self.state = self.starting_state

    def advance(self, action):
        '''
        Move one step in the environment

        Args:
        action - int - chosen action
        Returns:
            reward - double - reward
            newState - int - new state
            pContinue - 0/1 - flag for end of the episode
        '''


        reward = min(1, max(self.oil_prob(self.state, action, self.timestep) - self.cost_param*np.abs(self.state - action),0))
        newState = min(1, max(0, action + np.random.normal(0, np.sqrt(self.noise_variance(self.state, action, self.timestep)))))

        if self.timestep == self.epLen:
            pContinue = 0
            self.reset()
        else:
            pContinue = 1
        self.state = newState
        self.timestep += 1
        return reward, newState, pContinue

#-------------------------------------------------------------------------------

''' A discrete environment implemented over the interval [0,1] to check the performance
of the algorithm on simple problem instances.'''
class TestContinuousEnvironment(Environment):
    def __init__(self, epLen):
        '''
        epLen - number of steps
        '''
        self.starting_state = 0
        self.timestep = 0
        self.epLen = epLen


    def get_epLen(self):
        return self.epLen

    def reset(self):
        '''Reset the environment'''
        self.timestep = 0
        self.state = 0

    def advance(self, action):
        '''
        Move one step in the environment

        Args:
        action - int - chosen action
        Returns:
            reward - double - reward
            newState - int - new state
            pContinue - 0/1 - flag for end of the episode
        '''
        state = self.state

        # Four discrete states with discrete transitions and 0-1 rewards
        if state <= 0.5 and action <= 0.5:
            newState = np.random.uniform(0,0.5)
            reward = 0
        elif state <= 0.5 and action > 0.5:
            newState = np.random.uniform(0.5, 1)
            reward = 1
        elif state >= 0.5 and action < 0.5:
            newState = np.random.uniform(0,0.5)
            reward = 0
        else:
            newState = np.random.uniform(0.5, 1)
            reward = 1

        if self.timestep == self.epLen:
            pContinue = 0
            self.reset()
        else:
            pContinue = 1
        self.state = newState
        self.timestep += 1
        return reward, newState, pContinue


#-------------------------------------------------------------------------------
# Benchmark environments used when running an experiment

def oil_prob_1(x, lam, c):
    return np.exp(-1*lam*np.abs(x - c))

def oil_prob_2(x, lam, c):
    return 1 - lam*(x-c)**2

def makeOilEnvironment(epLen, oil_prob, starting_state, cost_param, noise_variance):
    return OilEnvironment(epLen, oil_prob, starting_state, cost_param, noise_variance)

def makeLaplaceOil(epLen, lam, starting_state):
    return OilEnvironment(epLen, lambda x,a,h: oil_prob_1(x, lam, .75), starting_state, 1, lambda x,a,h : 0)

def makeQuadraticOil(epLen, lam, starting_state):
    return OilEnvironment(epLen, lambda x,a,h: oil_prob_2(x, lam, .75), starting_state, 1, lambda x,a,h : 0)

def makeTestMDP(epLen):
    return TestContinuousEnvironment(epLen)

def make_ambulanceEnvMDP(epLen, arrivals, alpha, starting_state):
    return AmbulanceEnvironment(epLen, arrivals, alpha, starting_state)
