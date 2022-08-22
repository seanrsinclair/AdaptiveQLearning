import numpy as np
from src import agent

''' Agent which implements several heuristic algorithms'''
class dataUpdateAgent(agent.FiniteHorizonAgent):

    def __init__(self, epLen, func, alpha):
        '''args:
            epLen - number of steps
            func - function used to decide action
            data - all data observed so far
            alpha - alpha parameter in ambulance problem
        '''
        self.epLen = epLen
        self.func = func
        self.data = []
        self.alpha = alpha

    def reset(self):
        # resets data matrix to be empty
        self.data = []

    def update_obs(self, obs, action, reward, newObs, timestep):
        '''Add observation to records'''
        self.data.append(newObs)
        return

    def get_num_arms(self):
        return 0

    def update_policy(self, k):
        '''Update internal policy based upon records'''
        self.greedy = self.greedy


    def greedy(self, state, timestep, epsilon=0):
        '''
        Select action according to function
        '''
        if timestep == 0:
            return state
        else:
            action = self.func(self.data)
            return action


    def pick_action(self, state, step):
        action = self.greedy(state, step)
        return action
