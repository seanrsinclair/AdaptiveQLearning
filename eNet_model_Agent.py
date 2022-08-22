import numpy as np
import time
from src import agent

''' epsilon Net agent '''
class eNetModelBased(agent.FiniteHorizonAgent):

    def __init__(self, action_net, state_net, epLen, scaling, alpha, flag):
        '''
        args:
            - action_net - epsilon net of action space
            - state_net - epsilon net of state space
            - epLen - steps per episode
            - scaling - scaling parameter for UCB terms
            - alpha - used in adding a prior on the transition kernel
            - flag - used in determining full update (TRUE) or one-step update (FALSE)
        '''
        self.action_net = action_net
        self.state_net = state_net
        self.epLen = epLen
        self.scaling = scaling
        self.alpha = alpha
        self.flag = flag

        '''
        Initializes the estimates to be used in the algorithm:
            - qVals - estimate of the q values
            - vVals - estimate of the value function
            - rEst - average reward
            - num_visits - number of visits
            - pEst - array containing number of (x,a,x_new) visits
        '''
        self.qVals = np.ones([self.epLen, len(self.state_net), len(self.action_net)], dtype=np.float32) * self.epLen
        self.vVals = np.ones([self.epLen, len(self.state_net)], dtype=np.float32) * self.epLen
        self.rEst = np.zeros([self.epLen, len(self.state_net), len(self.action_net)], dtype=np.float32)
        self.num_visits = np.zeros([self.epLen, len(self.state_net), len(self.action_net)], dtype=np.float32)
        self.pEst = np.zeros([self.epLen, len(self.state_net), len(self.action_net), len(self.state_net)], dtype=np.float32)

        '''
            Resets the agent by overwriting all of the estimates back to zero
        '''
    def reset(self):
        self.qVals = np.ones([self.epLen, len(self.state_net), len(self.action_net)], dtype=np.float32) * self.epLen
        self.vVals = np.ones([self.epLen, len(self.state_net)], dtype=np.float32) * self.epLen
        self.rEst = np.zeros([self.epLen, len(self.state_net), len(self.action_net)], dtype=np.float32)
        self.num_visits = np.zeros([self.epLen, len(self.state_net), len(self.action_net)], dtype=np.float32)
        self.pEst = np.zeros([self.epLen, len(self.state_net), len(self.action_net), len(self.state_net)], dtype=np.float32)
        '''
            Adds the observation to records by using the update formula
        '''
    def update_obs(self, obs, action, reward, newObs, timestep):
        '''Add observation to records'''

        # returns the discretized state and action location
        state_discrete = np.argmin(np.abs(np.asarray(self.state_net) - obs))
        action_discrete = np.argmin(np.abs(np.asarray(self.action_net) - action))
        state_new_discrete = np.argmin(np.abs(np.asarray(self.state_net) - newObs))

        # increments the number of visits
        self.num_visits[timestep, state_discrete, action_discrete] += 1

        # updates the average reward and number of transitions
        self.pEst[timestep, state_discrete, action_discrete, state_new_discrete] += 1
        t = self.num_visits[timestep, state_discrete, action_discrete]
        self.rEst[timestep, state_discrete, action_discrete] = ((t-1)*self.rEst[timestep, state_discrete, action_discrete] + reward) / t



    def get_num_arms(self):
        ''' Returns the number of arms'''
        return self.epLen * len(self.state_net) * len(self.action_net)

    def update_policy(self, k):
        '''Update internal policy based upon records'''
        # print(k)
        # Update value estimates

        # Solves the full Bellman equations by looping backwards
        if self.flag:
            for h in np.arange(self.epLen-1,-1,-1):
                for state in range(len(self.state_net)):
                    for action in range(len(self.action_net)):

                        # If a state action has not been visited, initialize to H
                        # for optimism
                        if self.num_visits[h, state, action] == 0:
                            self.qVals[h, state, action] = self.epLen
                        else:

                            # Otherwise estimates based off the Bellman equations
                            if h == self.epLen - 1: # for the last step the value function at the next step is zero
                                self.qVals[h, state, action] = min(self.qVals[h, state, action], self.epLen, self.rEst[h, state, action] + self.scaling / np.sqrt(self.num_visits[h, state, action]))
                            else:
                                vEst = np.dot(self.vVals[h+1, :], (self.pEst[h, state, action, :] + self.alpha) / (np.sum(self.pEst[h, state, action, :]) + len(self.state_net)*self.alpha))
                                self.qVals[h, state, action]  = min(self.qVals[h, state, action], self.epLen, self.rEst[h, state, action] + self.scaling / np.sqrt(self.num_visits[h, state, action]) + vEst)
                    # Updates the estimate of the value function
                    self.vVals[h, state] = min(self.epLen, np.max(self.qVals[h, state, :]))

        # TODO: Again - verify if this is needed.
        # self.greedy = self.greedy


    def greedy(self, state, timestep, epsilon=0):
        '''
        Select action according to a greedy policy

        Args:
            state - int - current state
            timestep - int - timestep *within* episode

        Returns:
            action - int
        '''
        # returns the discretized state location and takes action based on
        # maximum q value
        state_discrete = np.argmin(np.abs(np.asarray(self.state_net) - state))
        qFn = self.qVals[timestep, state_discrete, :]
        action = np.random.choice(np.where(qFn == qFn.max())[0])
        return self.action_net[action]

    def pick_action(self, state, step):
        if self.flag == False:
            state_discrete = np.argmin(np.abs(np.asarray(self.state_net) - state))
            for action in range(len(self.action_net)):
                if self.num_visits[step, state_discrete, action] == 0:
                        self.qVals[step, state_discrete, action] = self.epLen
                else:
                    if step == self.epLen - 1: # for the last step the value function at the next step is zero
                        self.qVals[step, state_discrete, action] = min(self.qVals[step, state_discrete, action], self.epLen, self.rEst[step, state_discrete, action] + self.scaling / np.sqrt(self.num_visits[step, state_discrete, action]))
                    else:
                        vEst = np.dot(self.vVals[step+1, :], (self.pEst[step, state_discrete, action, :] + self.alpha) / (np.sum(self.pEst[step, state_discrete, action, :]) + len(self.state_net)*self.alpha))
                        self.qVals[step, state_discrete, action]  = min(self.qVals[step, state_discrete, action], self.epLen, self.rEst[step, state_discrete, action] + self.scaling / np.sqrt(self.num_visits[step, state_discrete, action]) + vEst)
                    # Updates the estimate of the value function
                    self.vVals[step, state_discrete] = min(self.epLen, np.max(self.qVals[step, state_discrete, :]))                   


        action = self.greedy(state, step)
        return action
