import numpy as np
from src import agent

class eNetModelBased(agent.FiniteHorizonAgent):

    def __init__(self, action_net, state_net, epLen, scaling, state_action_dim, alpha):
        '''
        args:
            - action_net - epsilon net of action space
            - state_net - epsilon net of state space
            - epLen - steps per episode
            - scaling - scaling parameter for UCB terms
        '''

        self.state_net = np.resize(state_net, (state_action_dim[0], len(state_net))).T
        self.action_net = np.resize(action_net, (state_action_dim[1], len(action_net))).T
        self.epLen = epLen
        self.scaling = scaling
        self.alpha = alpha
        self.state_action_dim = state_action_dim
        self.state_size = self.state_action_dim[0] * [len(state_net)]
        self.action_size = self.state_action_dim[1] * [len(action_net)]
        self.qVals = np.ones([self.epLen]+self.state_size+self.action_size, dtype=np.float32) * self.epLen
        self.num_visits = np.zeros([self.epLen]+ self.state_size+self.action_size, dtype=np.float32)
        self.vVals = np.ones([self.epLen]+ self.state_size, dtype=np.float32) * self.epLen
        self.rEst = np.zeros([self.epLen]+ self.state_size+ self.action_size, dtype=np.float32)
        self.pEst = np.zeros([self.epLen]+ self.state_size+ self.action_size+self.state_size,
                             dtype=np.float32)

        '''
            Resets the agent by overwriting all of the estimates back to zero
        '''
    def reset(self):
        self.qVals = np.ones([self.epLen]+ self.state_size+ self.action_size, dtype=np.float32) * self.epLen
        self.vVals = np.ones([self.epLen]+ self.state_size, dtype=np.float32) * self.epLen
        self.rEst = np.zeros([self.epLen]+ self.state_size+ self.action_size, dtype=np.float32)
        self.num_visits = np.zeros([self.epLen]+ self.state_size+ self.action_size, dtype=np.float32)
        self.pEst = np.zeros([self.epLen]+ self.state_size+ self.action_size+self.state_size,
                             dtype=np.float32)
        '''
            Adds the observation to records by using the update formula
        '''
    def update_obs(self, obs, action, reward, newObs, timestep):
        '''Add observation to records'''

        # returns the discretized state and action location
        state_discrete = np.argmin((np.abs(self.state_net - np.asarray(obs))), axis=0)
        action_discrete = np.argmin((np.abs(self.action_net - np.asarray(action))), axis=0)
        state_new_discrete = np.argmin((np.abs(self.state_net - np.asarray(newObs))), axis=0)

        dim = (timestep,) + tuple(state_discrete) + tuple(action_discrete)
        self.num_visits[dim] += 1
        # t = self.num_visits[dim]
        # lr = (self.epLen + 1) / (self.epLen + t)
        # bonus = self.scaling * np.sqrt(1 / t)
        #
        # if timestep == self.epLen-1:
        #     vFn = 0
        # else:
        #     vFn = np.max(self.qVals[(timestep+1,) + tuple(state_new_discrete)])
        # vFn = min(self.epLen, vFn)
        #
        # self.qVals[dim] = (1 - lr) * self.qVals[dim] + lr * (reward + vFn + bonus)
        self.pEst[dim+tuple(state_new_discrete)] += 1
        t = self.num_visits[dim]
        self.rEst[dim] = ((t - 1) * self.rEst[dim] + reward) / t



    def get_num_arms(self):
        ''' Returns the number of arms'''
        return self.epLen * len(self.state_net)**(self.state_action_dim[0]) * len(self.action_net)**(self.state_action_dim[1])

    def update_policy(self, k):
        '''Update internal policy based upon records'''
        # Update value estimates
        #axes_count = np.sum(np.array([self.epLen]+ self.state_size+ self.action_size))
        #self.qVals = np.where(self.num_visits==0,self.epLen,self.qVals)
        # self.qVals[self.epLen-1] = np.where(self.qVals[self.epLen-1]<self.rEst[self.epLen-1]+ self.scaling / np.sqrt(self.num_visits[self.epLen-1]), self.qVals[self.epLen-1], self.rEst[self.epLen-1]+ self.scaling / np.sqrt(self.num_visits[self.epLen-1]))
        # self.qVals[self.epLen-1] = np.where(self.qVals[self.epLen-1]<self.epLen, self.qVals[self.epLen-1], self.epLen)
        # vEst = np.dot(self.vVals[1:self.epLen, :], (self.pEst[0:self.epLen-1] + self.alpha) / (
        #                                 np.sum(self.pEst[0:self.epLen-1], axis = np.arange(axes_count)) + self.state_action_dim[0]*len(self.state_net) * self.alpha))
        # self.qVals[0:self.epLen-1]
        for h in np.arange(self.epLen - 1, -1, -1):
            for state1 in range(len(self.state_net)):
                for state2 in range(len(self.state_net)):
                    for action1 in range(len(self.action_net)):
                        for action2 in range(len(self.action_net)):
                            if self.num_visits[h, state1, state2, action1,action2] == 0:
                                self.qVals[h, state1, state2, action1, action2] = self.epLen
                            else:
                                if h == self.epLen - 1:
                                    self.qVals[h, state1, state2, action1, action2] = min(self.qVals[h, state1, state2, action1, action2], self.epLen,self.rEst[h, state1, state2,action1, action2] + self.scaling / np.sqrt(self.num_visits[h, state1, state2, action1, action2]))
                                else:
                                    vEst = np.sum(self.vVals[h + 1]* ((self.pEst[h, state1, state2, action1, action2] + self.alpha) / (
                                                np.sum(self.pEst[h, state1, state2, action1, action2]) + len(self.state_net) * self.alpha)))
                                    self.qVals[h, state1, state2, action1, action2] = min(self.qVals[h, state1, state2, action1, action2], self.epLen,
                                                                   self.rEst[h, state1, state2, action1, action2] + self.scaling / np.sqrt(
                                                                   self.num_visits[h, state1, state2, action1, action2]) + vEst)
                    self.vVals[h, state1, state2] = min(self.epLen, np.max(self.qVals[h, state1, state2, :]))
        self.greedy = self.greedy


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
        state_discrete = np.argmin((np.abs(np.asarray(self.state_net) - np.asarray(state))), axis=0)
        qFn = self.qVals[(timestep,)+tuple(state_discrete)]
        action = np.asarray(np.where(qFn == qFn.max()))
        a = len(action[0])
        index = np.random.choice(len(action[0]))

        actions = ()
        for val in action.T[index]:
            actions += (self.action_net[:,0][val],)
        return actions
        #a = self.action_net[tuple(action.T[index])]
        #b = action.T[index]
        #return self.action_net[action.T[index]]

    def pick_action(self, state, step):
        action = self.greedy(state, step)
        if len(action) ==1:
            return action[0]
        return action
