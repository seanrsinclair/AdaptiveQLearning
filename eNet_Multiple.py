import numpy as np
from src import agent

''' epsilon Net agent '''
class eNet(agent.FiniteHorizonAgent):

    def __init__(self, action_net, state_net, epLen, scaling, state_action_dim):
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
        self.state_action_dim = state_action_dim

        dim = [self.epLen]
        dim += self.state_action_dim[0] * [len(state_net)]
        dim += self.state_action_dim[1] * [len(action_net)]
        self.matrix_dim = dim
        self.qVals = np.ones(self.matrix_dim, dtype=np.float32) * self.epLen
        self.num_visits = np.zeros(self.matrix_dim, dtype=np.float32)

        '''
            Resets the agent by overwriting all of the estimates back to zero
        '''
    def reset(self):
        self.qVals = np.ones(self.matrix_dim, dtype=np.float32) * self.epLen
        self.num_visits = np.zeros(self.matrix_dim, dtype=np.float32)

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
        t = self.num_visits[dim]
        lr = (self.epLen + 1) / (self.epLen + t)
        bonus = self.scaling * np.sqrt(1 / t)

        if timestep == self.epLen-1:
            vFn = 0
        else:
            vFn = np.max(self.qVals[(timestep+1,) + tuple(state_new_discrete)])
        vFn = min(self.epLen, vFn)

        self.qVals[dim] = (1 - lr) * self.qVals[dim] + lr * (reward + vFn + bonus)

    def get_num_arms(self):
        ''' Returns the number of arms'''
        return self.epLen * len(self.state_net)**(self.state_action_dim[0]) * len(self.action_net)**(self.state_action_dim[1])

    def update_policy(self, k):
        '''Update internal policy based upon records'''
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
