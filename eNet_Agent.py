import numpy as np
from src import agent

''' epsilon Net agent '''
class eNet(agent.FiniteHorizonAgent):

    def __init__(self, action_net, state_net, epLen, scaling):
        '''
        args:
            - action_net - epsilon net of action space
            - state_net - epsilon net of state space
            - epLen - steps per episode
            - scaling - scaling parameter for UCB terms
        '''
        self.action_net = action_net
        self.state_net = state_net
        self.epLen = epLen
        self.scaling = scaling

        self.qVals = np.ones([self.epLen, len(self.state_net), len(self.action_net)], dtype=np.float32) * self.epLen
        self.num_visits = np.zeros([self.epLen, len(self.state_net), len(self.action_net)], dtype=np.float32)

        '''
            Resets the agent by overwriting all of the estimates back to zero
        '''
    def reset(self):
        self.qVals = np.ones([self.epLen, len(self.state_net), len(self.action_net)], dtype=np.float32) * self.epLen
        self.num_visits = np.zeros([self.epLen, len(self.state_net), len(self.action_net)], dtype=np.float32)


        '''
            Adds the observation to records by using the update formula
        '''
    def update_obs(self, obs, action, reward, newObs, timestep):
        '''Add observation to records'''

        # returns the discretized state and action location
        state_discrete = np.argmin(np.abs(np.asarray(self.state_net) - obs))
        action_discrete = np.argmin(np.abs(np.asarray(self.action_net) - action))
        state_new_discrete = np.argmin(np.abs(np.asarray(self.state_net) - newObs))

        self.num_visits[timestep, state_discrete, action_discrete] += 1
        t = self.num_visits[timestep, state_discrete, action_discrete]
        lr = (self.epLen + 1) / (self.epLen + t)
        bonus = self.scaling * np.sqrt(1 / t)

        if timestep == self.epLen-1:
            vFn = 0
        else:
            vFn = max(self.qVals[timestep+1, state_new_discrete, :])
        vFn = min(self.epLen, vFn)

        self.qVals[timestep, state_discrete, action_discrete] = (1 - lr) * self.qVals[timestep, state_discrete, action_discrete] + lr * (reward + vFn + bonus)

    def get_num_arms(self):
        ''' Returns the number of arms'''
        return self.epLen * len(self.state_net) * len(self.action_net)

    def update_policy(self, k):
        '''Update internal policy based upon records'''
        # TODO: Verify if this is needed
        # self.greedy = self.greedy
        # print('Update policy episode: ' + str(k))
        # print(self.qVals[self.epLen-1, :, :])
        return

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
        action = self.greedy(state, step)
        return action
