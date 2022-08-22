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

class eNet_model(agent.FiniteHorizonAgent):

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
        # axes_count = np.sum(np.array([self.epLen]+ self.state_size+ self.action_size))
        # self.qVals = np.where(self.num_visits==0,self.epLen,self.qVals)
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
                                    self.qVals[h, state1, state2, action1, action2] = min(self.qVals[h, state1, state2, action1, action2], self.epLen,
                                                                       self.rEst[h, state1, state2,action1, action2] + self.scaling / np.sqrt(
                                                                           self.num_visits[h, state1, state2, action1, action2]))
                                else:
                                    vEst = np.dot(self.vVals[h + 1, :], (self.pEst[h, state1, state2, action1, action2, :] + self.alpha) / (
                                                np.sum(self.pEst[h, state1, state2, action1, action2, :]) + len(self.state_net) * self.alpha))
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

class eNet_Discount(agent.FiniteHorizonAgent):

    def __init__(self, action_net, state_net, discount_factor, scaling, state_action_dim):
        '''
        args:
            - action_net - epsilon net of action space
            - state_net - epsilon net of state space
            - epLen - steps per episode
            - scaling - scaling parameter for UCB terms
        '''

        self.state_net = np.resize(state_net, (state_action_dim[0], len(state_net))).T
        self.action_net = np.resize(action_net, (state_action_dim[1], len(action_net))).T
        self.discount_factor = discount_factor
        self.scaling = scaling
        self.state_action_dim = state_action_dim

        dim = self.state_action_dim[0] * [len(state_net)]
        dim += self.state_action_dim[1] * [len(action_net)]
        self.matrix_dim = dim
        self.qVals = np.ones(self.matrix_dim, dtype=np.float32) * 1/(1-self.discount_factor)
        self.num_visits = np.zeros(self.matrix_dim, dtype=np.float32)

        '''
            Resets the agent by overwriting all of the estimates back to zero
        '''
    def reset(self):
        self.qVals = np.ones(self.matrix_dim, dtype=np.float32) * 1/(1-self.discount_factor)
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

        dim = tuple(state_discrete) + tuple(action_discrete)
        self.num_visits[dim] += 1
        t = self.num_visits[dim]
        lr = (200 + 1) / (200 + t)
        bonus = self.scaling * np.sqrt(1 / t)

        vFn = np.max(self.qVals[tuple(state_new_discrete)])
        vFn = min(1/(1-self.discount_factor), vFn)

        self.qVals[dim] = (1 - lr) * self.qVals[dim] + lr * (reward + self.discount_factor*vFn + bonus)

    def get_num_arms(self):
        ''' Returns the number of arms'''
        return len(self.state_net)**(self.state_action_dim[0]) * len(self.action_net)**(self.state_action_dim[1])

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
        qFn = self.qVals[tuple(state_discrete)]
        action = np.asarray(np.where(qFn == qFn.max()))
        a = len(action[0])
        index = np.random.choice(len(action[0]))

        actions = ()
        for val in action.T[index]:
            actions += (self.action_net[val,0],)
        return actions
        #a = self.action_net[tuple(action.T[index])]
        #b = action.T[index]
        #return self.action_net[action.T[index]]

    def pick_action(self, state, step):
        action = self.greedy(state, step)
        if len(action) ==1:
            return action[0]
        return action


''' epsilon Net agent '''
class eNetAmbulancce(agent.FiniteHorizonAgent):

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
        state_discrete = np.argmin(np.abs(np.asarray(self.state_net) - state))
        qFn = self.qVals[timestep, state_discrete, :]
        action = np.random.choice(np.where(qFn == qFn.max())[0])
        return self.action_net[action]

    def pick_action(self, state, step):
        action = self.greedy(state, step)
        return action

class eNet_Multiple(agent.FiniteHorizonAgent):

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

        self.qVals = np.ones([self.epLen, len(self.state_net), len(self.state_net),len(self.action_net), len(self.action_net)], dtype=np.float32) * self.epLen
        self.num_visits = np.zeros([self.epLen, len(self.state_net), len(self.state_net), len(self.action_net), len(self.action_net)], dtype=np.float32)

        '''
            Resets the agent by overwriting all of the estimates back to zero
        '''
    def reset(self):
        self.qVals = np.ones([self.epLen, len(self.state_net), len(self.state_net),len(self.action_net), len(self.action_net)], dtype=np.float32) * self.epLen
        self.num_visits = np.zeros([self.epLen, len(self.state_net), len(self.state_net), len(self.action_net), len(self.action_net)], dtype=np.float32)

        '''
            Adds the observation to records by using the update formula
        '''
    def update_obs(self, obs, action, reward, newObs, timestep):
        '''Add observation to records'''

        # returns the discretized state and action location
        state_discrete = (np.argmin(np.abs(np.asarray(self.state_net) - obs[0])), np.argmin(np.abs(np.asarray(self.state_net) - obs[1])))
        action_discrete = (np.argmin(np.abs(np.asarray(self.action_net) - action[0])), np.argmin(np.abs(np.asarray(self.action_net) - action[1])))
        state_new_discrete = (np.argmin(np.abs(np.asarray(self.state_net) - newObs[0])), np.argmin(np.abs(np.asarray(self.state_net) - newObs[1])))

        self.num_visits[timestep, state_discrete[0], state_discrete[1], action_discrete[0], action_discrete[1]] += 1
        t = self.num_visits[timestep, state_discrete[0], state_discrete[1], action_discrete[0], action_discrete[1]]
        lr = (self.epLen + 1) / (self.epLen + t)
        bonus = self.scaling * np.sqrt(1 / t)

        if timestep == self.epLen-1:
            vFn = 0
        else:
            vFn = np.max(self.qVals[timestep+1, state_new_discrete[0], state_new_discrete[1], :,:])
        vFn = min(self.epLen, vFn)

        self.qVals[timestep, state_discrete[0], state_discrete[1], action_discrete[0], action_discrete[1]] = (1 - lr) * self.qVals[timestep, state_discrete[0], state_discrete[1], action_discrete[0], action_discrete[1]] + lr * (reward + vFn + bonus)

    def get_num_arms(self):
        ''' Returns the number of arms'''
        return self.epLen * len(self.state_net)**2 * len(self.action_net)**2

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
        state_discrete = (np.argmin(np.abs(np.asarray(self.state_net) - state[0])), np.argmin(np.abs(np.asarray(self.state_net) - state[1])))
        qFn = self.qVals[timestep, state_discrete[0], state_discrete[1], :, :]

        # Get indices in qVals matrix for where the q function is maximized - should be two of them
        action_1, action_2 = np.where(qFn == qFn.max())
        index = np.random.choice(len(action_1))
        # from the two indices - return a tuple of the two actions for the action_net at those indices
        # self.action_net[i], self.action_net[j]
        return self.action_net[action_1[index]], self.action_net[action_2[index]] ##need to change!

    def pick_action(self, state, step):
        action = self.greedy(state, step)
        return action

class eNetPendulum(agent.FiniteHorizonAgent):

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

        self.qVals = np.ones([self.epLen, len(self.state_net), len(self.state_net),len(self.state_net), len(self.action_net)], dtype=np.float32) * self.epLen
        self.num_visits = np.zeros([self.epLen, len(self.state_net), len(self.state_net), len(self.state_net), len(self.action_net)], dtype=np.float32)

        '''
            Resets the agent by overwriting all of the estimates back to zero
        '''
    def reset(self):
        self.qVals = np.ones([self.epLen, len(self.state_net), len(self.state_net),len(self.state_net), len(self.action_net)], dtype=np.float32) * self.epLen
        self.num_visits = np.zeros([self.epLen, len(self.state_net), len(self.state_net), len(self.state_net), len(self.action_net)], dtype=np.float32)

        '''
            Adds the observation to records by using the update formula
        '''
    def update_obs(self, obs, action, reward, newObs, timestep):
        '''Add observation to records'''

        # returns the discretized state and action location
        state_discrete = (np.argmin(np.abs(np.asarray(self.state_net) - obs[0])), np.argmin(np.abs(np.asarray(self.state_net) - obs[1])), np.argmin(np.abs(np.asarray(self.state_net) - obs[2])))
        action_discrete = np.argmin(np.abs(np.asarray(self.action_net) - action))
        state_new_discrete = (np.argmin(np.abs(np.asarray(self.state_net) - newObs[0])), np.argmin(np.abs(np.asarray(self.state_net) - newObs[1])), np.argmin(np.abs(np.asarray(self.state_net) - newObs[2])))

        self.num_visits[timestep, state_discrete[0], state_discrete[1], state_discrete[2], action_discrete] += 1
        t = self.num_visits[timestep, state_discrete[0], state_discrete[1], state_discrete[2], action_discrete]
        lr = (self.epLen + 1) / (self.epLen + t)
        bonus = self.scaling * np.sqrt(1 / t)

        if timestep == self.epLen-1:
            vFn = 0
        else:
            vFn = max(self.qVals[timestep+1, state_new_discrete[0], state_new_discrete[1], state_new_discrete[2], :])
        vFn = min(self.epLen, vFn)
        self.qVals[timestep, state_discrete[0], state_discrete[1], state_discrete[2], action_discrete] = (1 - lr) * self.qVals[timestep, state_discrete[0], state_discrete[1], state_discrete[2], action_discrete] + lr * (reward + vFn + bonus)

    def get_num_arms(self):
        ''' Returns the number of arms'''
        return self.epLen * len(self.state_net)**3 * len(self.action_net)

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
        state_discrete = (np.argmin(np.abs(np.asarray(self.state_net) - state[0])), np.argmin(np.abs(np.asarray(self.state_net) - state[1])), np.argmin(np.abs(np.asarray(self.state_net) - state[2])))
        qFn = self.qVals[timestep, state_discrete[0], state_discrete[1], state_discrete[2], :]
        # Get indices in qVals matrix for where the q function is maximized - should be two of them
        action = np.where(qFn == qFn.max())[0]
        index = np.random.choice(len(action))
        # from the two indices - return a tuple of the two actions for the action_net at those indices
        return self.action_net[action[index]]

    def pick_action(self, state, step):
        action = self.greedy(state, step)
        return action
