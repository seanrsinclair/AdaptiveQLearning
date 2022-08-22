import numpy as np
from src import agent
from tree_model_based_multiple import Node, Tree


class AdaptiveModelBasedDiscretization(agent.FiniteHorizonAgent):

    def __init__(self, epLen, numIters, scaling, alpha, flag):
        '''args:
            epLen - number of steps per episode
            numIters - total number of iterations
            scaling - scaling parameter for UCB term
            alpha - parameter to add a prior to the transition kernels
        '''

        self.epLen = epLen
        self.numIters = numIters
        self.scaling = scaling
        self.alpha = alpha

        self.flag = flag

        # List of tree's, one for each step
        self.tree_list = []

        # Makes a new partition for each step and adds it to the list of trees
        for h in range(epLen):
            # print(h)
            tree = Tree(epLen, self.flag)
            self.tree_list.append(tree)

    def reset(self):
        # Resets the agent by setting all parameters back to zero
        # List of tree's, one for each step
        self.tree_list = []

        # Makes a new partition for each step and adds it to the list of trees
        for h in range(self.epLen):
            tree = Tree(self.epLen, self.flag)
            self.tree_list.append(tree)

        # Gets the number of arms for each tree and adds them together
    def get_num_arms(self):
        total_size = 0
        for tree in self.tree_list:
            total_size += tree.get_number_of_active_balls()
        return total_size

    def update_obs(self, obs, action, reward, newObs, timestep):
        '''Add observation to records'''
        # print('Updating observations at step: ' + str(timestep))
        # print('Old state: ' + str(obs) + ' action: ' + str(action) + ' newState: ' + str(newObs))
        # print('Reward: ' + str(reward))
        # Gets the active trees based on current timestep

        ''' Gets the tree that was used at that specific timestep '''
        tree = self.tree_list[timestep]

        # Gets the active ball by finding the argmax of Q values of relevant
        active_node, _ = tree.get_active_ball(obs)

        # Increments the number of visits
        active_node.num_visits += 1
        active_node.num_unique_visits += 1
        t = active_node.num_unique_visits
        # print('Num visits: ' + str(t))

        # Update empirical estimate of average reward for that node
        active_node.rEst = ((t-1)*active_node.rEst + reward) / t
        # print('Mean reward: ' + str(active_node.rEst))

        # If it is not the last timestep - updates the empirical estimate
        # of the transition kernel based on the induced state partition at the next step
        if timestep != self.epLen - 1:
            next_tree = self.tree_list[timestep+1]
            # update transition kernel based off of new transition
            #print(active_node.pEst)
            #print('timestep' + str(timestep))
            new_obs_loc = np.argmin(np.max(np.abs(np.asarray(next_tree.state_leaves) - np.array(newObs)),axis=1))
            active_node.pEst[new_obs_loc] += 1
            # print('Updating transition estimates!')
            # print(active_node.pEst)
            # print(next_tree.state_leaves)

        '''determines if it is time to split the current ball'''
        if t >= 4**active_node.num_splits:
            # print('Splitting a ball!!!!')
            if timestep >= 1:
                children = tree.split_node(active_node, timestep, self.tree_list[timestep-1])
            else:
                children = tree.split_node(active_node, timestep, None)

    def update_policy(self, k):
        '''Update internal policy based upon records'''
        # print('#######################')
        # print('Recomputing estimates at the end of an episode')
        # print('#######################')

        # Solves the empirical Bellman equations
        for h in np.arange(self.epLen-1,-1,-1):
            # print('Estimates for step: ' + str(h))

            # Gets the current tree for this specific time step
            tree = self.tree_list[h]
            for node in tree.tree_leaves:
                # If the node has not been visited before - set its Q Value
                # to be optimistic
                if node.num_unique_visits == 0:
                    node.qVal = self.epLen
                else:
                    # Otherwise solve for the Q Values with the bonus term

                    # If h == H then the value function for the next step is zero
                    if h == self.epLen - 1:
                        # print(node.qVal)
                        # print(self.epLen)
                        # print(node.rEst)
                        node.qVal = min(node.qVal, self.epLen, node.rEst + self.scaling / np.sqrt(node.num_unique_visits))

                    else: # Gets the next tree to estimate the transition kernel
                        next_tree = self.tree_list[h+1]
                        vEst = np.dot((np.asarray(node.pEst)+self.alpha) / (np.sum(np.array(node.pEst))+len(next_tree.state_leaves)*self.alpha), next_tree.vEst)
                        node.qVal = min(node.qVal, self.epLen, node.rEst + vEst + self.scaling / np.sqrt(node.num_unique_visits))
                # print(node.state_val, node.action_val, node.qVal)
            # After updating the Q Value for each node - computes the estimate of the value function
            index = 0
            for state_val in tree.state_leaves:
                _, qMax = tree.get_active_ball(state_val)
                tree.vEst[index] = min(qMax, self.epLen, tree.vEst[index])
                index += 1
            # print('### PRINTING STATE LEAVES  AND VALUE ESTIMATES!')
            # print(tree.state_leaves)
            # print(tree.vEst)
            # print('#### DDONEE ###')

        self.greedy = self.greedy

        pass

    def split_ball(self, node):
        children = self.node.split_ball()
        pass

    def greedy(self, state, timestep, epsilon=0):
        '''
        Select action according to a greedy policy

        Args:
            state - int - current state
            timestep - int - timestep *within* episode

        Returns:
            action - int
        '''
        # Considers the partition of the space for the current timestep
        tree = self.tree_list[timestep]

        # Gets the selected ball
        active_node, qVal = tree.get_active_ball(state)

        # Picks an action uniformly in that ball
        action_one = np.random.uniform(active_node.action_val[0] - active_node.radius,
                                       active_node.action_val[0] + active_node.radius)
        action_two = np.random.uniform(active_node.action_val[1] - active_node.radius,
                                       active_node.action_val[1] + active_node.radius)

        return action_one,action_two

    def pick_action(self, state, timestep):
        action = self.greedy(state, timestep)
        return action
