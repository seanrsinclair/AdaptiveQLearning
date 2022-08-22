import numpy as np
import multiple_ambulance_tree
from src import agent
# from multiple_ambulance_tree import MultipleAmbulanceTree

class MultipleAmbulanceAgent(agent.FiniteHorizonAgent):

    def __init__(self, epLen, numIters, scaling):
        '''args:
            epLen - number of steps per episode
            numIters - total number of iterations
            scaling - scaling parameter for UCB term
        '''
        self.epLen = epLen
        self.numIters = numIters
        self.scaling = scaling

        # List of tree's, one for each step
        self.tree_list = []

        # Makes a new partition for each step and adds it to the list of trees
        for h in range(epLen):
            tree = multiple_ambulance_tree.MultipleAmbulanceTree(epLen)
            self.tree_list.append(tree)

    def get_num_arms(self):
        total_size = 0
        for tree in self.tree_list:
            total_size += tree.get_number_of_active_balls()
        return total_size

    def pick_action(self, state, timestep):
        action = self.greedy(state, timestep)
        return action

    def split_ball(self, node):
        children = self.node.split_ball()


    def reset(self):
        # Resets the agent by setting all parameters back to zero
        self.tree_list = []
        for h in range(self.epLen):
            tree = multiple_ambulance_tree.MultipleAmbulanceTree(self.epLen)
            self.tree_list.append(tree)

    def update_obs(self, obs, action, reward, newObs, timestep):
        '''Add observation to records'''
        # Gets the active tree based on current timestep
        tree = self.tree_list[timestep]
        # Gets the active ball by finding the argmax of Q values of relevant

        active_node, _ = tree.get_active_ball(obs)

        if timestep == self.epLen - 1:
            vFn = 0
        else:
            # Gets the next tree to get the approximation to the value function
            # at the next timestep
            new_tree = self.tree_list[timestep + 1]
            new_active, new_q = new_tree.get_active_ball(newObs)
            vFn = min(self.epLen, new_q)
        # Updates parameters for the node
        active_node.num_visits += 1
        t = active_node.num_visits
        lr = (self.epLen + 1) / (self.epLen + t)
        bonus = self.scaling * np.sqrt(1 / t)
        active_node.qVal = (1 - lr) * active_node.qVal + lr * (reward + vFn + bonus)

        '''determines if it is time to split the current ball'''
        if t >= 4 ** active_node.num_splits:
            active_node.split_node()

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
        action_one = np.random.uniform(active_node.action_val[0] - active_node.radius, active_node.action_val[0] + active_node.radius)
        action_two = np.random.uniform(active_node.action_val[1] - active_node.radius, active_node.action_val[1] + active_node.radius)

        return action_one, action_two
