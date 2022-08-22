import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
''' Implementation of a tree structured used in the Adaptive Discretization Algorithm'''


''' First defines the node class by storing all relevant information'''
class Node():
    def __init__(self, qVal, rEst, pEst, num_visits, num_unique_visits, num_splits, state_val, action_val, radius):
        '''args:
        qVal - estimate of the q value
        num_visits - number of visits to the node or its ancestors
        num_splits - number of times the ancestors of the node has been split
        state_val - value of state at center
        action_val - value of action at center
        radius - radius of the node '''
        self.qVal = qVal
        self.rEst = rEst
        self.pEst = pEst
        self.num_visits = num_visits
        self.num_unique_visits = num_unique_visits
        self.num_splits = num_splits
        self.state_val = state_val
        self.action_val = action_val
        self.radius = radius
        self.children = None

        # Splits a node by covering it with four children, as here S times A is [0,1]^2
        # each with half the radius
    def split_node(self, flag, epLen):
        if flag == False:
            child_1 = Node(self.qVal, self.rEst, list.copy(self.pEst), self.num_visits, self.num_visits, self.num_splits+1, self.state_val+self.radius/2, self.action_val+self.radius/2, self.radius*(1/2))
            child_2 = Node(self.qVal, self.rEst, list.copy(self.pEst), self.num_visits, self.num_visits, self.num_splits+1, self.state_val+self.radius/2, self.action_val-self.radius/2, self.radius*(1/2))
            child_3 = Node(self.qVal, self.rEst, list.copy(self.pEst), self.num_visits, self.num_visits, self.num_splits+1, self.state_val-self.radius/2, self.action_val+self.radius/2, self.radius*(1/2))
            child_4 = Node(self.qVal, self.rEst, list.copy(self.pEst), self.num_visits, self.num_visits, self.num_splits+1, self.state_val-self.radius/2, self.action_val-self.radius/2, self.radius*(1/2))
        else:
            child_1 = Node(epLen,0, np.zeros(len(self.pEst)).tolist(), self.num_visits, 0, self.num_splits+1, self.state_val+self.radius/2, self.action_val+self.radius/2, self.radius*(1/2))
            child_2 = Node(epLen,0, np.zeros(len(self.pEst)).tolist(), self.num_visits, 0, self.num_splits+1, self.state_val+self.radius/2, self.action_val-self.radius/2, self.radius*(1/2))
            child_3 = Node(epLen,0, np.zeros(len(self.pEst)).tolist(), self.num_visits, 0, self.num_splits+1, self.state_val-self.radius/2, self.action_val+self.radius/2, self.radius*(1/2))
            child_4 = Node(epLen,0, np.zeros(len(self.pEst)).tolist(), self.num_visits, 0, self.num_splits+1, self.state_val-self.radius/2, self.action_val-self.radius/2, self.radius*(1/2))
        self.children = [child_1, child_2, child_3, child_4]
        return self.children


'''The tree class consists of a hierarchy of nodes'''
class Tree():
    # Defines a tree by the number of steps for the initialization
    def __init__(self, epLen, flag):
        self.head = Node(epLen, 0, [0], 0, 0, 0, .5, .5, .5)
        self.epLen = epLen
        self.flag = flag
        self.state_leaves = [.5] # [(.5, .5)]
        self.vEst = [self.epLen]
        self.tree_leaves = [self.head]

    # Returns the head of the tree
    def get_head(self):
        return self.head

    def split_node(self, node, timestep, previous_tree):
        children = node.split_node(self.flag, self.epLen)

        # Update the list of leaves in the tree
        self.tree_leaves.remove(node)
        for child in children:
            self.tree_leaves.append(child)


        # Determines if we also need to adjust the state_leaves and carry those
        # estimates down as well

        # Gets one of their state value
        child_1_state = children[0].state_val
        child_1_radius = children[0].radius

        # if np.min(np.max(np.abs(state_1 - child_state_1), np.abs(state_2 - child_state_2))) >= child_1_radius
        if np.min(np.abs(np.asarray(self.state_leaves) - child_1_state)) >= child_1_radius:
            # print('Adjusting the induced state partition')
            # print('Current node state: ' + str(node.state_val))
            # print('Child state: ' + str(child_1_state))
            # print('Current leaves: ' + str(self.state_leaves))

            parent = node.state_val
            # print('Getting parents index!')
            # print(self.state_leaves)
            parent_index = self.state_leaves.index(parent)
            parent_vEst = self.vEst[parent_index]

            self.state_leaves.pop(parent_index)
            self.vEst.pop(parent_index)

            # will be appending duplicate numbers here

            # self.state_leaves.append(child.state_val)

            # append(children[0].state_val(0), children[0].state_val(1))
            self.state_leaves.append(children[0].state_val)
            self.state_leaves.append(children[2].state_val)
            self.vEst.append(parent_vEst)
            self.vEst.append(parent_vEst)
            # print('Checking lengths: ')
            # print(len(self.state_leaves))
            # print(len(self.vEst))
            # Lastly we need to adjust the transition kernel estimates from the previous tree
            if timestep >= 1:
                previous_tree.update_transitions_after_split(parent_index, 2)


            # Need to remove parent's state value from state_leaves,
            # add in the state values for the children
            # copy over the estimate of the value function
            # also copy over the estimate of the transition function
        # print(self.state_leaves)
        return children

    def update_transitions_after_split(self, parent_index, num_children):
        # print('Adjusting transitions at previous timestep')
        # print('Number of leaves: ' + str(len(self.tree_leaves)))
        # print('Printing out length for each leaf!')
        # for node in self.tree_leaves:
        #     print(len(node.pEst))
        #     print(node.pEst)
        #     print(node)

        # print('Starting to adjust')
        for node in self.tree_leaves:
            # Adjust node.pEst
            # Should not just be copy pasting here....
            # print('Start adjust for a node!')
            # print(node)
            # print(len(node.pEst))
            # print(node.pEst)
            pEst_parent = node.pEst[parent_index]
            node.pEst.pop(parent_index)
            # print(len(node.pEst))
            # print('Adding on entries now!')
            for _ in range(num_children):
                node.pEst.append(pEst_parent / num_children)
            # print(len(node.pEst))
            # print(node.pEst)
            # print('Done')

    # Plot function which plots the tree on a graph on [0,1]^2 with the discretization
    def plot(self, fig):
        ax = plt.gca()
        self.plot_node(self.head, ax)
        plt.xlabel('State Space')
        plt.ylabel('Action Space')
        return fig

    # Recursive method which plots all subchildren
    def plot_node(self, node, ax):
        if node.children == None:
            # print('Child Node!')
            rect = patches.Rectangle((node.state_val - node.radius,node.action_val-node.radius),node.radius*2,node.radius*2,linewidth=1,edgecolor='k',facecolor='none')
            ax.add_patch(rect)
            # plt.text(node.state_val, node.action_val, np.around(node.qVal, 3))
        else:
            for child in node.children:
                self.plot_node(child, ax)


    # Recursive method which gets number of subchildren
    def get_num_balls(self, node):
        num_balls = 0
        if node.children == None:
            return 1
        else:
            for child in node.children:
                num_balls += self.get_num_balls(child)
        return num_balls

    def get_number_of_active_balls(self):
        return self.get_num_balls(self.head)

    # Recursive method which plots all subchildren
    def plot_q_help(self, node, ax, timestep,colors, min_q, max_q):
        if node.children == None:
            # print('Child Node!')
            rect = patches.Rectangle((node.state_val - node.radius, node.action_val - node.radius), node.radius * 2,
                                     node.radius * 2, linewidth=1, facecolor=colors(int(255*(node.qVal-min_q)/(max_q-min_q))), edgecolor='k')
            ax.add_patch(rect)
            # plt.text(node.state_val, node.action_val, np.around(node.qVal, 3))
        else:
            for child in node.children:
                self.plot_q_help(child, ax,timestep,colors, min_q, max_q)
    # Plot function which plots the tree on a graph on [0,1]^2 with the discretization

    
    def plot_q(self, fig, timestep):

        # Get maximum and minimum q_value across all leaves in that tree to be used
        # in setting the colour map

        min_q = self.epLen
        max_q = 0

        for node in self.tree_leaves:
            if node.qVal >= max_q:
                max_q = node.qVal
            elif node.qVal <= min_q:
                min_q = node.qVal

        ax = plt.gca()
        colors = plt.cm.RdYlGn
        self.plot_q_help(self.head, ax, timestep,colors, min_q, max_q)
        plt.xlabel('State Space')
        plt.ylabel('Action Space')
        plt.title('Heat Map of Q Values')
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=colors, norm=norm)
        sm.set_array([])
        return fig

    # A method which implements recursion and greedily selects the selected ball
    # to have the largest qValue and contain the state being considered

    def get_active_ball_recursion(self, state, node):
        # If the node doesn't have any children, then the largest one
        # in the subtree must be itself
        if node.children == None:
            return node, node.qVal
        else:
            # Otherwise checks each child node
            qVal = 0
            for child in node.children:

                # if the child node contains the current state
                if self.state_within_node(state, child):
                    # recursively check that node for the max one, and compare against all of them
                    new_node, new_qVal = self.get_active_ball_recursion(state, child)
                    if new_qVal >= qVal:
                        active_node, qVal = new_node, new_qVal
                else:
                    pass
        return active_node, qVal


    def get_active_ball(self, state):
        active_node, qVal = self.get_active_ball_recursion(state, self.head)
        return active_node, qVal

    # Helper method which checks if a state is within the node
    def state_within_node(self, state, node):
        return np.abs(state - node.state_val) <= node.radius
