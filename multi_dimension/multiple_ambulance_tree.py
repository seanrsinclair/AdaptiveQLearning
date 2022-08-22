import numpy as np

''' Implementation of a tree structured used in the Adaptive Discretization Algorithm for multiple ambulances'''

''' First defines the node class by storing all relevant information'''


class MultipleAmbulanceNode():
    def __init__(self, qVal, num_visits, num_splits, state_val, action_val, radius):
        '''args:
        qVal - estimate of the q value
        num_visits - number of visits to the node or its ancestors
        num_splits - number of times the ancestors of the node has been split
        state_val - value of state at center
        action_val - value of action at center
        radius - radius of the node '''
        self.qVal = qVal
        self.num_visits = num_visits
        self.num_splits = num_splits
        self.state_val = state_val
        self.action_val = action_val
        self.radius = radius
        self.flag = False
        self.children = None

    def split_node(self):
        child_1 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                          (self.state_val[0] - self.radius / 2, self.state_val[1] - self.radius / 2),
                                          (self.action_val[0] - self.radius / 2, self.action_val[1] - self.radius / 2),
                                          self.radius * (1 / 2))
        child_2 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                          (self.state_val[0] - self.radius / 2, self.state_val[1] - self.radius / 2),
                                          (self.action_val[0] - self.radius / 2, self.action_val[1] + self.radius / 2),
                                          self.radius * (1 / 2))
        child_3 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                          (self.state_val[0] - self.radius / 2, self.state_val[1] - self.radius / 2),
                                          (self.action_val[0] + self.radius / 2, self.action_val[1] - self.radius / 2),
                                          self.radius * (1 / 2))
        child_4 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                          (self.state_val[0] - self.radius / 2, self.state_val[1] - self.radius / 2),
                                          (self.action_val[0] + self.radius / 2, self.action_val[1] + self.radius / 2),
                                          self.radius * (1 / 2))
        child_5 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                          (self.state_val[0] - self.radius / 2, self.state_val[1] + self.radius / 2),
                                          (self.action_val[0] - self.radius / 2, self.action_val[1] - self.radius / 2),
                                          self.radius * (1 / 2))
        child_6 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                          (self.state_val[0] - self.radius / 2, self.state_val[1] + self.radius / 2),
                                          (self.action_val[0] - self.radius / 2, self.action_val[1] + self.radius / 2),
                                          self.radius * (1 / 2))
        child_7 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                          (self.state_val[0] - self.radius / 2, self.state_val[1] + self.radius / 2),
                                          (self.action_val[0] + self.radius / 2, self.action_val[1] - self.radius / 2),
                                          self.radius * (1 / 2))
        child_8 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                          (self.state_val[0] - self.radius / 2, self.state_val[1] + self.radius / 2),
                                          (self.action_val[0] + self.radius / 2, self.action_val[1] + self.radius / 2),
                                          self.radius * (1 / 2))
        child_9 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                          (self.state_val[0] + self.radius / 2, self.state_val[1] - self.radius / 2),
                                          (self.action_val[0] - self.radius / 2, self.action_val[1] - self.radius / 2),
                                          self.radius * (1 / 2))
        child_10 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                           (self.state_val[0] + self.radius / 2, self.state_val[1] - self.radius / 2),
                                           (self.action_val[0] - self.radius / 2, self.action_val[1] + self.radius / 2),
                                           self.radius * (1 / 2))
        child_11 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                           (self.state_val[0] + self.radius / 2, self.state_val[1] - self.radius / 2),
                                           (self.action_val[0] + self.radius / 2, self.action_val[1] - self.radius / 2),
                                           self.radius * (1 / 2))
        child_12 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                           (self.state_val[0] + self.radius / 2, self.state_val[1] - self.radius / 2),
                                           (self.action_val[0] + self.radius / 2, self.action_val[1] + self.radius / 2),
                                           self.radius * (1 / 2))
        child_13 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                           (self.state_val[0] + self.radius / 2, self.state_val[1] + self.radius / 2),
                                           (self.action_val[0] - self.radius / 2, self.action_val[1] - self.radius / 2),
                                           self.radius * (1 / 2))
        child_14 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                           (self.state_val[0] + self.radius / 2, self.state_val[1] + self.radius / 2),
                                           (self.action_val[0] - self.radius / 2, self.action_val[1] + self.radius / 2),
                                           self.radius * (1 / 2))
        child_15 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                           (self.state_val[0] + self.radius / 2, self.state_val[1] + self.radius / 2),
                                           (self.action_val[0] + self.radius / 2, self.action_val[1] - self.radius / 2),
                                           self.radius * (1 / 2))
        child_16 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                           (self.state_val[0] + self.radius / 2, self.state_val[1] + self.radius / 2),
                                           (self.action_val[0] + self.radius / 2, self.action_val[1] + self.radius / 2),
                                           self.radius * (1 / 2))

        self.children = [child_1, child_2, child_3, child_4, child_5, child_6, child_7, child_8, child_9, child_10,
                         child_11, child_12, child_13, child_14, child_15, child_16]
        return self.children


'''The tree class consists of a hierarchy of nodes for multiple ambulances'''


class MultipleAmbulanceTree():
    # Defines a tree by the number of steps for the initialization
    def __init__(self, epLen):
        self.head = MultipleAmbulanceNode(epLen, 0, 0, (0.5, 0.5), (0.5, 0.5), 0.5)
        self.epLen = epLen
        self.numCalled = 0

    def get_head(self):
        return self.head

        # Recursive method which gets number of subchildren
        def get_num_balls(self, node):
            num_balls = 0
            if node.children is None:
                return 1
            else:
                for child in node.children:
                    num_balls += self.get_num_balls(child)
            return num_balls

        def get_number_of_active_balls(self):
            return self.get_num_balls(self.head)

    def get_active_ball_recursion(self, state, node):
        # If the node doesn't have any children, then the largest one
        # in the subtree must be itself
        self.numCalled+=1
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

    def get_active_ball(self, state):
        active_node, qVal = self.get_active_ball_recursion(state, self.head)
        return active_node, qVal

    # Helper method which checks if a state is within the node
    def state_within_node(self, state, node):
        return max(np.abs(state[0] - node.state_val[0]), np.abs(state[1] - node.state_val[1])) <= node.radius
