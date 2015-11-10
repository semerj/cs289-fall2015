import numpy as np


class Node(object):
    def __init__(self, data, label, node_type=None, split_variable=None, split_value=None):
        self.data = data
        self.label = label
        self.node_type = node_type
        self.split_variable = split_variable
        self.split_value = split_value
        self.left_child = None
        self.right_child = None

    def __repr__(self):
        return self.node_type

    def get_prob(self):
        '''return probability of 0 in node data'''
        ones = np.count_nonzero(self.label)
        total = len(self.label)
        return (total-ones)/total

    def traverse(self, row):
        if row[int(self.split_variable)] <= self.split_value:
            if self.left_child.node_type == 'leaf':
                return self.get_prob()
            return self.left_child.traverse(row)
        else:
            if self.right_child.node_type == 'leaf':
                return self.get_prob()
            return self.right_child.traverse(row)
