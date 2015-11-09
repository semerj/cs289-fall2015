from Node import *
import numpy as np
from collections import namedtuple


class DecisionTree(object):
    def __init__(self, max_depth=5, min_obs=100):
        self.number_of_nodes = 1
        self.max_depth = max_depth
        self.min_obs = min_obs

    def entropy(self, class_probs):
        '''
        class_probs:
            {'left': Prob(1, 2), 'right': Prob(1, 2)}
        '''
        Prob_0, Prob_1 = class_probs.values()
        p0 = Prob_0.num/Prob_0.den
        p1 = 1 - p0

        if p0 == 0:
            return -p1*np.log2(p1)
        elif p1 == 0:
            return -p0*np.log2(p0)
        else:
            return -p0*np.log2(p0)-p1*np.log2(p1)

    def information_gain(self, parent_entropy, child_prob_tuple):
        '''
        parent_entropy:
            float
        child_prob_tuple:
            (class_probs, class_probs)
        '''
        left, right = child_prob_tuple
        left_den = [x for x in left.values()][0].den
        right_den = [x for x in right.values()][0].den
        child_den = left_den + right_den
        left_weight = left_den/child_den * self.entropy(left)
        right_weight = right_den/child_den * self.entropy(right)
        return parent_entropy - (left_weight + right_weight)/2

    def feature_split(self, label, feature, feature_name, parent_entropy):
        '''Find best value to split

        return:
            [('feature1', infogain), ('feature2', infogain), ...]
        '''
        Prob = namedtuple('prob', ['num', 'den'])
        uniq_values = np.unique(feature)

        feature_scores = []
        for val in uniq_values:
            l0_counts = sum(label[feature <= val] == 0)
            l1_counts = sum(label[feature <= val] == 1)

            r0_counts = sum(label[feature > val] == 0)
            r1_counts = sum(label[feature > val] == 1)

            l_den = l0_counts + l1_counts
            r_den = r0_counts + r1_counts

            if l_den == 0 or r_den == 0:
                break

            split_name = "{} <= {}".format(feature_name, val)

            # add a 1 if split results in too few observations
            if min(l_den, r_den) <= self.min_obs:
                feature_scores.append((None, 0))

            l_prob = {
                '0': Prob(l0_counts, l_den),
                '1': Prob(l1_counts, l_den)
            }

            r_prob = {
                '0': Prob(r0_counts, r_den),
                '1': Prob(r1_counts, r_den)
            }

            child_prob_tuple = (l_prob, r_prob)
            info_gain = self.information_gain(parent_entropy, child_prob_tuple)
            # print(split_name, info_gain)
            feature_scores.append((split_name, info_gain))

        return feature_scores

    def best_split(self, data, label):
        '''
        returns:
            ("{feature} <= {value to split on}", entropy)
        '''
        Prob = namedtuple('prob', ['num', 'den'])
        _, p_counts = np.unique(label, return_counts=True)
        den = sum(p_counts)
        parent_prob = {
            '0': Prob(p_counts[0], den),
            '1': Prob(p_counts[1], den)
        }
        parent_entropy = self.entropy(parent_prob)

        feature_scores = []
        for i, feature in enumerate(data.T):
            feature_scores += self.feature_split(label, feature, i, parent_entropy)

        print(feature_scores)
        info_gain = max(feature_scores, key=lambda x: x[1])
        print("BEST INFO GAIN:", info_gain)
        return info_gain

    def split_data(self, node):
        '''Find best feature and value to split on in entire data set
        returns:
            (Node(), Node())
        '''
        data = node.data
        label = node.label

        split_name, entropy = self.best_split(data, label)
        feature = split_name.split()[0]
        split_value = float(split_name.split()[2])

        index = data[:, feature] <= split_value
        data_left,  label_left  = data[index],  label[index]
        data_right, label_right = data[~index], label[~index]

        left_child  = Node(data_left,  label_left,
                           split_variable=feature,
                           split_value=split_value)

        right_child = Node(data_right, label_right,
                           split_variable=feature,
                           split_value=split_value)

        return left_child, right_child

    def grow_tree(self, data, label, depth=0):
        if len(np.unique(label)) == 1:
            # print("PURE BRANCH")
            return Node(data=None, label=None, node_type='leaf')

        elif self.max_depth == depth:
            # print("MAX DEAPTH:", self.max_depth)
            return Node(data=None, label=None, node_type='leaf')

        elif self.min_obs >= data.shape[0]:
            # print("MIN OBSERVATIONS")
            return Node(data=None, label=None, node_type='leaf')

        else:
            if self.number_of_nodes == 1:
                node_type = 'root'
            else:
                node_type = 'node'

            tree = Node(data, label, node_type=node_type)
            left_child, right_child = self.split_data(tree)
            tree.split_variable = left_child.split_variable
            tree.split_value = left_child.split_value

            # figure out how to count nodes?
            self.number_of_nodes += 2

            # print("split on feature {} at value {} and go left" \
            #         .format(tree.split_variable, tree.split_value))
            tree.left_child = self.grow_tree(left_child.data,
                                             left_child.label,
                                             depth + 1)

            # print("split on feature {} at value {} and go right" \
            #         .format(tree.split_variable, tree.split_value))
            tree.right_child = self.grow_tree(right_child.data,
                                              right_child.label,
                                              depth + 1)
            return tree

    def fit(self, data, label):
        self.tree = self.grow_tree(data, label)

    def predict_proba(self, test_data):
        return np.array([self.tree.traverse(row) for row in test_data])

    def predict(self, test_data):
        proba = self.predict_proba(test_data)
        return np.where(proba > 0.5, 0, 1)
