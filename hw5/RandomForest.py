import numpy as np
from DecisionTree import *


class RandomForest(object):
    def __init__(self, max_depth=10, min_obs=100, num_trees=20, num_features=10):
        self.max_depth = max_depth
        self.min_obs = min_obs
        self.num_trees = num_trees
        self.num_features = num_features
        self.trees = []

    def fit(self, data, label):
        num_samples, total_features = data.shape
        for tree_num in range(self.num_trees):
            # print("TREE:", tree_num)
            random_rows = np.random.randint(0, num_samples, num_samples)
            random_features = np.random.choice(total_features,
                                               self.num_features,
                                               replace=False)
            random_data = data[random_rows,:][:,random_features]
            random_labels = label[random_rows]
            dt = DecisionTree(self.max_depth, self.min_obs)
            dt.fit(random_data, random_labels)
            self.trees += [(random_features, dt)]

    def predict_proba(self, test_data):
        total_votes = []
        for feature_tree in self.trees:
            random_features, dtree = feature_tree
            test_subset_data = test_data[:,random_features]
            pred = dtree.predict_proba(test_subset_data)
            total_votes += [pred]
        proba = np.mean(np.array(total_votes).T, axis=1)
        return proba

    def predict(self, test_data):
        proba = self.predict_proba(test_data)
        return np.where(proba > 0.5, 0, 1)
