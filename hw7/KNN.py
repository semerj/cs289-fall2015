import numpy as np
from scipy.spatial.distance import cdist


class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        self.X = X

    def _compute_neighbor_avg(self, x_test):
        n = x_test.shape[0]
        distances = cdist(self.X, x_test.reshape((1,n)), 'euclidean')
        neighbor_ids = np.argsort(distances, axis=0)[:self.k].T[0]
        neighbor_scores = self.X[neighbor_ids]
        neighbor_avg = np.mean(neighbor_scores, axis=0)
        return neighbor_avg

    def predict(self, X_test):
        predictions = np.empty(X_test.shape)
        for i, x in enumerate(X_test):
            predictions[i] = self._compute_neighbor_avg(x)
        return predictions
