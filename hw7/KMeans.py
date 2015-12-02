import numpy as np
from scipy.spatial.distance import cdist
np.random.seed(289)


class KMeans:

    def __init__(self, k, iterations):
        self.k = k
        self.iterations = iterations

    def fit(self, X):
        self.n = X.shape[0]
        idx = np.random.randint(self.n, size=self.k)
        self.kmeans = X[idx,:]
        self.distances = np.zeros((self.k, self.n))
        self.assignments = np.zeros((self.n,))

        for iteration in range(self.iterations):
            print('Iteration: {}'.format(iteration), end='\r')
            new_assignments = self.classify(X)
            # if np.all(self.assignments == new_assignments):
                # return
            self.assignments = new_assignments
            for i in range(self.k):
                points_i = X[self.assignments == i,:]
                if points_i.shape[0] > 0:
                    self.kmeans[i] = np.mean(points_i, axis=0)

    def classify(self, X):
        self.distances = cdist(self.kmeans, X, 'euclidean')
        return np.argmin(self.distances, axis=0)
