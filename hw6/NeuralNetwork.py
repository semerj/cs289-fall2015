import numpy as np
from scipy.special import expit


class NeuralNetwork(object):

    def __init__(self, cost='mean_squared', n_hidden=200, n_input=784, n_output=10):
        self.cost = cost
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.W1 = self._init_weights(self.n_input, self.n_hidden)
        self.W2 = self._init_weights(self.n_hidden, self.n_output)

    def _init_weights(self, n, d):
        return np.random.normal(0, 0.01, (n, d))

    # def _add_bias(self, X, axis):
    #     if axis == 'column':
    #         b = np.ones((X.shape[0], 1))
    #         return np.hstack((X, b))
    #     elif axis == 'row':
    #         b = np.ones((1, X.shape[1]))
    #         return np.vstack((X, b))

    def _forward_pass(self, X):
        # input layer
        # self.a1 = self._add_bias(X, axis='column')

        # hidden layer
        self.z2 = X.dot(self.W1) # total weighted sum of inputs
        self.a2 = np.tanh(self.z2) # hidden layer activation function
        # self.a2 = self._add_bias(self.a2, axis='column')

        # output layer
        self.z3 = self.a2.dot(self.W2) # total weighted sum of hidden inputs
        h = expit(self.z3) # output layer activation function
        return h

    def _sigmoid_prime(self, z):
        s = expit(z)
        return s*(1-s)

    def _tanh_prime(self, z):
        return 1 - np.tanh(z)**2

    def _mean_squared_error(self, X, y, h):
        J = 0.5*np.sum((y - h)**2)
        return J

    def _mean_squared_prime(self, y, h):
        dJdh = -(y-h)
        return dJdh

    def _cross_entropy_error(self, X, y, h):
        J = -np.sum(y*np.log(h) + (1-y)*np.log(1-h))
        return J

    def _cross_entropy_prime(self, y, h):
        dJdh = (1-y)/(1-h) - y/h
        return dJdh

    def _backpropagation(self, X, y, h):
        if self.cost == 'mean_squared':
            dJdh = self._mean_squared_prime(y, h)
        elif self.cost == 'cross_entropy':
            dJdh = self._cross_entropy_prime(y, h)

        delta3 = dJdh*self._sigmoid_prime(self.z3)
        dJdW2 = self.a2.T.dot(delta3)
        delta2 = delta3.dot(self.W2.T)*self._tanh_prime(self.z2)
        dJdW1 = X.T.dot(delta2)
        return dJdW1, dJdW2

    def fit(self, X, y, epochs, eta=0.001, alpha=0.01, batch_size=200, iter_size=100):
        delta_W1_prev = 0
        delta_W2_prev = 0
        n = X.shape[0]
        passes = epochs * iter_size
        cost_list = []
        acc_list = []

        for i in range(passes):
            idx = np.random.randint(0, n, batch_size)
            X_batch = X[idx]
            y_batch = y[idx]
            h = self._forward_pass(X_batch)
            dJdW1, dJdW2 = self._backpropagation(X_batch, y_batch, h)

            # compute cost and accuracy
            if i % iter_size == 0:
                if self.cost == 'mean_squared':
                    cost_list.append(self._mean_squared_error(X_batch, y_batch, h))
                elif self.cost == 'cross_entropy':
                    cost_list.append(self._cross_entropy_error(X_batch, y_batch, h))
                pred = self.predict(X)
                acc = sum(pred == np.argmax(y, axis=1))/n
                acc_list.append(acc)

            # update weights
            delta_W1 = eta * dJdW1
            delta_W2 = eta * dJdW2
            self.W1 -= (delta_W1 + (alpha * delta_W1_prev))
            self.W2 -= (delta_W2 + (alpha * delta_W2_prev))
            delta_W1_prev = delta_W1
            delta_W2_prev = delta_W2

        return cost_list, acc_list

    def predict(self, X_test):
        h = self._forward_pass(X_test)
        return np.argmax(h, axis=1)
