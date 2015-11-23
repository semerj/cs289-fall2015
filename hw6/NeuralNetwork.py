import numpy as np
from scipy.special import expit


class NeuralNetwork(object):

    def __init__(self, loss='cross_entropy', bias=0, n_hidden=200, n_input=784,
                 n_output=10, mu=0, sd=0.01):
        self.bias = bias
        self.mu = mu
        self.sd = sd
        self.loss = loss
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.W1 = self._init_weights(self.n_input, self.n_hidden)
        self.W2 = self._init_weights(self.n_hidden, self.n_output)

    def _init_weights(self, n, d):
        return np.random.normal(self.mu, self.sd, (n, d))

    def _bias(self, X, axis):
        if axis == 'column':
            b = np.ones((X.shape[0], 1))
            return np.hstack((b, X))
        elif axis == 'row':
            b = np.ones((1, X.shape[1]))
            return np.vstack((b, X))

    def _forward_pass(self, X):
        # hidden layer
        self.z2 = X.dot(self.W1)         # N x H = N x D * D x H
        self.a2 = np.tanh(self.z2)       # N x H
        # output layer
        self.z3 = self.a2.dot(self.W2)   # N x O = N x H * H x O
        h = expit(self.z3)               # N x O
        return h

    def _sigmoid_prime(self, z):
        s = expit(z)
        return s*(1-s)

    def _tanh_prime(self, z):
        return 1-np.tanh(z)**2

    def _mean_squared_error(self, y, h):
        J = 0.5*np.sum((y-h)**2)
        return J

    def _mean_squared_prime(self, y, h):
        dJdh = -(y-h)
        return dJdh

    def _cross_entropy_error(self, y, h):
        J = -np.sum(y*np.log(h) + (1-y)*np.log(1-h))
        return J

    def _cross_entropy_prime(self, y, h):
        dJdh = (1-y)/(1-h) - y/h
        return dJdh

    def _backpropagation(self, X, y, h):
        if self.loss == 'mean_squared':
            dJdh = self._mean_squared_prime(y, h)
        elif self.loss == 'cross_entropy':
            dJdh = self._cross_entropy_prime(y, h)
        delta3 = dJdh*self._sigmoid_prime(self.z3)
        dJdW2 = self.a2.T.dot(delta3)
        # print("delta3: ", delta3.shape)
        # print("a2.T    ", self.a2.T.shape)
        # print("dJdW2   ", dJdW2.shape)
        # print("W2:     ", self.W2.shape)
        # print("W1:     ", self.W1.shape)
        delta2 = delta3.dot(self.W2.T)*self._tanh_prime(self.z2)
        # print("delta2: ", delta2.shape)
        dJdW1 = X.T.dot(delta2)
        # print("dJdW1:  ", dJdW1.shape)
        return dJdW1, dJdW2

    def fit(self, X, y, iterations, eta=0.001, alpha=0.01, batch_size=500,
            compute_at_iter=100):
        n_obs = X.shape[0]
        y_argmax = np.argmax(y, axis=1)
        delta_W1_old, delta_W2_old = 0, 0
        loss, accuracy = [], []

        for i in range(iterations):
            idx = np.random.randint(0, n_obs, batch_size)
            X_batch = X[idx]
            y_batch = y[idx]
            h = self._forward_pass(X_batch)
            # print(i, "iter", np.isfinite(h).all())
            dJdW1, dJdW2 = self._backpropagation(X_batch, y_batch, h)

            # compute loss and accuracy
            if i % compute_at_iter == 0:
                h_all = self._forward_pass(X)
                if self.loss == 'mean_squared':
                    loss.append(self._mean_squared_error(y, h_all))
                elif self.loss == 'cross_entropy':
                    loss.append(self._cross_entropy_error(y, h_all))
                y_hat = np.argmax(h_all, axis=1)
                accuracy.append(sum(y_hat == y_argmax)/n_obs)

            # update weights
            delta_W1 = eta*dJdW1
            delta_W2 = eta*dJdW2
            self.W1 -= delta_W1 + (alpha*delta_W1_old)
            self.W2 -= delta_W2 + (alpha*delta_W2_old)
            delta_W1_old = delta_W1
            delta_W2_old = delta_W2

        return loss, accuracy

    def predict(self, X_test):
        h = self._forward_pass(X_test)
        return np.argmax(h, axis=1)
