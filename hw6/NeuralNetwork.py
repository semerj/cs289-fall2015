import numpy as np
from scipy.special import expit


class NeuralNetwork(object):

    def __init__(self, n_hidden=200, n_input=784, n_output=10):
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.W1 = self.initWeights(self.n_input + 1, self.n_hidden)
        self.W2 = self.initWeights(self.n_hidden + 1, self.n_output)

    def initWeights(self, n, d):
        return np.random.normal(0, 0.01, (n, d))

    def addBias(self, X):
        b = np.ones((X.shape[0], 1))
        Xb = np.hstack((X, b))
        return Xb

    def forwardPropogation(self, X):
        '''propagate inputs through network'''
        # layer input-hidden layer
        Xb = self.addBias(X)
        self.z2 = self.addBias(Xb.dot(self.W1)) # total weighted sum of inputs
        self.a2 = np.tanh(self.z2) # activation function
        # layer hidden-output layer
        self.z3 = self.a2.dot(self.W2) # total weighted sum of hidden inputs
        yHat = expit(self.z3) # activation function
        return yHat

    def backPropogation(self, X):
        pass

    def sigmoidPrime(self, z):
        '''gradient of sigmoid/expit'''
        s = expit(z)
        return s*(1-s)

    def tanhPrime(self, z):
        return 1 - np.tanh(z)**2

    def meanSquaredError(self, X, y):
        '''compute cost (scalar) for given X,y use weights already stored in class'''
        self.yHat = self.forwardPropogation(X)
        J = 0.5*np.sum((y - self.yHat)**2)
        return J

    def meanSquaredErrorPrime(self, X, y):
        '''compute derivative with respect to W1 and W2 for a given X and y'''
        self.yHat = self.forwardPropogation(X)
        delta3 = -(y-self.yHat)*self.sigmoidPrime(self.z3)
        dJdW2 = self.a2.T.dot(delta3)
        delta2 = delta3.dot(self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = X.T.dot(delta2)
        return dJdW1, dJdW2

    def crossEntropyError(self, X, y):
        self.yHat = self.forwardPropogation(X)
        J = -np.sum(y*np.log(self.yHat) + (1-self.yHat)*np.log(1-self.yHat))
        return J

    def crossEntropyErrorPrime(self, X, y):
        pass

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.meanSquaredErrorPrime(X, y)
        return np.hstack((dJdW1.ravel(), dJdW2.ravel()))

    def checkGradient(self):
        pass

    def fit(self, X, y, eta):
        '''while (some stopping criteria):
            a. pick one data point (x,y) at random from the training set
            b. perform forward pass (computing necessary values for gradient descent update)
            c. perform backward pass (again computing necessary values)
            d. perform stochastic gradient descent update
        return W^1, W^2
        '''
        pass

    def predict(self, X_test):
        pass
