import matplotlib.pylab as plt
import numpy as np


def plot_kmeans(digits, nrows, ncols, figsize):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for k, (data, ax) in enumerate(zip(digits, axes.flat)):
        im = ax.imshow(data.reshape(28,28), cmap='Greys')
        ax.set_title('K: {}'.format(k))
        ax.axis('off')
        fig.tight_layout()
    plt.show()

def kmeans_cost(model):
    cost = []
    for k in range(model.k):
        min_values = np.min(model.distances, axis=0)
        k_norm = np.linalg.norm(min_values[model.assignments == k])
        cost.append(k_norm)
    return np.mean(cost)

def validation_accuracy(validation, joke_scores):
    accuracy = []
    for row in validation:
        row_acc = np.sum(joke_scores[int(row[1]) - 1,:] == int(row[2]))/100
        accuracy.append(row_acc)
    return np.mean(accuracy)

def reshape_array(array, i):
    new = np.zeros(100)
    user = array[array[:,0] == i,:]
    joke_id = (user[:,1] - 1).astype(np.int)
    ratings = user[:,2]
    new[joke_id] = ratings
    return new.reshape(1,100)

def reshape_validation(data, user_ids):
    n_users = user_ids.shape[0]
    matrix = np.empty((n_users, 100))
    for i, user in enumerate(user_ids):
        result = reshape_array(data, user)
        matrix[i] = result
    return matrix
