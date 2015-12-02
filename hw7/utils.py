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

def validation_accuracy(validation, pred_matrix):
    results = []
    for user, item, rating in validation:
        result = pred_matrix[user-1, item-1] == rating
        results.append(result)
    return np.sum(results)/len(results)

def reshape_array(array, i):
    new = np.zeros(100)
    user = array[array[:,0] == i,:]
    joke_id = (user[:,1] - 1).astype(np.int)
    ratings = user[:,2]
    new[joke_id] = ratings
    return new.reshape(1,100)

def reshape_long_to_wide(data, user_ids):
    n_users = user_ids.shape[0]
    wide_matrix = np.empty((n_users, 100))
    for i, user in enumerate(user_ids):
        result = reshape_array(data, user)
        wide_matrix[i] = result
    return wide_matrix

def wide_to_long(predictions, query):
    scores = []
    for row in query:
        user, item = row[1]-1, row[2]-1
        scores.append(predictions[user, item])
    return np.array(scores, dtype=np.int)
