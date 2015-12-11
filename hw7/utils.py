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

def validation_accuracy_and_MSE(validation, pred_matrix):
    tf_results = []
    mse_results = []
    for user, item, rating in validation:
        tf = pred_matrix[user-1, item-1] == rating
        tf_results.append(tf)
        diff = (pred_matrix[user-1, item-1] - rating)**2
        mse_results.append(diff)
    acc = np.sum(tf_results)/len(tf_results)
    mse = np.mean(mse_results)
    return acc, mse

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

def MSE(data_matrix, pred_matrix):
    total = []
    obs = []
    for data, pred in zip(data_matrix, pred_matrix):
        ind = np.where(~np.isnan(data))
        row_n = len(ind[0])
        row_total = sum((data[ind] - pred[ind])**2)
        obs.append(row_n)
        total.append(row_total)
    return np.sum(total)/np.sum(obs)
