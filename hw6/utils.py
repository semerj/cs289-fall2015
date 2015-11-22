import numpy as np
import matplotlib.pylab as plt


def accuracy(pred, true_label):
    acc = np.sum(pred == true_label)/len(pred)
    return acc

def save_prediction(array, fname):
    pred = np.vstack((np.arange(1,len(array)+1), array)).T
    np.savetxt(fname=fname, X=pred, fmt='%d', delimiter=',',
               header='Id,Category', comments='')

def make_plot(data, labels, title, ylim, xlab, ylab):
    plt.figure(figsize=(10, 6))
    for k, v in enumerate(data):
        plt.plot(data[k], label=labels[k])
    plt.legend(frameon=False, loc='center right')
    plt.ylim([0, ylim])
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    plt.title(title)
    plt.show()
