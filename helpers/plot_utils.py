from matplotlib import pyplot as plt
import numpy as np


def get_axes_lims(X):
    x_min = X[:, 0].min() - (X[:, 0].mean() * 0.5)
    x_max = X[:, 0].max() + (X[:, 0].mean() * 0.5)

    y_min = X[:, 1].min() - (X[:, 1].mean() * 0.5)
    y_max = X[:, 1].max() + (X[:, 1].mean() * 0.5)

    return x_min, x_max, y_min, y_max


def plot_data(X, labels=None, centers=None, title='Data plotted', savename=None):
    x_min, x_max, y_min, y_max = get_axes_lims(X)

    # plot result
    f, axes = plt.subplots(1, 1, figsize=(11, 5))
    axes.set_xlim(x_min, x_max)
    axes.set_ylim(y_min, y_max)

    if not labels is None:
        axes.scatter(X[:, 0], X[:, 1], c=labels, alpha=.4)
    else:
        axes.scatter(X[:, 0], X[:, 1], alpha=.4)
    if not centers is None:
        # axes[0].scatter(centers[:,0], centers[:,1], marker="+", s=500, c='b')
        axes.scatter(centers[:, 0], centers[:, 1], marker="+", s=500, c='black')

    plt.title(title)
    if not savename is None:
        plt.savefig(savename)
    plt.show()


def plot_federated_clusters(global_learner, xlims, title='', savename=None):
    # plot result
    local_learners = global_learner.local_learners

    x_min, x_max, y_min, y_max = xlims

    f, axes = plt.subplots(len(local_learners), 1, figsize=(11, 5 * len(local_learners)))
    f.tight_layout(h_pad=6)

    for i, local_learner in enumerate(local_learners):
        cluster_labels = np.argmax(local_learner.cluster_assignment(), axis=1)
        axes[i].scatter(local_learner.client_data[:, 0], local_learner.client_data[:, 1],
                        c=cluster_labels, alpha=.4)
        axes[i].scatter(local_learner.centers[:, 0], local_learner.centers[:, 1], marker='+', s=500, c='black')
        axes[i].set_title('Federated clusters for client {0}.'.format(i + 1))
        axes[i].set_xlim(x_min, x_max)
        axes[i].set_ylim(y_min, y_max)

    plt.title(title)
    if not savename is None:
        plt.savefig(savename)
    plt.show()
