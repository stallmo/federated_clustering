import numpy as np
from scipy.spatial import distance


def distribute_data_among_clients(X, n_clients):
    """
    Distributes data <X> to <n_clients> randomly, but uniformly.

    :param X: numpy matrix. The data to distribute.
    :param n_clients: int. Number of clients
    :return: list of numpy arrays.
    """
    client_labels = np.random.randint(low=0, high=n_clients, size=X.shape[0])
    X_labeled = np.hstack((X, client_labels[:, None])).copy()
    X_labeled = X_labeled[X_labeled[:, -1].argsort()]
    clients_data = np.split(X_labeled[:, :-1], np.unique(X_labeled[:, -1], return_index=True)[1][1:])

    return clients_data


def calculate_overlap(X, gt_centers, gt_partitioning):
    """
    Calculates how big the overlap of a data partitioning is, i.e.
    That is, how many points belonging to one partition have a neighbour of another partition that is closer than its own partition's center.
    """
    X_partitioned = np.concatenate((X, gt_partitioning), axis=1)
    clust_inds = np.unique(gt_partitioning).astype(int)

    total_overlap_count = 0

    for clust_ind in clust_inds:
        same_clust_mask = X_partitioned[:, -1] == clust_ind

        # Prepare data
        X_same_clust = X[same_clust_mask]
        X_different_clust = X[~same_clust_mask]
        clust_center = gt_centers[clust_ind].reshape((-1, X_same_clust.shape[1]))

        # Calculate distances
        dists_own_center = distance.cdist(clust_center, X_same_clust)[0]
        smallest_dist_different_cluster = distance.cdist(X_same_clust,
                                                         X_different_clust).min(axis=1)

        # Overlap -> Where is the own center further than a point from another cluster?
        overlap_mask = dists_own_center > smallest_dist_different_cluster
        overlap_count = (overlap_mask).sum()
        total_overlap_count += overlap_count

    overlap = total_overlap_count / X.shape[0]

    return overlap
