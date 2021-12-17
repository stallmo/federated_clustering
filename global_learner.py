import numpy as np
import copy
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


class GlobalClusterer:

    def __init__(self, local_learners, num_clusters, data_dim, max_rounds=1000, tol=0.0001,
                 global_center_update='fedavg', weighing_function=None):
        self.__local_learners = local_learners
        self.__tol = tol
        self.__num_clusters = num_clusters
        self.__max_rounds = max_rounds
        self.__data_dim = data_dim
        self.__weighing_function = weighing_function
        self.__global_center_update = global_center_update  # method applied to update global cluster centers

        # going to be set when fit() is called
        self.__W = None  # clients cluster supports
        self.__global_centers = None
        self.__iterations = 0

    @property
    def local_learners(self):
        return self.__local_learners

    @property
    def tol(self):
        return self.__tol

    @property
    def num_clusters(self):
        return self.__num_clusters

    @property
    def max_rounds(self):
        return self.__max_rounds

    @property
    def cluster_centers(self):
        return self.__global_centers

    @property
    def client_cluster_supports(self):
        return self.__W

    @property
    def data_dimension(self):
        return self.__data_dim

    @property
    def iterations(self):
        return self.__iterations

    def __fedavg_update(self, local_center_supports, new_centers):
        """
        Method that applied federated averaging to update the global cluster centers.

        :param local_center_supports: List of np.arrays. Each array has <num_cluster> entries, i.e. the support for each cluster.
        :param new_centers:list of np.arrays. Each array has <num_cluster> rows and <data_dimension> columns.
        :return: None
        """
        support_normalizer = np.array(local_center_supports).sum(axis=0)
        new_centers = np.array(new_centers)
        # multiple each client's center with its support
        weighted_new_centers = np.array(new_centers).reshape(-1, self.__data_dim) * local_center_supports.reshape(-1, 1)
        # reshape to prepare for normalizing by total support
        weighted_new_centers = weighted_new_centers.reshape(-1, self.__num_clusters, self.__data_dim)
        # calculate new global centers
        self.__global_centers = weighted_new_centers.sum(axis=0) / support_normalizer.reshape(-1, 1)

    def __kmeans_center_update(self, new_centers, local_center_supports):
        """
        Method that applies kmeans to the

        :param new_centers:
        :param local_center_supports:
        :return:
        """

        if self.__iterations < 2:
            init_met = 'k-means++'
        else:
            init_met = self.__global_centers
        kmeans_model = KMeans(n_clusters=self.__num_clusters,
                              init=init_met,
                              max_iter=500,
                              # precompute_distances='auto',
                              random_state=43,
                              # n_jobs=None
                              )

        new_centers = np.array(new_centers).reshape(-1, self.__data_dim)
        kmeans_model.fit(new_centers)

        self.__global_centers = kmeans_model.cluster_centers_

        pass

    def __update_global_centers(self, new_centers, local_center_supports):
        """
        Update routine that calculates new centers from each clients information.

        :param new_centers: list of np.arrays. Each array has <num_cluster> rows and <data_dimension> columns.
        :param local_center_supports: List of np.arrays. Each array has <num_cluster> entries, i.e. the support for each cluster.
        :return: None.
        """
        # we will need to normalize the centers by their respective total support
        local_center_supports = np.array(local_center_supports)
        if not self.__weighing_function is None:
            local_center_supports = self.__weighing_function(local_center_supports)

        if self.__global_center_update == 'fedavg':
            self.__fedavg_update(local_center_supports, new_centers)
        elif self.__global_center_update == 'kmeans':
            self.__kmeans_center_update(new_centers, local_center_supports)

    def fit(self):
        """
        Fits the global fuzzy c-means model from the local learners data.

        :return:
        """
        # global cluster initialization
        if self.__global_centers is None:
            self.__global_centers = np.random.randint(low=-10, high=10, size=(self.__num_clusters, self.__data_dim))

        # start federation rounds
        for _global_round in range(self.__max_rounds):
            self.__iterations += 1
            new_centers = []
            local_center_supports = []
            for local_learner in self.__local_learners:
                # share global cluster centers with clients
                local_learner.set_centers(self.__global_centers)
                # ask clients to recalculate cluster centers locally
                local_learner.update_centers_locally()
                # ask clients for each cluster's center and support
                new_centers.append(local_learner.centers)
                local_center_supports.append(local_learner.get_center_support())

            # recalculate global cluster centers
            prev_global_centers = copy.deepcopy(self.__global_centers)  # needed to check convergence
            self.__update_global_centers(new_centers, local_center_supports)

            # check convergence
            if np.linalg.norm(self.__global_centers - prev_global_centers) < self.__tol:
                break
        # communicate final global centers
        for local_learner in self.__local_learners:
            local_learner.set_centers(self.__global_centers)

        pass
