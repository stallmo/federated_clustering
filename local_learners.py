import os
import sys

this_dir = os.path.dirname(__file__)
sys.path.append('../cluster_library')

from cluster_library.fuzzy_cmeans import FuzzyCMeans


class FuzzyCMeansClient(FuzzyCMeans):

    def __init__(self, client_data, num_clusters, m=2, max_iter=1000, tol=0.0001):
        self.__client_data = client_data
        super().__init__(num_clusters=num_clusters, m=m, max_iter=max_iter, tol=tol)

    @property
    def client_data(self):
        return self.__client_data

    def update_centers_locally(self):
        """
        Fits the local fuzzy c-means model with this client's data.
        Make sure to communicate global cluster centers beforehand.
        :return:
        """
        super().fit(self.__client_data, initialization='federated')

    def get_center_support(self):
        """
        Calculates the sum of fuzzy assignments to each cluster.

        :return: np.array. One entry for each cluster.
        """
        U_fuzzy = super().predict(self.client_data)
        cluster_supports = U_fuzzy.sum(axis=0)

        return cluster_supports

    def cluster_assignment(self):
        """
        Calls the predict() function on the local model to obtain fuzzy assignments.

        :return: np.array of cluster assignments. Shape: (clients_data.shape[0], num_clusters).
        """
        return super().predict(self.__client_data)
