import os
import sys

this_dir = os.path.dirname(__file__)
sys.path.append('../cluster_library')
from cluster_library.fuzzy_cmeans import FuzzyCMeans

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import label_binarize


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


class KMeansClient:

    def __init__(self, client_data, num_clusters, max_iter=1000, tol=0.0001):
        self.__client_data = client_data
        self.__num_clusters = num_clusters
        self.__max_iter = max_iter
        self.__tol = tol
        self.__kmeans_model = KMeans(n_clusters=self.__num_clusters,
                                     init='k-means++',
                                     max_iter=self.__max_iter,
                                     tol=self.__tol,
                                     precompute_distances='auto',
                                     random_state=43,
                                     n_jobs=None)

        self.__local_iters = 0
        self.__cur_centers = None  # set when fit is called or global centers are communicated

    @property
    def centers(self):
        return self.__cur_centers

    @property
    def client_data(self):
        return self.__client_data

    @property
    def num_clusters(self):
        return self.__num_clusters

    @property
    def iterations(self):
        return self.__local_iters

    @property
    def tol(self):
        return self.__tol

    @property
    def max_iter(self):
        return self.__max_iter

    def set_centers(self, global_centers):
        """
        Method that re-initializes the local kmeans model with centers set to <global_centers>.

        :param global_centers: np.array of shape (num_centers, n_features).
        :return:
        """
        self.__cur_centers = global_centers
        self.__kmeans_model = KMeans(n_clusters=self.__num_clusters,
                                     init=global_centers,
                                     max_iter=self.__max_iter,
                                     tol=self.__tol,
                                     precompute_distances='auto',
                                     random_state=43,
                                     n_jobs=None)

    def update_centers_locally(self):
        """
        Updates the cluster centers according to the local learner's local data.
        :return:
        """
        self.__kmeans_model.fit(self.__client_data)
        self.__local_iters += self.__kmeans_model.n_iter_
        self.__cur_centers = self.__kmeans_model.cluster_centers_

    def cluster_assignment(self):
        """
        Calls the predict() function on the local model to obtain obtain labels and translates labels into assignment matrix.

        :return: np.array of cluster assignments. Shape: (clients_data.shape[0], num_clusters).
        """
        labels = self.__kmeans_model.predict(self.__client_data)
        if self.__num_clusters == 2:
            labels = np.append(labels, 2)

        assignment_matrix = label_binarize(labels, classes=sorted(list(set(labels))))
        if self.__num_clusters == 2:
            assignment_matrix = assignment_matrix[0:-1, 0:-1]

        return assignment_matrix

    def get_center_support(self):
        """
        Calculates the sum of assignmented points to each cluster.

        :return: np.array. One entry for each cluster.
        """
        assignment_matrix = self.cluster_assignment()
        center_supports = assignment_matrix.sum(
            axis=0)  # in case of k-means, this is the number of points assigned to each cluster

        return center_supports
