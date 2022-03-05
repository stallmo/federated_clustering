import numpy as np


def distribute_data_among_clients(X, n_clients):
    client_labels = np.random.randint(low=0, high=n_clients, size=X.shape[0])
    X_labeled = np.hstack((X, client_labels[:, None])).copy()
    X_labeled = X_labeled[X_labeled[:, -1].argsort()]
    clients_data = np.split(X_labeled[:, :-1], np.unique(X_labeled[:, -1], return_index=True)[1][1:])

    return clients_data
