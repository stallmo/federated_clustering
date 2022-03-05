import numpy as np


class DataGenerator:

    def __init__(self):
        pass

    def generate_gaussian_data(self, n_samples, center, spread):
        X = np.random.normal(loc=center,
                             scale=spread,
                             size=(n_samples, len(center))
                             )
        return X
