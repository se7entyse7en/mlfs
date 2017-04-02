from collections import Counter
from operator import itemgetter

import numpy as np
from numpy import linalg as LA

from scipy.spatial import distance

from sklearn.datasets import load_digits

import fire

from mlfs import datasets
from mlfs.model_selection import KFoldCrossValidation
from mlfs.supervised.base import IModel


class KNearestNeighbors(IModel):

    def __init__(self, k=1, distance='L2'):
        # Use L1 (taxicab distance) or L2 distance (Euclidean distance)
        distance = distance.upper()
        assert distance in {'L1', 'L2'}

        self._k = k
        self._distance = distance
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train

    def predict(self, X_test):
        return np.array([self._predict(x) for x in X_test])

    def _predict(self, x):
        # Calculate the distance between `x` and each training sample
        distances = np.zeros(self._X_train.shape[0])
        for i, x_tr in enumerate(self._X_train):
            # Calculate the L1 or L2 distance using the definition of norm
            distances[i] = LA.norm(x - x_tr, ord=int(self._distance[1]))

        # Combine the distance of each training sample to its label
        neighbors = zip(distances, self._y_train)
        # Get the labels of the k nearest ones
        k_nearest_neighbors = sorted(neighbors, key=itemgetter(0))[:self._k]
        k_nearest_labels = [x[1] for x in k_nearest_neighbors]
        # The prediction is calculated through a majority vote
        prediction = Counter(k_nearest_labels).most_common(1)[0][0]

        return prediction

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)

        return {'accuracy': accuracy}


class KNearestNeighborsDemo(object):

    def run_simple(self, k=1, distance='L2', training_set_percentage=80,
                   seed=0):
        mnist = load_digits()
        ds = datasets.split(mnist, training_set_percentage, seed=seed)

        m = KNearestNeighbors(k=k, distance=distance)
        m.fit(ds['training_set']['data'], ds['training_set']['target'])
        score = m.evaluate(ds['test_set']['data'], ds['test_set']['target'])

        return score

    def run_k_fold(self, k=1, distance='L2', folds_n=3):
        mnist = load_digits()

        m = KNearestNeighbors(k=k, distance=distance)
        kfold_cv = KFoldCrossValidation(folds_n=folds_n)

        scores = kfold_cv.fit_and_evaluate(m, mnist['data'], mnist['target'])

        return scores.tolist()

    run = run_simple


if __name__ == '__main__':
    fire.Fire(KNearestNeighborsDemo)
