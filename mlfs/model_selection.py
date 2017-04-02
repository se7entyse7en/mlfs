from collections import deque

import numpy as np


class KFoldCrossValidation(object):

    def __init__(self, folds_n=3):
        self._folds_n = folds_n
        self._scores = np.zeros(self._folds_n)

    def split_folds(self, X, y, folds_n):
        n = y.shape[0]
        fold_size = int(n / folds_n)
        folds = deque(maxlen=folds_n)

        for i in range(folds_n):
            start = i * fold_size
            end = min(start + fold_size, n - 1)
            folds.append((X[start:end], y[start:end]))

        return folds

    def fit_and_evaluate(self, model, X, y, metric='accuracy'):
        folds = self.split_folds(X, y, self._folds_n)
        for i in range(self._folds_n):
            # Use the first fold as test set
            X_target, y_target = folds[0]

            # Use the other folds as training set
            train_fold = list(folds)[1:]
            X_train = np.concatenate([x[0] for x in train_fold])
            y_train = np.concatenate([x[1] for x in train_fold])

            # Fit the model and evaluate it
            model.fit(X_train, y_train)
            self._scores[i] = model.evaluate(X_target, y_target)[metric]

            # Rotate the folds queue in order to test all possibile
            # combinations
            folds.rotate(1)

        return self._scores
