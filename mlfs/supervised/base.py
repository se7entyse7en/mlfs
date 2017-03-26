class IModel(object):

    def fit(self, X_train, y_train):
        raise NotImplementedError()

    def predict(self, X_test):
        raise NotImplementedError()

    def evaluate(self, X_test, y_test):
        raise NotImplementedError()
