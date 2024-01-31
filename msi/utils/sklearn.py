# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2023
Author: Arne Thomsen
"""


class GeneralizedSklearnModel:
    """
    Wrapper class for sklearn models that are only fit with an X and no y (like StandardScaler and PCA) to handle
    multidimensional sample dimensions.
    """

    def __init__(self, model):
        self.model = model

    def fit(self, X):
        X, _ = self._flatten(X)
        self.model.fit(X)

    def fit_transform(self, X):
        X, samples_shape = self._flatten(X)
        X = self.model.fit_transform(X)
        X = self._unflatten(X, samples_shape)
        return X

    def transform(self, X):
        X, samples_shape = self._flatten(X)
        X = self.model.transform(X)
        X = self._unflatten(X, samples_shape)
        return X

    @staticmethod
    def _flatten(X):
        if X.ndim > 2:
            samples_shape = X.shape[:-1]
            features_shape = X.shape[-1]
            X = X.reshape(-1, features_shape)
        else:
            samples_shape = None

        return X, samples_shape

    @staticmethod
    def _unflatten(X, samples_shape):
        if samples_shape is not None:
            X = X.reshape(samples_shape + (-1,))

        return X
