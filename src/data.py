"""Deterministic datasets; sklearn is used only to load the real baseline dataset."""
from dataclasses import dataclass
import numpy as np

SEED = 1729


def blobs(n=100, d=5, overlap=1.0, seed=SEED, separable=False):
    if n < 4 or d < 1 or overlap <= 0:
        raise ValueError('Use n>=4, d>=1, and positive overlap')
    rng = np.random.default_rng(seed)
    y = np.where(np.arange(n) % 2, 1., -1.)
    rng.shuffle(y)
    X = rng.normal(size=(n, d)) * overlap
    X[:, 0] += 1.5 * y
    if separable:
        X[:, 0] = y * (0.5 + np.abs(X[:, 0]))
    return X, y


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X):
        scale = np.std(X, axis=0)  # (25)
        return cls(np.mean(X, axis=0), np.where(scale > 0., scale, 1.))

    def transform(self, X):
        return (X - self.mean) / self.scale  # (25)


def standardized(X):
    return Standardizer.fit(X).transform(X)


def split(X, y, test_fraction=0.25, seed=SEED):
    rng = np.random.default_rng(seed)
    train, test = [], []
    for label in [-1, 1]:
        idx = rng.permutation(np.flatnonzero(y == label))
        count = max(1, int(len(idx) * test_fraction))
        test.extend(idx[:count]); train.extend(idx[count:])
    train, test = np.array(train), np.array(test)
    return X[train], X[test], y[train], y[test]


def breast_cancer():
    from sklearn.datasets import load_breast_cancer
    dataset = load_breast_cancer()
    X, Xt, y, yt = split(dataset.data, 2. * dataset.target - 1.)
    scaler = Standardizer.fit(X)  # (25), fit on training rows only
    return scaler.transform(X), scaler.transform(Xt), y, yt


def conditioned(n=150, d=8, condition=10., scale_span=0., seed=SEED):
    if n <= d or condition < 1:
        raise ValueError('Controlled conditioning requires n>d and condition>=1')
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(n, d)))
    V, _ = np.linalg.qr(rng.normal(size=(d, d)))
    singular = np.geomspace(1., 1. / np.sqrt(condition), d)  # (25)
    X = np.sqrt(n) * (U * singular) @ V.T  # (25)
    signal = U[:, 0] * np.sqrt(n) + rng.normal(size=n) * 0.6
    y = np.where(signal >= np.median(signal), 1., -1.)
    X *= np.geomspace(1., 10.**scale_span, d)  # (25)
    return X, y
