"""Identical primal formulations for custom solvers and CVXPY. Equations (1)-(3)."""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit


@dataclass
class Problem:
    X: np.ndarray
    y: np.ndarray
    kind: str = 'svm'
    reg: float = 1.0

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.X.ndim != 2 or self.y.shape != (len(self.X),):
            raise ValueError('X must be a matrix and y must have one entry per row')
        if not np.isfinite(self.X).all() or not np.isfinite(self.y).all():
            raise ValueError('Data must be finite')
        if set(self.y) != {-1., 1.}:
            raise ValueError('Both labels -1 and +1 are required')
        if self.kind not in {'svm', 'logreg'} or not np.isfinite(self.reg) or self.reg <= 0:
            raise ValueError('Choose svm or logreg and positive finite regularization')
        self.n, self.d = self.X.shape
        if self.d < 1:
            raise ValueError('At least one feature is required')
        self.k = self.d + 1
        self.Z = np.column_stack([self.X, np.ones(self.n)])  # (1)
        self.A = self.y[:, None] * self.Z  # (1)
        self.aux = self.n if self.kind == 'svm' else self.d
        self.size = self.k + self.aux
        self.m = 2 * self.aux  # (1), (2)

    def initial(self):
        return np.r_[np.zeros(self.k), np.full(self.aux, 2. if self.kind == 'svm' else 1.)]  # (1), (2)

    def objective(self, z):
        if self.kind == 'svm':
            return float(0.5 * z[:self.d] @ z[:self.d] + self.reg * z[self.k:].sum())  # (1)
        return float(np.logaddexp(0, -self.A @ z[:self.k]).sum() + self.reg * z[self.k:].sum())  # (2)

    def loss_gradient(self, z):
        if self.kind == 'svm':
            return np.r_[z[:self.d], 0.]  # (1)
        q = expit(-self.A @ z[:self.k])  # (2)
        return -self.A.T @ q  # (2)

    def gradient(self, z):
        return np.r_[self.loss_gradient(z), np.full(self.aux, self.reg)]  # (1), (2)

    def core_hessian(self, z):
        if self.kind == 'svm':
            return np.diag(np.r_[np.ones(self.d), 0.])  # (1)
        a = self.A @ z[:self.k]
        p = expit(a) * expit(-a)  # (2), avoids cancellation for large a
        return self.A.T @ (p[:, None] * self.A)  # (2)

    def hessian(self, z):
        H = np.zeros((self.size, self.size))
        H[:self.k, :self.k] = self.core_hessian(z)  # (1), (2)
        return H

    def constraints(self, z):
        if self.kind == 'svm':
            return np.r_[1. - self.A @ z[:self.k] - z[self.k:], -z[self.k:]]  # (1)
        return np.r_[z[:self.d] - z[self.k:], -z[:self.d] - z[self.k:]]  # (2)

    def jacobian(self):
        # Only the full reference path materializes the identity blocks. (1), (2)
        I = np.eye(self.aux)
        if self.kind == 'svm':
            return np.block([[-self.A, -I], [np.zeros_like(self.A), -I]])
        B = np.column_stack([np.eye(self.d), np.zeros(self.d)])
        return np.block([[B, -I], [-B, -I]])

    def jvp(self, dz):
        if self.kind == 'svm':
            return np.r_[-self.A @ dz[:self.k] - dz[self.k:], -dz[self.k:]]  # (1)
        return np.r_[dz[:self.d] - dz[self.k:], -dz[:self.d] - dz[self.k:]]  # (2)

    def vjp(self, v):
        a, b = v[:self.aux], v[self.aux:]
        if self.kind == 'svm':
            return np.r_[-self.A.T @ a, -a-b]  # (1)
        return np.r_[a-b, 0., -a-b]  # (2)

    def weighted_matrix(self, z, weights, structure=True):
        a, b = weights[:self.aux], weights[self.aux:]
        if not structure:
            G = self.jacobian()
            return self.hessian(z) + G.T @ (weights[:, None] * G)  # (6), (15)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        harmonic = lo / (1. + lo / hi)  # (18)
        if self.kind == 'svm':
            return self.core_hessian(z) + self.A.T @ (harmonic[:, None] * self.A)  # (18)
        M = self.core_hessian(z)
        M[np.arange(self.d), np.arange(self.d)] += 4. * harmonic  # (19)
        return M

    def newton_solve(self, z, weights, rhs, ledger, structure=True, phase='newton'):
        M = self.weighted_matrix(z, weights, structure)
        if not structure:
            return ledger.solve(M, rhs, phase=phase)
        a, b = weights[:self.aux], weights[self.aux:]
        total = a + b  # (18), (19)
        ru, rv = rhs[:self.k], rhs[self.k:]
        if self.kind == 'svm':
            reduced_rhs = ru - self.A.T @ ((a / total) * rv)  # (18)
        else:
            reduced_rhs = ru.copy()
            reduced_rhs[:self.d] -= ((b-a) / total) * rv  # (19)
        du = ledger.solve(M, reduced_rhs, phase=phase)
        cross = a * (self.A @ du) if self.kind == 'svm' else (b-a) * du[:self.d]  # (18), (19)
        dv = (rv - cross) / total  # (18), (19)
        return np.r_[du, dv]

    def objective_difference(self, z, dz, step):
        if self.kind == 'svm':
            w, dw = z[:self.d], dz[:self.d]
            return float(step * (w @ dw + self.reg * dz[self.k:].sum()) +
                         0.5 * step**2 * (dw @ dw))  # (7)
        return self.objective(z + step * dz) - self.objective(z)  # (2), (7)

    def predict(self, z, X=None):
        X = self.X if X is None else np.asarray(X)
        scores = X @ z[:self.d] + z[self.d]  # (1), (2)
        return np.where(scores >= 0., 1., -1.)


def barrier_terms(problem, z, tau):
    s = -problem.constraints(z)  # (3)
    if np.any(s <= 0):
        raise ValueError('The logarithmic barrier needs strict feasibility')
    value = problem.objective(z) - np.log(s).sum() / tau  # (3), (6)
    gradient = problem.gradient(z) + problem.vjp(1. / (tau * s))  # (6)
    G = problem.jacobian()
    hessian = problem.hessian(z) + G.T @ ((1. / (tau * s**2))[:, None] * G)  # (6)
    return value, gradient, hessian
