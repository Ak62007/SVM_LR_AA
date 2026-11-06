import numpy as np
import pytest
from src.problems import Problem
from src.data import blobs, standardized
from src.baselines import cvxpy_solve, sklearn_solve


@pytest.mark.parametrize('kind', ['svm', 'logreg'])
@pytest.mark.parametrize('separable', [False, True])
def test_baseline_formulations_match(kind, separable):
    X, y = blobs(60, 4, separable=separable)
    p = Problem(standardized(X), y, kind, 1.)
    reference = cvxpy_solve(p)
    scs = cvxpy_solve(p, 'SCS')
    sk = sklearn_solve(p)
    assert reference.status in {'optimal', 'optimal_inaccurate'}
    assert abs(scs.objective-reference.objective) / max(np.finfo(float).eps, abs(reference.objective)) < 1e-5
    assert abs(sk.objective-reference.objective) / max(np.finfo(float).eps, abs(reference.objective)) < 1e-6
    assert np.mean(p.predict(sk.z) == p.predict(reference.z)) >= .99
