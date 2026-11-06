import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from src.problems import Problem
from src.data import blobs, standardized
from src.baselines import cvxpy_solve, sklearn_solve
from src import barrier, primal_dual
from src.linsolve import SolveLedger
from src.metrics import residuals


def make_problem(kind='svm', case='ordinary', reg=1.):
    X, y = blobs(30, 3, separable=case == 'separable', overlap=1.5)
    if case == 'conflicting':
        X[1] = X[0]; y[1] = -y[0]
    if case == 'duplicate':
        X[3:6] = X[0]; y[3:6] = y[0]
    if case == 'constant':
        X[:, -1] = 2.
    return Problem(standardized(X), y, kind, reg)


@pytest.mark.parametrize('kind', ['svm', 'logreg'])
def test_elimination_and_kkt_jacobian(kind):
    p = make_problem(kind)
    z = p.initial()
    z[:p.k] += .01
    multipliers = np.linspace(.3, 1.3, p.m)
    s = -p.constraints(z)
    target = .17
    reduced_ledger, full_ledger = SolveLedger(), SolveLedger()
    reduced = primal_dual.direction(p, z, multipliers, target, reduced_ledger, True)
    full = primal_dual.direction(p, z, multipliers, target, full_ledger, False)
    for r, f in zip(reduced, full):
        np.testing.assert_allclose(r, f, rtol=2e-10, atol=2e-10)
    def residual(q):
        zz, ll, ss = q[:p.size], q[p.size:p.size+p.m], q[p.size+p.m:]
        return np.r_[p.gradient(zz) + p.vjp(ll), p.constraints(zz)+ss, ss*ll-target]
    base = np.r_[z, multipliers, s]
    jac = approx_derivative(residual, base)
    direction = np.r_[*reduced]
    np.testing.assert_allclose(jac @ direction, -residual(base), rtol=2e-8, atol=2e-8)
    assert reduced_ledger.count == full_ledger.count == 1
    assert reduced_ledger.entries[0]['dimension'] == p.k
    assert full_ledger.entries[0]['dimension'] == p.size


@pytest.mark.parametrize('method', ['barrier', 'primal_dual'])
@pytest.mark.parametrize('kind', ['svm', 'logreg'])
@pytest.mark.parametrize('case,reg', [('ordinary', 1.), ('duplicate', 1.),
                                     ('separable', 1.), ('constant', 1.), ('conflicting', 1.),
                                     ('ordinary', 1e-4), ('ordinary', 1e4)])
def test_solution_against_reference(method, kind, case, reg):
    p = make_problem(kind, case, reg)
    result = getattr({'barrier': barrier, 'primal_dual': primal_dual}[method], 'solve')(p)
    reference = cvxpy_solve(p)
    assert result.status == 'converged', result.message
    assert abs(result.objective-reference.objective) / max(np.finfo(float).eps, abs(reference.objective)) < 1e-6
    for key in ['primal_residual', 'dual_residual', 'complementarity_residual', 'dual_nonnegativity']:
        assert result.metrics[key] < 1e-6, (key, result.metrics)
    assert result.certificate_feasibility < 1e-6
    assert result.certified_gap >= -1e-7
    assert result.certified_gap / max(1., abs(result.objective)) < 1e-6
    assert result.solves == len(result.ledger.entries)
    assert result.solves >= result.newton_iterations
    if method == 'barrier':
        assert result.barrier_gap < 1e-8
        assert result.barrier_gap == p.m/result.tau
        for row in result.history:
            assert row['barrier_gap'] == p.m/row['tau']
        assert np.all(result.raw_multipliers > 0.)
        assert result.raw_metrics['dual_residual'] < 1e-6
        assert result.raw_metrics['r_pri'] < 1e-6
        if kind == 'svm':
            alpha = result.multipliers[:p.n]
            assert abs(p.y @ alpha) < 1e-7
            assert np.all(alpha >= 0.) and np.all(alpha <= p.reg)
    else:
        assert result.metrics['surrogate_gap'] < 1e-8
        assert result.metrics['dual_residual'] < 1e-8


@pytest.mark.parametrize('kind', ['svm', 'logreg'])
@pytest.mark.parametrize('separable', [False, True])
def test_prediction_agreement(kind, separable):
    X, y = blobs(100, 5, separable=separable)
    p = Problem(standardized(X), y, kind, 1.)
    sk = sklearn_solve(p)
    for method in [barrier.solve, primal_dual.solve]:
        result = method(p)
        agreement = np.mean(p.predict(result.z) == p.predict(sk.z))
        assert agreement == 1. if separable else agreement >= .99


def test_reject_infeasible_start_and_invalid_policy():
    p = make_problem()
    for method in [barrier.solve, primal_dual.solve]:
        with pytest.raises(ValueError):
            method(p, start=np.zeros(p.size))
    with pytest.raises(ValueError):
        barrier.solve(p, mu=1.)


def test_failure_and_solve_accounting():
    p = make_problem()
    r = primal_dual.solve(p, max_iterations=1)
    assert r.status == 'iteration_limit'
    assert r.newton_iterations == 1
    ledger = SolveLedger()
    solution = ledger.solve(np.array([[1., 1.], [1., 1.]]), np.ones(2))
    assert ledger.count == 2
    assert not ledger.entries[0]['success'] and ledger.entries[1]['success']
    np.testing.assert_allclose(np.ones((2, 2)) @ solution, np.ones(2))


def test_newton_coordinate_invariance():
    rng = np.random.default_rng(1729)
    A = rng.normal(size=(7, 7))
    H = A.T @ A + np.eye(7)
    g = rng.normal(size=7)
    T = np.diag(np.geomspace(.001, 1000., 7))
    ledger = SolveLedger()
    original = ledger.solve(H, -g)
    transformed = T @ ledger.solve(T.T @ H @ T, -T.T @ g)
    np.testing.assert_allclose(original, transformed, rtol=2e-10, atol=2e-10)
    assert ledger.count == 2
