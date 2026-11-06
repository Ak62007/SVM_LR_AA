import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from src.problems import Problem, barrier_terms
from src.data import blobs, conditioned, Standardizer, breast_cancer


@pytest.mark.parametrize('kind', ['svm', 'logreg'])
def test_objective_and_barrier_derivatives(kind):
    X, y = blobs(12, 3)
    p = Problem(X, y, kind, .7)
    z = p.initial()
    z[:p.k] += np.random.default_rng(22).normal(size=p.k) * .03
    g = approx_derivative(lambda q: np.array([p.objective(q)]), z).ravel()
    H = approx_derivative(p.gradient, z)
    np.testing.assert_allclose(p.gradient(z), g, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(p.hessian(z), H, rtol=2e-6, atol=2e-7)
    _, bg, bH = barrier_terms(p, z, 3.)
    fg = approx_derivative(lambda q: np.array([barrier_terms(p, q, 3.)[0]]), z).ravel()
    fH = approx_derivative(lambda q: barrier_terms(p, q, 3.)[1], z)
    np.testing.assert_allclose(bg, fg, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(bH, fH, rtol=2e-6, atol=2e-7)
    G = p.jacobian()
    v = np.random.default_rng(2).normal(size=p.size)
    w = np.random.default_rng(3).normal(size=p.m)
    np.testing.assert_allclose(p.jvp(v), G @ v)
    np.testing.assert_allclose(p.vjp(w), G.T @ w)
    np.testing.assert_allclose(G, approx_derivative(p.constraints, z), atol=1e-8)
    assert np.all(p.constraints(p.initial()) < 0.)


def test_data_condition_and_constant_scaling():
    X, y = conditioned(condition=1e6)
    assert np.linalg.cond(X.T @ X) == pytest.approx(1e6, rel=1e-4)
    X[:, -1] = 3.
    Z = Standardizer.fit(X).transform(X)
    assert np.isfinite(Z).all() and np.all(Z[:, -1] == 0.)
    X, Xt, y, yt = breast_cancer()
    np.testing.assert_allclose(X.mean(0), 0., atol=1e-11)
    assert len(y) + len(yt) == 569


@pytest.mark.parametrize('reg', [0., -1., float('nan')])
def test_invalid_regularization(reg):
    with pytest.raises(ValueError):
        Problem(*blobs(), reg=reg)


def test_logistic_loss_is_finite_in_both_tails():
    p = Problem(*blobs(10, 2), kind='logreg')
    z = p.initial()
    z[:p.d] = [1e6, -1e6]
    assert np.isfinite(p.objective(z))
    assert np.isfinite(p.gradient(z)).all()
    assert np.isfinite(p.hessian(z)).all()
