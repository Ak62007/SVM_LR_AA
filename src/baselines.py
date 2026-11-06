"""Optimizer libraries belong only in this module. Equations (1), (2), (26)."""
from dataclasses import dataclass
from time import perf_counter
import warnings
import numpy as np
from .problems import Problem


@dataclass
class BaselineResult:
    z: np.ndarray
    objective: float
    status: str
    iterations: int
    setup_time: float
    solve_time: float
    iteration_unit: str
    multipliers: np.ndarray | None = None
    native_setup_time: float | None = None


def cvxpy_solve(p: Problem, solver='CLARABEL', tolerance=1e-10):
    import cvxpy as cp
    started = perf_counter()
    u = cp.Variable(p.k)
    v = cp.Variable(p.aux)
    if p.kind == 'svm':
        constraints = [1. - p.A @ u - v <= 0., -v <= 0.]  # (1)
        objective = .5 * cp.sum_squares(u[:p.d]) + p.reg * cp.sum(v)  # (1)
    else:
        constraints = [u[:p.d] - v <= 0., -u[:p.d] - v <= 0.]  # (2)
        objective = cp.sum(cp.logistic(-p.A @ u)) + p.reg * cp.sum(v)  # (2)
    model = cp.Problem(cp.Minimize(objective), constraints)
    options = (dict(tol_gap_abs=tolerance, tol_gap_rel=tolerance, tol_feas=tolerance,
                    max_iter=300) if solver == 'CLARABEL' else
               dict(eps_abs=max(tolerance, 1e-7), eps_rel=max(tolerance, 1e-7), max_iters=100000))
    # Canonicalize before the solve timer so setup includes CVXPY transformation.
    model.get_problem_data(solver)
    setup = perf_counter() - started
    started = perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        model.solve(solver=solver, **options)
    elapsed = perf_counter() - started
    if u.value is None or v.value is None:
        raise RuntimeError(f'{solver}: {model.status}')
    z = np.r_[u.value, v.value]
    multipliers = np.concatenate([c.dual_value for c in constraints])
    stats = model.solver_stats
    return BaselineResult(z, p.objective(z), model.status, stats.num_iters, setup,
                          elapsed, 'conic IPM' if solver == 'CLARABEL' else 'SCS splitting',
                          multipliers, stats.setup_time)


def sklearn_solve(p: Problem):
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    started = perf_counter()
    if p.kind == 'svm':
        model = SVC(kernel='linear', C=p.reg, tol=1e-10, max_iter=200000,
                    shrinking=False, random_state=1729)  # (26)
    else:
        model = LogisticRegression(penalty='l1', solver='saga', C=1./p.reg,
                                   tol=1e-11, max_iter=100000, random_state=1729)  # (26)
    setup = perf_counter() - started
    started = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(p.X, p.y)
    elapsed = perf_counter() - started
    u = np.r_[model.coef_.ravel(), model.intercept_]
    v = (np.maximum(1. - p.A @ u, 0.) if p.kind == 'svm' else np.abs(u[:p.d]))  # (1), (2)
    z = np.r_[u, v]
    status = 'iteration_limit' if any(issubclass(w.category, ConvergenceWarning) for w in caught) else 'converged'
    return BaselineResult(z, p.objective(z), status, int(np.sum(model.n_iter_)), setup,
                          elapsed, 'SMO' if p.kind == 'svm' else 'SAGA epochs')
