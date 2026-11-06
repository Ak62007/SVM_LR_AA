"""Residuals and independently checkable dual certificates. Equations (21)-(23)."""
from dataclasses import dataclass, field
import numpy as np
from scipy.special import expit, xlogy


def residuals(p, z, multipliers, target=0.):
    f = p.constraints(z)
    s = -f  # (14)
    dual = p.gradient(z) + p.vjp(multipliers)  # (14)
    return dict(primal_residual=float(max(0., np.max(f))),
                r_pri=0., dual_residual=float(np.linalg.norm(dual, np.inf)),
                complementarity_residual=float(np.max(np.abs(s * multipliers - target))),
                surrogate_gap=float(s @ multipliers),
                dual_nonnegativity=float(max(0., -np.min(multipliers))))  # (14), (23)


def project_svm(p, alpha):
    # Bisection enforces intercept balance while retaining box bounds. (21)
    lo, hi = -2. * (p.reg + np.max(np.abs(alpha))), 2. * (p.reg + np.max(np.abs(alpha)))
    for _ in range(100):
        mid = (lo + hi) / 2.
        trial = np.clip(alpha - mid * p.y, 0., p.reg)  # (21)
        if p.y @ trial > 0.:
            lo = mid
        else:
            hi = mid
    return np.clip(alpha - ((lo + hi) / 2.) * p.y, 0., p.reg)


def certificate(p, z, raw, nominal_gap, ledger=None):
    if p.kind == 'svm':
        alpha = project_svm(p, raw[:p.n])
        margins = p.A @ z[:p.k]  # (1)
        threshold = max(1e-7, min(1e-3, 20. * nominal_gap / max(p.reg, 1e-20)))  # (21)
        free = np.abs(margins - 1.) < threshold
        fixed = ~free
        candidate = np.where(margins < 1., p.reg, 0.)  # (21)
        if ledger is not None and free.any():
            rhs = np.r_[z[:p.d], 0.] - p.A[fixed].T @ candidate[fixed]  # (21)
            candidate[free] = ledger.least_squares(p.A[free].T, rhs)
            if (candidate.min() >= -1e-9 and candidate.max() <= p.reg + 1e-9 and
                    abs(p.y @ candidate) <= 1e-7):
                candidate = project_svm(p, candidate)
                current_error = np.linalg.norm(p.A.T @ alpha - np.r_[z[:p.d], 0.], np.inf)  # (21)
                candidate_error = np.linalg.norm(p.A.T @ candidate - np.r_[z[:p.d], 0.], np.inf)  # (21)
                if candidate_error < current_error:
                    alpha = candidate
        multipliers = np.r_[alpha, p.reg-alpha]  # (21)
        wd = p.X.T @ (p.y * alpha)  # (21)
        lower = float(alpha.sum() - .5 * wd @ wd)  # (21)
        feasibility = float(max(abs(p.y @ alpha), max(0., -alpha.min()), max(0., alpha.max()-p.reg)))
    else:
        # Balance probabilities by a scalar intercept shift before scaling. (22)
        a = p.A @ z[:p.k]
        lo, hi = -max(100., np.max(np.abs(a)) + 100.), max(100., np.max(np.abs(a)) + 100.)
        for _ in range(100):
            mid = (lo + hi) / 2.
            q = expit(-a - p.y * mid)  # (22)
            if p.y @ q > 0.:
                lo = mid
            else:
                hi = mid
        q = expit(-a - p.y * ((lo + hi) / 2.))
        norm = np.max(np.abs(p.X.T @ (p.y * q)))  # (22)
        q *= min(1., p.reg / max(norm, np.finfo(float).tiny))  # (22)
        lower = float(-np.sum(xlogy(q, q) + xlogy(1.-q, 1.-q)))  # (22)
        gradient = p.loss_gradient(z)[:p.d]
        plus = np.clip((p.reg-gradient) / 2., 0., p.reg)  # (22)
        multipliers = np.r_[plus, p.reg-plus]  # (22)
        feasibility = float(max(abs(p.y @ q), max(0., np.max(np.abs(p.X.T @ (p.y*q))) - p.reg)))
    return multipliers, lower, feasibility


@dataclass
class SolverResult:
    z: np.ndarray
    multipliers: np.ndarray
    raw_multipliers: np.ndarray
    history: list
    ledger: object
    status: str
    objective: float
    setup_time: float
    solve_time: float
    diagnostics_time: float
    outer_iterations: int = 0
    newton_iterations: int = 0
    barrier_gap: float = float('nan')
    tau: float = float('nan')
    dual_bound: float = float('nan')
    certificate_feasibility: float = float('nan')
    last_condition: float = float('nan')
    equilibrated_condition: float = float('nan')
    message: str = ''
    metrics: dict = field(default_factory=dict)
    raw_metrics: dict = field(default_factory=dict)

    @property
    def solves(self):
        return self.ledger.count

    @property
    def recovery_solves(self):
        return sum(e['phase'] == 'certificate' for e in self.ledger.entries)

    @property
    def certified_gap(self):
        return self.objective - self.dual_bound  # (23)
