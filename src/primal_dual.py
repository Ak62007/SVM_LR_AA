"""One perturbed-KKT Newton direction per iteration. Equations (14)-(16)."""
from time import perf_counter
import numpy as np
from .linsolve import SolveLedger
from .metrics import SolverResult, certificate, residuals


def direction(p, z, multipliers, target, ledger, structure=True):
    s = -p.constraints(z)  # (14)
    rd = p.gradient(z) + p.vjp(multipliers)  # (14)
    rc = s * multipliers - target  # (14)
    weights = multipliers / s  # (15)
    rhs = -rd + p.vjp(rc / s)  # (15), r_pri=0
    dz = p.newton_solve(z, weights, rhs, ledger, structure)
    ds = -p.jvp(dz)  # (15)
    dl = (-rc - multipliers * ds) / s  # (15)
    return dz, dl, ds


def solve(problem, tolerance=1e-8, gap_tolerance=5e-9, sigma=10.,
          max_iterations=120, structure=True, start=None, diagnostics=True):
    if tolerance <= 0. or gap_tolerance <= 0. or sigma <= 1.:
        raise ValueError('Use positive tolerances and sigma>1')
    started = perf_counter()
    p = problem
    z = p.initial() if start is None else np.array(start, dtype=float, copy=True)
    if z.shape != (p.size,) or not np.isfinite(z).all() or np.any(p.constraints(z) >= 0.):
        raise ValueError('A finite strictly feasible start is required')
    multipliers = np.ones(p.m)
    ledger, history = SolveLedger(), []
    status, message = 'iteration_limit', ''
    directions = 0
    setup_time = perf_counter() - started
    started = perf_counter()
    try:
        for iteration in range(max_iterations + 1):
            s = -p.constraints(z)  # (14)
            gap = float(s @ multipliers)  # (14)
            average_gap = gap / p.m  # (14)
            dual_norm = np.linalg.norm(p.gradient(z) + p.vjp(multipliers), np.inf)  # (14)
            local = dual_norm <= min(.1, np.sqrt(average_gap))  # (28)
            centering_fraction = min(1. / sigma, np.sqrt(average_gap)) if local else 1. / sigma  # (28)
            target = centering_fraction * average_gap  # (28)
            stats = residuals(p, z, multipliers, target)
            record = dict(iteration=iteration, objective=p.objective(z), target=target,
                          centering_fraction=centering_fraction,
                          step=0., backtracks=0, solves=ledger.count, **stats)
            history.append(record)
            if (stats['r_pri'] < tolerance and stats['primal_residual'] < tolerance and
                    stats['dual_residual'] < tolerance and gap < gap_tolerance):
                status = 'converged'
                record['event'] = 'converged'
                break
            if iteration == max_iterations:
                break
            dz, dl, ds = direction(p, z, multipliers, target, ledger, structure)
            directions += 1
            positive = p.jvp(dz) > 0.
            negative = dl < 0.
            primal_step = np.min(s[positive] / p.jvp(dz)[positive]) if positive.any() else np.inf
            dual_step = np.min(-multipliers[negative] / dl[negative]) if negative.any() else np.inf
            fraction = 1. - min(.01, np.sqrt(average_gap)) if local else .99  # (28)
            step = min(1., fraction * primal_step, fraction * dual_step)  # (16), (28)
            rd = p.gradient(z) + p.vjp(multipliers)  # (14)
            norm = np.linalg.norm(np.r_[rd, s * multipliers - target])  # (16)
            accepted = False
            for backtracks in range(80):
                candidate, proposed = z + step * dz, multipliers + step * dl
                cs = -p.constraints(candidate)
                if np.all(cs > 0.) and np.all(proposed > 0.):
                    crd = p.gradient(candidate) + p.vjp(proposed)  # (14)
                    cnorm = np.linalg.norm(np.r_[crd, cs * proposed - target])  # (16)
                    if cnorm <= (1. - .01 * step) * norm:
                        accepted = True
                        break
                step *= .5
            record.update(step=step if accepted else 0., backtracks=backtracks,
                          solves=ledger.count, event='step')
            if not accepted or np.array_equal(z, candidate):
                raise FloatingPointError('Primal-dual line search stalled')
            z, multipliers = candidate, proposed
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
        status, message = 'numerical_failure', str(exc)
    # A certificate is diagnostic; primal-dual stopping uses its own raw residuals.
    _, lower, feasibility = certificate(p, z, multipliers,
                                        float((-p.constraints(z)) @ multipliers), ledger)
    solve_time = perf_counter() - started
    started = perf_counter()
    conditions = ledger.condition_numbers() if diagnostics else (float('nan'), float('nan'))
    metrics = residuals(p, z, multipliers, 0.)
    return SolverResult(z, multipliers, multipliers.copy(), history, ledger, status,
                        p.objective(z), setup_time, solve_time, perf_counter()-started,
                        newton_iterations=directions, dual_bound=lower,
                        certificate_feasibility=feasibility, last_condition=conditions[0],
                        equilibrated_condition=conditions[1], message=message,
                        metrics=metrics, raw_metrics=metrics.copy())
