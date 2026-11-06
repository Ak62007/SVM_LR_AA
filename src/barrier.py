"""Feasible-start log barrier with Newton centering. Equations (3)-(8)."""
from time import perf_counter
import numpy as np
from .linsolve import SolveLedger
from .metrics import SolverResult, certificate, residuals


def solve(problem, gap_tolerance=5e-9, centering_tolerance=1e-16, mu=10.,
          tau0=1., max_outer=30, max_inner=100, structure=True,
          start=None, recover_certificate=True, diagnostics=True):
    if mu <= 1. or tau0 <= 0. or min(gap_tolerance, centering_tolerance) <= 0.:
        raise ValueError('Use mu>1 and positive tolerances and tau0')
    started = perf_counter()
    p = problem
    z = p.initial() if start is None else np.array(start, dtype=float, copy=True)
    if z.shape != (p.size,) or not np.isfinite(z).all() or np.any(p.constraints(z) >= 0.):
        raise ValueError('A finite strictly feasible start is required')
    ledger = SolveLedger()
    tau, history, status, message = float(tau0), [], 'iteration_limit', ''
    directions, outer_count = 0, 0
    s = -p.constraints(z)  # (27), retain small affine slacks without cancellation
    setup_time = perf_counter() - started
    started = perf_counter()
    try:
        for outer in range(max_outer):
            outer_count += 1
            centered = False
            for inner in range(max_inner):
                raw = 1. / (tau * s)  # (4)
                g = p.gradient(z) + p.vjp(raw)  # (6)
                weights = raw / s  # (6)
                dz = p.newton_solve(z, weights, -g, ledger, structure)
                directions += 1
                decrement2 = float(-tau * g @ dz)  # (6)
                if decrement2 < -1e-7 or not np.isfinite(decrement2):
                    raise FloatingPointError('Newton direction is not a finite descent direction')
                decrement2 = max(0., decrement2)
                record = dict(outer=outer, inner=inner, tau=tau,
                              objective=p.objective(z), decrement=np.sqrt(decrement2),
                              decrement2=decrement2, barrier_gap=p.m / tau,
                              step=0., backtracks=0, solves=ledger.count,
                              newton_iteration=directions, **residuals(p, z, raw, 1./tau))  # (6), (23)
                record['r_pri'] = float(np.linalg.norm(p.constraints(z) + s, np.inf))  # (27)
                record['surrogate_gap'] = float(s @ raw)  # (27)
                record['complementarity_residual'] = float(np.max(np.abs(s * raw - 1./tau)))  # (27)
                history.append(record)
                if decrement2 / 2. <= centering_tolerance:  # (6)
                    centered = True
                    record['event'] = 'centered'
                    break
                change = p.jvp(dz)  # (7)
                positive = change > 0.
                boundary = np.min(s[positive] / change[positive]) if positive.any() else np.inf
                step = min(1., .99 * boundary)  # (7)
                directional = float(g @ dz)
                accepted = False
                for backtracks in range(80):
                    candidate = z + step * dz
                    candidate_slacks = s - step * change  # (27)
                    if np.all(candidate_slacks > 0.):
                        change_psi = (p.objective_difference(z, dz, step) -
                                      np.log1p(-step * change / s).sum() / tau)  # (7)
                        # Small rounding allowance is in objective units, not residual units.
                        allowance = 8. * np.finfo(float).eps * max(1., abs(p.objective(z)))
                        if np.isfinite(change_psi) and change_psi <= .01 * step * directional + allowance:
                            accepted = True
                            break
                    step *= .5
                record.update(step=step if accepted else 0., backtracks=backtracks, event='step')
                if not accepted or (np.array_equal(z, candidate) and np.array_equal(s, candidate_slacks)):
                    raise FloatingPointError('Centering line search stalled at floating-point precision')
                z, s = candidate, candidate_slacks
            if not centered:
                message = 'Inner iteration limit reached'
                break
            if p.m / tau < gap_tolerance:  # (8)
                status = 'converged'
                break
            tau *= mu  # (8)
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
        status, message = 'numerical_failure', str(exc)
    # The raw central multipliers and recovered certificate remain separate. (21), (22)
    raw = 1. / (tau * s)  # (4), (27)
    final_gap = p.m / tau  # (5)
    multipliers, lower, feasibility = raw.copy(), float('nan'), float('nan')
    if recover_certificate:
        multipliers, lower, feasibility = certificate(p, z, raw, final_gap, ledger)
    solve_time = perf_counter() - started
    started = perf_counter()
    conditions = ledger.condition_numbers() if diagnostics else (float('nan'), float('nan'))
    metrics = residuals(p, z, multipliers, 0.)
    raw_metrics = residuals(p, z, raw, 1. / tau)
    raw_metrics['r_pri'] = float(np.linalg.norm(p.constraints(z) + s, np.inf))  # (27)
    raw_metrics['surrogate_gap'] = float(s @ raw)  # (27)
    raw_metrics['complementarity_residual'] = float(np.max(np.abs(s * raw - 1./tau)))  # (27)
    result = SolverResult(z, multipliers, raw, history, ledger, status, p.objective(z),
                          setup_time, solve_time, perf_counter()-started,
                          outer_count, directions, final_gap, tau, lower, feasibility,
                          *conditions, message=message, metrics=metrics, raw_metrics=raw_metrics)
    result.internal_slacks = s.copy()  # (27)
    return result
