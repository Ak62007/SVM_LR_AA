"""Shared experiment machinery. All metrics use theory equations (23)-(26)."""
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import json
import platform
import subprocess
import sys
import traceback
import numpy as np
import pandas as pd
from src import barrier, primal_dual
from src.baselines import cvxpy_solve, sklearn_solve
from src.data import blobs, standardized, breast_cancer, conditioned, SEED
from src.problems import Problem, barrier_terms
from src.linsolve import SolveLedger
from src.metrics import residuals
from src import plots

METHODS = ['barrier', 'primal_dual', 'CLARABEL', 'SCS', 'sklearn']


@dataclass
class Config:
    output: Path = Path('results')
    profile: str = 'full'
    repeats: int = 3

    def __post_init__(self):
        self.output = Path(self.output)
        for folder in ['tables', 'figures', 'traces']:
            (self.output / folder).mkdir(parents=True, exist_ok=True)

    @property
    def smoke(self):
        return self.profile == 'smoke'


def table(config, name, rows):
    frame = pd.DataFrame(rows)
    frame.to_csv(config.output / 'tables' / f'{name}.csv', index=False)
    return frame


def trace(config, name, result, reference=None):
    history = pd.DataFrame(result.history)
    if reference is not None:
        history['objective_error'] = history.objective - reference  # (23)
        for row, err in zip(result.history, history.objective_error):
            row['objective_error'] = float(err)
    history.to_csv(config.output / 'traces' / f'{name}.csv', index=False)
    pd.DataFrame(result.ledger.entries).to_csv(config.output / 'traces' / f'{name}_solves.csv', index=False)


def run_method(p, method, repeats=1, **options):
    values, setup, elapsed, diagnostics = [], [], [], []
    exception = None
    for _ in range(repeats):
        try:
            if method == 'barrier':
                r = barrier.solve(p, **options)
            elif method == 'primal_dual':
                r = primal_dual.solve(p, **options)
            elif method == 'sklearn':
                r = sklearn_solve(p)
            else:
                r = cvxpy_solve(p, method)
            values.append(r); setup.append(r.setup_time); elapsed.append(r.solve_time)
            diagnostics.append(getattr(r, 'diagnostics_time', 0.))
        except Exception as exc:
            exception = f'{type(exc).__name__}: {exc}'
            break
    if not values:
        return None, dict(method=method, problem=p.kind, status='exception', message=exception,
                          objective=np.nan, solve_time=np.nan, setup_time=np.nan,
                          newton_iterations=np.nan, solves=np.nan, last_condition=np.nan,
                          dual_residual=np.nan, primal_residual=np.nan)
    r = values[-1]
    row = dict(method=method, problem=p.kind, n=p.n, d=p.d, regularization=p.reg,
               status='repeat_exception' if exception else r.status,
               message=exception or getattr(r, 'message', ''), objective=r.objective,
               setup_time=float(np.median(setup)), solve_time=float(np.median(elapsed)),
               diagnostics_time=float(np.median(diagnostics)), repeats_completed=len(values),
               solve_time_min=min(elapsed), solve_time_max=max(elapsed),
               repeat_statuses=json.dumps([value.status for value in values]),
               failed_repetitions=sum(value.status not in {'converged', 'optimal', 'optimal_inaccurate'} for value in values))
    if method in ['barrier', 'primal_dual']:
        row.update(newton_iterations=r.newton_iterations, outer_iterations=r.outer_iterations,
                   solves=r.solves, recovery_solves=r.recovery_solves, linear_solve_seconds=float(np.median([value.ledger.seconds for value in values])),
                   final_gap=r.barrier_gap if method == 'barrier' else r.metrics['surrogate_gap'],
                   certified_gap=r.certified_gap, dual_bound=r.dual_bound,
                   certificate_feasibility=r.certificate_feasibility,
                   last_condition=r.last_condition, equilibrated_condition=r.equilibrated_condition,
                   raw_dual_residual=r.raw_metrics['dual_residual'], raw_r_pri=r.raw_metrics['r_pri'],
                   **r.metrics)  # (23), (27)
        row['validation_passed'] = bool(r.status == 'converged' and row['final_gap'] < 1e-8 and
                                         all(r.metrics[key] < 1e-6 for key in ['primal_residual', 'dual_residual',
                                             'complementarity_residual', 'dual_nonnegativity']) and
                                         r.certificate_feasibility < 1e-6)
        row['raw_kkt_passed'] = bool(r.raw_metrics['dual_residual'] < 1e-6 and
                                    r.raw_metrics['r_pri'] < 1e-6 and
                                    r.raw_metrics['complementarity_residual'] < 1e-6)
        if method == 'barrier':
            row['barrier_tau'] = r.tau
            row['barrier_m'] = p.m
        row['certificate_gap_relative'] = r.certified_gap / max(1., abs(r.objective))  # (23)
    else:
        # Baseline iterations use separate columns and unavailable solve counts stay missing. (26)
        column = {'CLARABEL': 'conic_ipm_iterations', 'SCS': 'scs_splitting_iterations',
                  'sklearn': 'sklearn_iterations'}[method]
        row.update({column: r.iterations, 'iteration_unit': r.iteration_unit,
                    'solves': np.nan, 'newton_iterations': np.nan,
                    'primal_residual': max(0., float(np.max(p.constraints(r.z))))})
        if r.multipliers is not None:
            row.update(residuals(p, r.z, r.multipliers))
    return r, row


def correctness(config):
    rows = []
    datasets = [('synthetic', *blobs(40 if config.smoke else 100, 5)),
                ('separable', *blobs(40 if config.smoke else 100, 5, separable=True))]
    if not config.smoke:
        X, Xt, y, yt = breast_cancer()
        datasets.append(('breast_cancer', X, y))
    for dataset, X, y in datasets:
        if dataset != 'breast_cancer':
            X = standardized(X)
        for kind in ['svm', 'logreg']:
            p = Problem(X, y, kind, 1.)
            reference = cvxpy_solve(p)
            sk = sklearn_solve(p)
            for method in METHODS:
                r, row = run_method(p, method, repeats=1)
                row['dataset'] = dataset
                if r is not None:
                    row['objective_error'] = r.objective - reference.objective  # (23)
                    row['relative_objective_error'] = abs(row['objective_error']) / max(np.finfo(float).eps, abs(reference.objective))  # (23)
                    row['sklearn_agreement'] = float(np.mean(p.predict(r.z) == p.predict(sk.z)))  # (26)
                    row['train_accuracy'] = float(np.mean(p.predict(r.z) == y))
                    if dataset == 'breast_cancer':
                        row['test_accuracy'] = float(np.mean(p.predict(r.z, Xt) == yt))
                    if method in ['barrier', 'primal_dual']:
                        trace(config, f'01_{dataset}_{kind}_{method}', r, reference.objective)
                rows.append(row)
    frame = table(config, '01_correctness', rows)
    # Required correctness checks fail the pipeline; stress experiments retain failures.
    custom = frame[frame.method.isin(['barrier', 'primal_dual'])]
    if not (custom.validation_passed.all() and (custom.relative_objective_error < 1e-6).all() and
            (frame.sklearn_agreement >= .99).all() and
            (frame[frame.dataset == 'separable'].sklearn_agreement == 1.).all()):
        raise AssertionError('Required correctness verification failed; see 01_correctness.csv')
    return frame


def convergence(config):
    results, rows = {}, []
    X, y = blobs(40 if config.smoke else 120, 5, overlap=1.5)
    for kind in ['svm', 'logreg']:
        p = Problem(standardized(X), y, kind)
        reference = cvxpy_solve(p)
        for method in ['barrier', 'primal_dual']:
            r, row = run_method(p, method)
            if r is not None:
                results[kind, method] = r
                trace(config, f'02_{kind}_{method}', r, reference.objective)
                # Empirical order is diagnostic, not a claimed global rate theorem. (11)
                gaps = np.array([h['barrier_gap' if method == 'barrier' else 'surrogate_gap'] for h in r.history])
                ratios = gaps[1:] / gaps[:-1]
                row['last_gap_ratio'] = float(ratios[-1]) if len(ratios) else np.nan
            rows.append(row)
    plots.convergence(results, config.output / 'figures')
    return table(config, '02_convergence', rows)


def barrier_tuning(config):
    X, y = blobs(40 if config.smoke else 120, 5)
    rows = []
    for kind in ['svm', 'logreg']:
        p = Problem(standardized(X), y, kind)
        for mu in [2., 10., 50., 100.]:
            r, row = run_method(p, 'barrier', config.repeats, mu=mu, max_outer=60)
            row['mu'] = mu
            if r is not None:
                trace(config, f'03_{kind}_mu{int(mu)}', r)
            rows.append(row)
    frame = table(config, '03_barrier_tuning', rows)
    plots.tuning(frame, config.output / 'figures')
    return frame


def solver_comparison(config):
    X, y = blobs(40 if config.smoke else 200, 8)
    datasets = [('synthetic', standardized(X), y)]
    if not config.smoke:
        X, Xt, y, yt = breast_cancer()
        datasets.append(('breast_cancer', X, y))
    rows = []
    # One untimed warm-up per method prevents import overhead dominating repeated timings.
    warm = Problem(*blobs(20, 3))
    for method in METHODS:
        run_method(warm, method)
    for dataset, X, y in datasets:
        for kind in ['svm', 'logreg']:
            started = perf_counter()
            p = Problem(X, y, kind)
            data_setup = perf_counter() - started
            for method in METHODS:
                _, row = run_method(p, method, config.repeats)
                row.update(dataset=dataset, problem_setup_time=data_setup)
                rows.append(row)
    frame = table(config, '04_solver_comparison', rows)
    plots.comparison(frame, config.output / 'figures')
    return frame


def scaling_conditioning(config):
    rows = []
    n, d = (40, 4) if config.smoke else (150, 8)
    cases = []
    for span in [0., 3., 6.]:
        cases.append(('feature_scales', span, *conditioned(n, d, 10., span)))  # (25)
    for condition in ([10., 1e10] if config.smoke else [10., 1e4, 1e7, 1e10]):
        cases.append(('correlation', condition, *conditioned(n, d, condition)))  # (25)
    for experiment, knob, X, y in cases:
        for scaling in ['raw', 'standardized']:
            data = X if scaling == 'raw' else standardized(X)  # (25)
            cond = float(np.linalg.cond(data.T @ data))  # (24)
            for kind in ['svm', 'logreg']:
                p = Problem(data, y, kind)
                for method in ['barrier', 'primal_dual']:
                    r, row = run_method(p, method, config.repeats)
                    row.update(experiment=experiment, knob=knob, scaling=scaling, gram_condition=cond)
                    if r is not None:
                        trace(config, f'05_{experiment}_{knob:g}_{scaling}_{kind}_{method}', r)
                    rows.append(row)
    # Vary regularization on one fixed dataset, independently of scaling.
    X, y = blobs(n, d)
    for reg in ([1e-4, 1e4] if config.smoke else [1e-4, .01, 1., 100., 1e4]):
        for kind in ['svm', 'logreg']:
            for method in ['barrier', 'primal_dual']:
                p = Problem(standardized(X), y, kind, reg)
                r, row = run_method(p, method, config.repeats)
                row.update(experiment='regularization', knob=reg, scaling='standardized',
                           gram_condition=float(np.linalg.cond(p.X.T @ p.X)))  # (24)
                if r is not None:
                    trace(config, f'05_regularization_{reg:g}_{kind}_{method}', r)
                rows.append(row)
    frame = table(config, '05_scaling_conditioning', rows)
    plots.sensitivity(frame, config.output / 'figures')
    return frame


def problem_size(config):
    rows = []
    # Two slices cover the requested ranges without an expensive Cartesian grid.
    sizes = ([('n', n, 5) for n in [100, 1000, 10000]] +
             [('d', 1000, d) for d in [5, 50, 500]] + [('n', 300, 5)])
    if config.smoke:
        sizes = [('n', 30, 3), ('n', 50, 3), ('d', 50, 3), ('d', 50, 5)]
    for sweep, n, d in sizes:
        X, y = blobs(n, d)
        for kind in ['svm', 'logreg']:
            p = Problem(standardized(X), y, kind)
            for method in ['barrier', 'primal_dual']:
                for structure in ['reduced', 'full']:
                    if structure == 'full' and (p.size > 400 or (kind == 'logreg' and d > 50)):
                        rows.append(dict(sweep=sweep, n=n, d=d, problem=kind, method=method,
                                         structure=structure, status='not_run_memory_work_limit',
                                         solve_time=np.nan, newton_iterations=np.nan))
                        continue
                    r, row = run_method(p, method, config.repeats, structure=structure == 'reduced')
                    row.update(sweep=sweep, structure=structure)
                    if r is not None:
                        trace(config, f'06_{sweep}_{n}_{d}_{kind}_{method}_{structure}', r)
                        row['last_solve_dimension'] = r.ledger.last_matrix.shape[0] if r.ledger.last_matrix is not None else 0
                    rows.append(row)
    frame = table(config, '06_problem_size', rows)
    plots.sizes(frame, config.output / 'figures')
    return frame


def analytic_newton(p, tau=1., tolerance=1e-8):
    z = p.initial()
    ledger = SolveLedger()
    rows = []
    for iteration in range(200):
        F, g, H = barrier_terms(p, z, tau)  # (6)
        dz = ledger.solve(H, -g)
        delta = np.sqrt(max(0., float(-tau * g @ dz)))  # (6)
        full = delta <= .25  # (10)
        step = 1. if full else 1. / (1. + delta)  # (10)
        rows.append(dict(iteration=iteration, objective=tau * F, decrement=delta, step=step,
                         phase='quadratic' if full else 'damped', solves=ledger.count))
        if delta**2 / 2. <= tolerance:
            break
        z += step * dz
        if np.any(p.constraints(z) >= 0.):
            raise AssertionError('Analytic self-concordant step left the domain')
    frame = pd.DataFrame(rows)
    if frame.iloc[-1].decrement >= 1.:
        raise AssertionError('No local certificate for theoretical bound')
    final_delta = float(frame.iloc[-1].decrement)
    omega_star = -final_delta - np.log1p(-final_delta)  # (10)
    gap_upper = frame.iloc[0].objective - frame.iloc[-1].objective + omega_star  # (10)
    damped_bound = int(np.ceil(max(0., gap_upper) / (.25 - np.log1p(.25))))  # (10)
    quadratic_bound = 0
    delta_bound = .25
    while delta_bound**2 / 2. > tolerance:
        delta_bound = (delta_bound / (1. - delta_bound))**2  # (10)
        quadratic_bound += 1
    return frame, dict(observed_steps=len(frame)-1, observed_damped=int((frame.phase == 'damped').sum()),
                       observed_quadratic=int((frame.phase == 'quadratic').sum())-1,
                       damped_bound=damped_bound, quadratic_bound=quadratic_bound,
                       total_bound=damped_bound + quadratic_bound, initial_gap_upper=gap_upper,
                       tau=tau, centering_tolerance=tolerance, applies_to='SVM analytic policy only')


def self_concordance(config):
    X, y = blobs(30 if config.smoke else 100, 5)
    rows = []
    bounds = []
    for kind in ['svm', 'logreg']:
        p = Problem(standardized(X), y, kind)
        r = barrier.solve(p)
        for h in r.history:
            rows.append(dict(problem=kind, **h))
        if kind == 'svm':
            analytic, bound = analytic_newton(p)
            bounds.append(bound)
            analytic.to_csv(config.output / 'traces' / '07_analytic_newton.csv', index=False)
    frame = table(config, '07_self_concordance', rows)
    table(config, '07_theoretical_bound', bounds)
    plots.self_concordance(frame, analytic, config.output / 'figures')
    return frame


EXPERIMENTS = [correctness, convergence, barrier_tuning, solver_comparison,
               scaling_conditioning, problem_size, self_concordance]


def cli(index=None):
    import argparse
    from threadpoolctl import threadpool_limits
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=['full', 'smoke'], default='full')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    config = Config(args.output, args.profile, args.repeats)
    with threadpool_limits(limits=1):
        if index is None:
            for number, experiment in enumerate(EXPERIMENTS, 1):
                print(f'{number:02d}: {experiment.__name__}', flush=True)
                experiment(config)
        else:
            EXPERIMENTS[index-1](config)


if __name__ == '__main__':
    cli()
