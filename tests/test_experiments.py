"""Run every experiment at small sizes and verify numerical and artifact contracts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits


def test_every_experiment_writes_readable_figures_and_tables(experiment_output):
    from PIL import Image
    output = experiment_output
    for index in range(1, 8):
        assert list((output / 'tables').glob(f'{index:02d}_*.csv'))
        assert list((output / 'figures').glob(f'{index:02d}_*.png')) or index == 1
    for image in (output / 'figures').glob('*.png'):
        with Image.open(image) as img:
            img.verify()
        assert image.with_suffix('.pdf').stat().st_size > 1000
    frame = pd.read_csv(output / 'tables/01_correctness.csv')
    custom = frame[frame.method.isin(['barrier', 'primal_dual'])]
    assert custom.validation_passed.all()
    assert (custom.relative_objective_error < 1e-6).all()
    assert (frame.sklearn_agreement >= .99).all()


def test_experiment_invariants(experiment_output):
    output = experiment_output
    tuning = pd.read_csv(output / 'tables/03_barrier_tuning.csv')
    assert set(tuning.mu) == {2., 10., 50., 100.}
    assert tuning.validation_passed.all()
    timing = pd.read_csv(output / 'tables/04_solver_comparison.csv')
    for method, column in [('SCS', 'scs_splitting_iterations'), ('CLARABEL', 'conic_ipm_iterations'),
                           ('sklearn', 'sklearn_iterations')]:
        rows = timing[timing.method == method]
        assert rows[column].notna().all()
        assert rows.newton_iterations.isna().all() and rows.solves.isna().all()
    sizes = pd.read_csv(output / 'tables/06_problem_size.csv')
    pivot = sizes.pivot_table(index=['sweep', 'n', 'd', 'problem', 'method'],
                             columns='structure', values='objective')
    np.testing.assert_allclose(pivot.full, pivot.reduced, rtol=1e-6, atol=1e-7)
    bound = pd.read_csv(output / 'tables/07_theoretical_bound.csv').iloc[0]
    assert bound.observed_damped <= bound.damped_bound
    assert bound.observed_quadratic <= bound.quadratic_bound
    assert bound.observed_steps <= bound.total_bound
    for trace in (output / 'traces').glob('02_*_solves.csv'):
        ledger = pd.read_csv(trace)
        assert ledger.seconds.ge(0.).all()
        assert ledger.success.all()
    trace = pd.read_csv(output / 'traces/02_svm_primal_dual.csv')
    ratios = trace.surrogate_gap.to_numpy()[1:] / trace.surrogate_gap.to_numpy()[:-1]
    assert ratios[-1] < .1
