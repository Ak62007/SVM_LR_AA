#!/usr/bin/env python3
"""One command verifies the milestones and reproduces all figures and the report."""
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version
from time import perf_counter
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
os.environ['PYTHONHASHSEED'] = '1729'
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[name] = '1'


def verify(arguments, log):
    command = [sys.executable, '-m', 'pytest', '-q', *arguments]
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end='', flush=True)
    log.write('$ ' + ' '.join(command) + '\n' + result.stdout + '\n')
    log.flush()
    if result.returncode:
        raise SystemExit(result.returncode)


def metadata(config, started):
    from threadpoolctl import threadpool_info
    sources = [p for folder in ['src', 'experiments', 'tests', 'theory', 'report']
               for p in (ROOT / folder).rglob('*') if p.suffix in {'.py', '.md'} and p.name != 'report.md']
    sources += [ROOT / 'run_all.py', ROOT / 'requirements.txt', ROOT / 'requirements.lock', ROOT / 'README.md', ROOT / 'MILESTONES.md', ROOT / 'pytest.ini']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), profile=config.profile,
                repeats=config.repeats, seed=1729, python=sys.version, platform=platform.platform(),
                machine=platform.machine(), packages={p: version(p) for p in
                    ['numpy','scipy','matplotlib','pandas','cvxpy','clarabel','scs','scikit-learn',
                     'pytest','reportlab','pypdf','threadpoolctl']},
                blas_policy='one thread', threadpools=threadpool_info(),
                source_sha256=hashes, total_seconds=perf_counter()-started,
                counts='Every attempted custom solve, including fallbacks and certification; baseline counts unavailable',
                data='Seeded synthetic blobs and packaged sklearn breast cancer; training-only scaling',
                size_policy='Two slices; full dense systems above 400 variables excluded; logistic full d>50 excluded')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['full', 'smoke'], default='full')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    from threadpoolctl import threadpool_limits
    from experiments.common import Config, EXPERIMENTS
    from report.build_report import build_report
    config = Config(args.output.resolve(), args.profile, args.repeats)
    started = perf_counter()
    note_lines = []
    with (config.output / 'verification.txt').open('w') as log, threadpool_limits(limits=1):
        for label, tests, note in [
            ('Theory', ['tests/test_theory.py'], 'Numbered derivations and numerical scope checked.'),
            ('Problems and data', ['tests/test_derivatives.py'], 'Finite-difference gradients/Hessians, tails, conditioning, and training-only scaling checked.'),
            ('Baselines', ['tests/test_baselines.py'], 'Matched primal objectives, intercepts, and predictions checked.'),
            ('Barrier on SVM', ['tests/test_solvers.py', '-k', 'svm and barrier'], 'Reference objectives, raw/recovered KKT, gap, and edge cases checked.'),
            ('Primal-dual on SVM', ['tests/test_solvers.py', '-k', 'svm and primal_dual'], 'Raw stopping, reference objectives, and edge cases checked.'),
            ('Both problems', ['tests/test_solvers.py'], 'Full/reduced Newton directions, KKT Jacobian, accounting, predictions, and affine invariance checked.')]:
            print(f'Check: {label}', flush=True)
            verify(tests, log)
            note_lines.append(f'- {label}: passed. {note}')
        for index, experiment in enumerate(EXPERIMENTS, 1):
            print(f'Experiment {index:02d}: {experiment.__name__}', flush=True)
            experiment(config)
            if index == 4:
                verify(['tests/test_experiments.py'], log)
                note_lines.append('- Experiments 01-04: passed. Correctness, convergence, timing units, and barrier tuning contracts checked.')
            if index == 7:
                verify(['tests/test_experiments.py', 'tests/test_solvers.py'], log)
                note_lines.append('- Experiments 05-07: passed. Conditioning outputs, reduced/full equivalence, and analytic self-concordance bound checked; full-size failure statuses retained.')
        report_dir = ROOT / 'report' if config.output == (ROOT / 'results').resolve() else config.output / 'report'
        build_report(config.output, report_dir)
        verify([], log)
        note_lines.append('- Report and README: passed. PDF parsing, complete sections, equation appendix, all smoke experiments, and final full test suite checked.')
        (config.output / 'milestone_checks.md').write_text('# Reproduction milestone checks\n\n' + '\n'.join(note_lines) + '\n')
        data = metadata(config, started)
        (config.output / 'metadata.json').write_text(json.dumps(data, indent=2, default=str) + '\n')
    print(f'Complete: {config.output}; report: {report_dir}', flush=True)


if __name__ == '__main__':
    main()
