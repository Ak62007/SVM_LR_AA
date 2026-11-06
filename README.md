# IPM-ML

A transparent log-barrier method and a primal-dual interior-point method for
soft-margin SVM and L1 logistic regression. Both custom algorithms are written
from scratch. CVXPY/CLARABEL, CVXPY/SCS, and sklearn are baselines only.

## Reproduce

Python 3.13 was used for the recorded results. Create an environment if needed:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
```

One command runs milestone tests, all seven experiments, every figure, and the
Markdown/PDF report:

```bash
.venv/bin/python run_all.py
```

The full profile uses three timing repetitions, one BLAS thread, fixed seed
1729, the real breast-cancer dataset packaged with sklearn, n up to 10,000,
and d up to 500. No dataset download is required. Reproduction fixes numeric
inputs and algorithms; timings depend on the computer and package platform.
The existing local `.venv` is ready to use in this workspace.

For a smaller independent run, or tests alone:

```bash
.venv/bin/python run_all.py --profile smoke --repeats 1 --output /tmp/ipm-smoke
.venv/bin/python -m pytest
```

Individual experiments accept the same profile, repeats, and output flags:

```bash
.venv/bin/python experiments/05_scaling_conditioning.py --repeats 3
```

## Read the results

- [Report](report/report.md) and [PDF](report/report.pdf).
- [Theory and full derivations](theory/theory_notes.md), with stable equation
  identifiers referenced by formulas in the source.
- [Main figure](results/figures/02_gap_vs_solves.png): log gap versus cumulative
  actual linear-system solves. Only custom methods expose comparable traces.
- `results/tables/`: CSV summaries, with solver-specific iteration columns,
  separate setup/solve/diagnostic times, residuals, conditions, and failures.
- `results/traces/`: every custom iteration and its individual solve ledger.
- `results/metadata.json`: package versions, platform, seed, thread policy,
  experiment scope, and SHA256 hashes of the source.
- `results/verification.txt` and `results/milestone_checks.md`: passing test
  output and short checks for each milestone.
- [Development checks](MILESTONES.md): what was checked during implementation.

## Same formulations

For SVM, variables are `(w,b,xi)` and the objective is
`0.5*w.T@w + C*sum(xi)` with inequalities `1-y*(X@w+b)-xi <= 0`
and `-xi <= 0` (exactly 2n). For logistic regression, variables are `(w,b,v)`
and the objective is `sum(logaddexp(0,-y*(X@w+b))) + lambda*sum(v)`
with `w-v <= 0` and `-w-v <= 0` (exactly 2d). The intercept is free in both.

`v` denotes the user's epigraph `t`; `tau` is the barrier parameter.
Starts are `(w,b)=(0,0), xi=2` and `(w,b)=(0,0), v=1`.
SVC uses ordinary hinge loss and C unchanged. SAGA L1 uses `C=1/lambda`
and leaves the intercept unpenalized. Liblinear's penalized intercept and
LinearSVC's default squared hinge are avoided.

## Algorithms and accounting

The barrier method increases `tau` by `mu` and centers with Newton and
backtracking. Default inner decrement tolerance is `1e-16`; the nominal
outer gap must be below `5e-9`. Primal-dual has one Newton direction per
iteration and no inner loop; it stops only when raw primal/dual residuals
are below `1e-8` and the surrogate gap is below `5e-9`. Its early centering
fraction is capped at 0.1; a stationarity guard enables a smaller local target
and a boundary fraction approaching 1. Mehrotra is derived but not implemented.

Every attempted NumPy/SciPy solve is counted, including failed symmetric
attempts, least-squares fallbacks, and dual-certificate recovery. A direction
used to check the Newton decrement counts as a Newton direction even if no
step is accepted. `step>0` counts accepted steps. Certificate work is included
in total solve counts and solve time, with its own count. Matrix condition
numbers refer to the last Newton matrix and are computed outside solve time.

The SVM reduced path eliminates xi and multipliers, leaving a d+1 system.
`structure=False` uses the full primal Newton matrix. The logistic epigraph
variables can also be eliminated. No optimizer library, hidden ridge penalty,
or baseline call is used by a custom algorithm. All solves go through
`src/linsolve.py`; condition-number SVDs and dataset QR factorizations are
not linear-system solves.

SCS splitting iterations, CLARABEL conic IPM iterations, sklearn SMO/epochs,
and custom Newton directions have separate columns. Baseline internal solve
counts are unavailable and left blank. Baseline histories are not fabricated
for the solve-count plot.

## Accuracy and numerical limits

`barrier_gap=m/tau` is computed with that expression exactly. It certifies an
exact central point; at an approximate center the logs also show raw
stationarity and a separately recovered feasible dual certificate. The main
barrier line is explicitly a nominal central-path bound. `objective_error`
is a signed difference from CLARABEL, not a mathematical duality gap.
`certified_gap` uses an actual SVM box-dual or logistic entropy-dual bound.

The barrier solver stores affine slacks with `s <- s-alpha*G*dz` to preserve
tiny positive values. It reports the identity defect `f(z)+s` as `r_pri` and
direct inequality violations separately. Raw central multipliers and recovered
multipliers are retained separately in the returned result; CSVs distinguish
their residuals. The primal-dual solver uses `s=-f(z)` directly, so its slack
equality residual is zero by construction, with inequality violations also
checked. Equation (27) explains the floating-point distinction.

Correctness tests require custom objectives within 1e-6 relative to CLARABEL,
KKT residuals under 1e-6, gaps under 1e-8, feasible dual certificates, >=99%
sklearn prediction agreement, and exact agreement on separable cases. Tests
cover duplicate/conflicting points, constant features, tiny/huge regularization,
finite-difference derivatives and KKT directions, full/reduced systems, solve
accounting, and all seven experiments at small sizes.

Stress runs retain genuine failures. In the recorded full size study, adaptive
primal-dual runs can stall slightly above the strict stopping thresholds at
SVM n=1000,d=5 and logistic n=1000,d=500. The SVM configuration appears in
both slices. SAGA reaches its epoch cap on real-data logistic regression.
Check status and validation columns rather than assuming a returned point
converged. Large dense full solves are explicitly marked not run according
to the work/memory policy, rather than reported as actual memory failures.

The required sensitivity cases use feature scale spans 0-6 decades, Gram
conditions 10-1e10, raw/standardized features, regularization 1e-4-1e4, and
mu in {2,10,50,100}. Rescaling features with the same penalty changes the
model; it is not a pure affine-invariance experiment. A separate coordinate
test checks Newton invariance. Standard self-concordance bounds apply to the
SVM, while logistic loss is only generalized self-concordant.

## Repository

`src/problems.py` defines the shared primal objectives and constraints;
`barrier.py` and `primal_dual.py` implement the algorithms; `linsolve.py`
counts work; `metrics.py` checks residuals and dual certificates;
`baselines.py` contains all optimizer-library calls; `data.py` fixes datasets;
`plots.py` exports figures. Numbered experiment entry points delegate to
`experiments/common.py`. `report/build_report.py` generates both report formats
from saved evidence, including the complete theory appendix.

The optional kernel/dual-SVM implementation is not included. The SVM dual is
derived and used for verification; the compared optimization formulation is
always the primal. See the theory notes for primary references.
