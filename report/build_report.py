"""Build the Markdown report and a typeset PDF directly from recorded results."""
from pathlib import Path
from html import escape
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def fmt(value):
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return '-'
    if isinstance(value, (float, np.floating)):
        return f'{value:.3g}'
    return str(value)


def markdown_table(frame, columns, labels=None):
    labels = columns if labels is None else labels
    lines = ['| ' + ' | '.join(labels) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for _, row in frame.iterrows():
        lines.append('| ' + ' | '.join(fmt(row.get(c, np.nan)).replace('|', '/') for c in columns) + ' |')
    return '\n'.join(lines)


def build_markdown(results, report_dir):
    results, report_dir = Path(results), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    tables = {name: pd.read_csv(results / 'tables' / f'{name}.csv') for name in
              ['01_correctness', '02_convergence', '03_barrier_tuning', '04_solver_comparison',
               '05_scaling_conditioning', '06_problem_size', '07_theoretical_bound']}
    correct = tables['01_correctness']
    custom = correct[correct.method.isin(['barrier', 'primal_dual'])]
    convergence = tables['02_convergence']
    sensitivity = tables['05_scaling_conditioning']
    sizes = tables['06_problem_size']
    executed = sizes[sizes.status != 'not_run_memory_work_limit']
    failures = executed[executed.status != 'converged']
    timing = tables['04_solver_comparison']
    bound = tables['07_theoretical_bound'].iloc[0]
    all_frames = [f for f in tables.values() if 'status' in f]
    failure_summary = pd.concat([f.assign(experiment=name) for name, f in tables.items() if 'status' in f], ignore_index=True)
    failure_summary = failure_summary[~failure_summary.status.isin(['converged', 'optimal', 'optimal_inaccurate', 'not_run_memory_work_limit'])]
    raw_worst = sensitivity.raw_dual_residual.max()
    recovered_worst = sensitivity.dual_residual.max()
    img = lambda name: f'![{name.replace("_", " ")}]({Path(__import__("os").path.relpath(results / "figures" / (name + ".png"), report_dir))})'
    source = ROOT / 'theory/theory_notes.md'
    report = f'''# Why do interior-point methods need so few iterations?

A reproducible comparison on soft-margin SVM and L1 logistic regression

## Introduction and central question

Iteration counts alone conceal the cost of an optimization algorithm. We compare
a log-barrier method and a primal-dual Newton method written from scratch, then
measure objective accuracy, residuals, linear-system work, and elapsed time.
The main question is whether the small number of Newton steps offsets their
matrix cost, and how conditioning changes that answer.

The correctness study contains {len(correct)} solver runs. Across its {len(custom)}
custom runs, the largest relative objective error against high-accuracy CLARABEL
is {custom.relative_objective_error.max():.3g}. The smallest prediction agreement
with sklearn is {custom.sklearn_agreement.min():.1%}. These are checks on the
listed datasets, not a claim of universal robustness. The size study records
{len(failures)} unsuccessful rows, including repeated configurations in two
slices; these are retained below with the unmet stopping criteria.

## Theory

With f(z)=Gz+h<=0 and m inequalities, use phi=-sum log(-f_i). At the exact
minimizer of F_tau=tau f0+phi, stationarity gives lambda_i=-1/(tau f_i).
Convexity makes this point minimize its Lagrangian; substituting shows the
primal-dual objective gap is exactly m/tau. This justifies a barrier outer loop
that multiplies tau by mu. At an approximate center, m/tau is a nominal bound:
stationarity, the stored slack identity, and a genuine dual certificate are
checked separately. Equations (3)-(8) in the appendix give the derivation.

The Newton direction solves H dz=-g. The squared decrement is -tau g'dz when
using the numerically scaled objective f0+phi/tau. Backtracking preserves
positive slacks and accepts a sufficient decrease. The SVM objective plus its
log barrier is standard self-concordant: the third directional derivative is
bounded by twice the 3/2 power of the second. Analytic damping has a guaranteed
objective decrease; once the decrement is small, full steps converge
quadratically. The short-step path bound is O(sqrt(m) log(m/epsilon)), with
initialization and centering assumptions. Our aggressive fixed mu values and
Armijo line search are measured separately from that theorem. Equations
(9)-(11) give both the short-step argument and a conservative fixed-mu bound.

Logistic loss satisfies a different inequality: its third derivative is bounded
by its second derivative times a design-direction norm. This generalized
self-concordance does not imply the standard SVM Newton-step bound. We make no
standard self-concordance guarantee for the logistic barrier objective.

The primal-dual method applies Newton to stationarity, affine feasibility,
and s_i lambda_i=kappa. Eliminating slack and multiplier directions leaves
H0+G' diag(lambda/s) G. There is one direction per iteration and no inner
centering loop. Near a sufficiently stationary solution the centering target
is proportional to (eta/m)^(3/2), and the boundary fraction approaches one.
This permits a faster local phase without an extra predictor solve. The
stationarity guard prevents prematurely collapsing complementarity under huge
regularization. Equations (13)-(16), (28) derive the method and its limitations.

## Problem formulations

SVM uses z=(w,b,xi), objective 0.5||w||^2+C sum xi, and constraints
1-diag(y)[X,1](w,b)-xi<=0 and -xi<=0. There are exactly 2n inequalities.
Logistic regression uses z=(w,b,v), objective sum log(1+exp(-y(Xw+b)))
+lambda sum v and constraints w-v<=0 and -w-v<=0, exactly 2d inequalities.
Every custom and CVXPY solve uses these variables and inequalities.
The intercept is unpenalized. No dual SVM or kernel implementation is used.

The synthetic data have Gaussian noise, controlled overlap, and a fixed seed
1729. Separable examples force a strict signed margin along one feature.
The real-data study uses the packaged Breast Cancer Wisconsin Diagnostic data,
a stratified 75/25 train/test split, and training-only standardization.
Its source and description are in the [sklearn dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).
All optimization comparisons use the same training rows; test accuracy is
reported as a descriptive statistic, without hyperparameter selection.

## Implementation

Only NumPy/SciPy linear algebra is called by the custom methods. A single
ledger counts every attempted solve, including failed direct attempts,
least-squares fallbacks, and certificate recovery. There is no optimizer
library inside the algorithms. Feature and identity Jacobian products are
computed without materializing full constraint matrices on the reduced path.

For SVM, eliminating xi turns the (n+d+1)-square Newton matrix into
Q+A' diag(ab/(a+b)) A of size d+1. Its formation costs O(nd^2) and its dense
factorization O(d^3), with O(nd+d^2) storage. Full dense factorization grows
as O((n+d)^3). Logistic epigraph variables can also be eliminated, adding
4ab/(a+b) to the w diagonal. Equations (18)-(20) derive both reductions.
Diagonal equilibration changes coordinates without adding regularization.

The barrier method retains small affine slacks by s_new=s-alpha G dz. It
reports the discrepancy f(z)+s and direct primal violations. Raw central
multipliers and recovered certificates remain distinct. An SVM certificate
uses box constraints and intercept balance, with a counted active-margin
least-squares candidate only if stationarity improves. The logistic dual
certificate balances sigmoid probabilities and shrinks them into the L-infinity
constraint. Equations (21)-(23), (27) explain these checks.

Timings are medians of repeated runs with one BLAS thread. The main timing
study warms each method once. Problem data construction, method setup, solve,
and condition-number diagnostics are separate fields. Custom solve time
includes certification. CVXPY setup includes canonicalization. Native solver
setup can still be part of its wall-clock solve call. Baseline internal solve
counts are unavailable and left blank. Hardware-dependent times are not
bitwise reproducible; data and algorithmic settings are fixed.

## Verification

The test suite checks objective and barrier gradients and Hessians by central
finite differences, affine Jacobian products, and the full perturbed-KKT
Jacobian against the eliminated direction. Full and reduced directions agree
to 2e-10. It checks box-dual feasibility, true dual lower bounds, final KKT
residuals below 1e-6, exact computation of m/tau, gaps below 1e-8, prediction
agreement of at least 99%, and exact agreement on separable cases.

Edge cases include duplicate and conflicting points, a constant feature,
separable and overlapping data, regularization 1e-4 and 1e4, and logistic tails
of magnitude 1e6. A coordinate transformation test checks Newton invariance
without changing the statistical model. Integration tests execute all seven
experiments at small sizes and check their numeric and artifact contracts.
Full-size stress failures are logged, not converted to passing solver returns.

{markdown_table(custom, ['dataset','problem','method','relative_objective_error','sklearn_agreement','final_gap'], ['Dataset','Problem','Method','Rel. obj. error','Agreement','Final gap'])}

## Computational comparison

The primary figure puts nominal barrier bounds and primal-dual surrogate gaps
against cumulative actual linear-system solves. The two quantities agree with
true primal-dual gaps only under their stated stationarity assumptions.
Cross markers show recovered certificate gaps after recovery work. CLARABEL
and SCS do not expose comparable per-system solve histories, so their lines
are not invented. Their endpoints and times appear in the comparison table.

{img('02_gap_vs_solves')}

The convergence examples show a local acceleration in the primal-dual method:

{markdown_table(convergence, ['problem','method','newton_iterations','solves','last_gap_ratio'], ['Problem','Method','Newton dirs.','Solves','Last gap ratio'])}

A barrier outer bound stays constant during centering and drops when tau
changes. Therefore its within-center bound trace cannot diagnose Newton's
local rate; the decrement and objective traces serve that purpose. The
primal-dual method changes both point and target each iteration. A fixed
centering fraction would impose a linear contraction floor; the adaptive
policy removes that floor near stationarity. The observed ratios support,
but do not prove, a local rate theorem for every problem.

{img('02_gap_vs_iteration_and_primal')}

Raw stationarity, target-specific complementarity, and absolute objective
error against the reference are plotted separately. A reference error is not
a certified gap and may have either sign; its absolute magnitude is plotted.

{img('02_residuals_and_objective_error')}

### Barrier growth

{markdown_table(tables['03_barrier_tuning'], ['problem','mu','outer_iterations','newton_iterations','solves','solve_time'], ['Problem','mu','Outer','Newton dirs.','Solves','Seconds'])}

{img('03_barrier_growth')}

Small mu makes more outer centers but gives closer warm starts. Larger mu
reduces the outer count and can require more centering work after each jump.
The table records this tradeoff; outer counts alone are not total work.
A Newton direction used solely to certify an inner stopping test is included
in the direction and solve counts. Accepted steps can be counted from step>0.

### Timings and iteration units

{markdown_table(timing, ['dataset','problem','method','setup_time','solve_time','newton_iterations','solves','conic_ipm_iterations','scs_splitting_iterations','sklearn_iterations'], ['Data','Problem','Method','Setup s','Solve s','Newton','Solves','Conic IPM','SCS','sklearn'])}

{img('04_solver_times')}

SCS splitting iterations are first-order iterations, CLARABEL iterations are
conic IPM steps, SVC reports SMO work, and SAGA reports epochs. These are
separate columns and are never placed on a common Newton-iteration axis.
SVC(kernel=linear) matches ordinary hinge loss and the free intercept. SAGA
uses C=1/lambda and an unpenalized intercept. Liblinear's penalized synthetic
intercept would change the logistic problem and is avoided.

## Sensitivity study

The study contains {len(sensitivity)} custom runs covering feature-scale spans
0, 3, and 6 decades, target Gram condition numbers through 1e10,
raw versus standardized features, and independent C/lambda sweeps from 1e-4
to 1e4. The CSV retains observed Gram condition numbers, final raw and
recovered residuals, solve counts, median time, status, and original and
equilibrated last Newton-matrix condition numbers.

{img('05_conditioning_iterations')}

{img('05_matrix_condition_and_residual')}

{img('05_regularization')}

The largest raw final dual residual in these required sensitivity cases is
{raw_worst:.3g}; the largest reported certificate/solver KKT dual residual is
{recovered_worst:.3g}. All {int(sensitivity.validation_passed.sum())} of
{len(sensitivity)} cases pass the explicit final validation checks.
Standardization removes column magnitudes but cannot generally remove
near-linear dependence. Positive objective curvature and barrier curvature
also mean a badly conditioned X'X does not determine the Newton matrix's
condition number by itself.

Exact Newton steps are invariant under an invertible coordinate transform:
if z=T v, Hess_v=T'Hess_z T and T dv=dz. Floating-point solves do not preserve
this identity exactly; large condition numbers amplify rounding. Moreover,
rescaling X while keeping the same Euclidean or L1 penalty changes the model.
The raw-versus-standardized comparison therefore includes a change in
regularization geometry, rather than isolating numerical affine invariance.

### Problem size and the cost of a step

{img('06_problem_size')}

The two full-profile slices use n=100,1000,10000 at d=5 and d=5,50,500 at
n=1000, plus n=300 for an additional dense comparison. They are not a complete
Cartesian grid. Full systems above the explicit work/memory limit are marked
not run; this is not an out-of-memory claim. At n=10000 the reduced SVM solves
a 6-square system rather than a 10006-square one. Dense d=500 factorization
still costs substantially more per direction than d=5, even after eliminating
slacks. Logistic constraints depend on d, so its full system can remain small
when n is large and d is small.

{markdown_table(executed[(executed.n == executed.n.max()) | (executed.d == executed.d.max())], ['problem','method','n','d','structure','newton_iterations','solve_time','status'], ['Problem','Method','n','d','System','Newton','Seconds','Status'])}

### Self-concordance

{img('07_self_concordance')}

For the separate SVM trajectory that uses analytic damping followed by full
steps at decrement <=1/4, there are {int(bound.observed_steps)} accepted steps:
{int(bound.observed_damped)} damped and {int(bound.observed_quadratic)} local.
The computable conservative bound is {int(bound.total_bound)} steps
({int(bound.damped_bound)} damped plus {int(bound.quadratic_bound)} local).
Its starting suboptimality upper bound is {bound.initial_gap_upper:.3g},
obtained from the final local certificate. The large slack between bound and
observation is expected: a global worst-case guarantee does not predict a
particular dataset. The ordinary Armijo trajectories are labeled by decrement
regions; the logistic trace has no standard self-concordance bound.

## Discussion: limits and what went wrong

The successful correctness cases establish numerical agreement on matching
formulations, not that every requested size converges at the stringent raw
stopping tolerances. Failed solver returns across the recorded experiments are:

{markdown_table(failure_summary, ['experiment','dataset','problem','method','n','d','status','final_gap','dual_residual'], ['Experiment','Data','Problem','Method','n','d','Status','Final gap','Dual res.']) if len(failure_summary) else 'No solver returned a failure status in this profile.'}

The full-profile SAGA baseline can reach its epoch cap despite close objective
agreement. The full-profile adaptive primal-dual runs can stall just above
the 1e-8 gap/residual thresholds on the SVM n=1000,d=5 and logistic
n=1000,d=500 cases. The SVM case appears in both size slices. Backtracking
shrinks steps at the floating-point residual floor; keeping the strict stopping
criteria yields an iteration-limit status. The final gaps and raw residuals
remain visible. These runs must not be described as converged based only on
objective agreement. The logs show exactly how many directions were spent.

Two implementation issues were corrected during development. Recomputing
tiny SVM slacks by subtracting nearly equal margins caused large-C centering
to stall; the affine slack recurrence exposes rather than conceals its identity
defect. An active-margin certificate recovery could worsen stationarity after
feature scaling; the candidate is now accepted only if it improves the balanced
box projection. Raw and recovered multipliers remain available in memory,
and their metrics are separate in the tables.

Few Newton steps do not imply universally low cost. Matrix construction can
outweigh factorization, dense d-cubed work limits large feature counts, and
small slacks can amplify arithmetic error. SCS targets a different per-step
cost/accuracy tradeoff. The optional SVM dual, kernels, sparse factorizations,
Mehrotra correction, and homogeneous self-dual infeasibility detection are
outside this implementation; their absence limits generalization to other
problem classes. Equation (17) derives the optional correction without
claiming that it was benchmarked.

## Conclusion

On the correctness data, both custom methods agree with CLARABEL and sklearn
at the requested tolerances. The measured primal-dual method uses fewer
Newton systems than repeated barrier centering. Eliminating the SVM slacks
makes large sample counts practical when d is small. Conditioning, feature
geometry, and floating-point stopping floors remain visible limits. The
solve-count figure, raw CSV histories, separate iteration units, and failure
statuses make these conclusions auditable.

## Reproduction and provenance

Run `.venv/bin/python run_all.py` from the repository root. It checks the
milestones, runs all seven full-profile experiments with three repetitions,
rebuilds this Markdown and PDF report, and runs the final tests. Use
`--profile smoke --repeats 1 --output /tmp/ipm-smoke` for a small independent
run. `results/metadata.json` records package versions, platform, seed, thread
policy, and source hashes. `results/verification.txt` contains test output.
Tables and traces are machine-readable CSV; figures are PNG and PDF.

The theoretical derivations follow [Boyd and Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).
The generalized logistic analysis is motivated by [Bach](https://www.di.ens.fr/~fbach/bach_ejs_self_concordance.pdf).
The appendix below supplies the derivations used by the code.

# Appendix: complete derivations

'''
    report += source.read_text().replace('# Interior-point methods for two machine-learning problems', '## Theory notes and equation index', 1)
    path = report_dir / 'report.md'
    path.write_text(report)
    return path


def inline(text):
    text = escape(text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<link href="\2" color="#167d9a">\1</link>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`([^`]+)`', r'<font name="Courier">\1</font>', text)
    return text


def render_pdf(markdown_path, pdf_path):
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table,
                                   TableStyle, Preformatted, PageBreak, KeepTogether)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.pagesizes import A4
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ReportBody', fontName='Helvetica', fontSize=9.2,
                              leading=13.2, spaceAfter=7, textColor=colors.HexColor('#263342')))
    styles.add(ParagraphStyle(name='ReportTitle', fontName='Helvetica-Bold', fontSize=24,
                              leading=29, spaceAfter=16, textColor=colors.HexColor('#12364a')))
    styles.add(ParagraphStyle(name='ReportH1', fontName='Helvetica-Bold', fontSize=16,
                              leading=21, spaceBefore=14, spaceAfter=9, keepWithNext=True,
                              textColor=colors.HexColor('#12364a')))
    styles.add(ParagraphStyle(name='ReportH2', fontName='Helvetica-Bold', fontSize=12.5,
                              leading=17, spaceBefore=12, spaceAfter=7, keepWithNext=True,
                              textColor=colors.HexColor('#167d9a')))
    styles.add(ParagraphStyle(name='ReportH3', fontName='Helvetica-Bold', fontSize=10.2,
                              leading=14, spaceBefore=10, spaceAfter=5, keepWithNext=True))
    styles.add(ParagraphStyle(name='TableCellSmall', fontName='Helvetica', fontSize=6.5,
                              leading=8.3, spaceAfter=0, splitLongWords=True))
    styles.add(ParagraphStyle(name='CaptionSmall', fontName='Helvetica-Oblique', fontSize=8,
                              leading=11, textColor=colors.HexColor('#596676'), spaceAfter=12))
    styles.add(ParagraphStyle(name='EquationCode', fontName='Courier', fontSize=7.5,
                              leading=10.2, spaceBefore=5, spaceAfter=9, leftIndent=9))
    page_width, page_height = A4
    margin = 43
    width = page_width - 2 * margin
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=margin, rightMargin=margin,
                            topMargin=47, bottomMargin=43, title='Interior-point methods for machine learning',
                            author='IPM-ML reproducible harness')
    story = []
    lines = Path(markdown_path).read_text().splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            title = line[level:].strip()
            if title == 'Appendix: complete derivations':
                story.append(PageBreak())
            style = 'ReportTitle' if level == 1 and not story else f'ReportH{min(level,3)}'
            story.append(Paragraph(inline(title), styles[style]))
            index += 1
        elif line.startswith('!['):
            match = re.match(r'!\[([^]]+)\]\(([^)]+)\)', line)
            path = Path(markdown_path).parent / match.group(2)
            w, h = ImageReader(str(path)).getSize()
            ratio = min(width / w, 290. / h)
            image = Image(str(path), width=w * ratio, height=h * ratio)
            story.append(KeepTogether([image, Spacer(1, 4),
                                      Paragraph(inline(match.group(1)), styles['CaptionSmall'])]))
            index += 1
        elif line.startswith('| '):
            data = []
            while index < len(lines) and lines[index].startswith('| '):
                cells = [c.strip() for c in lines[index].strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in cells):
                    data.append([Paragraph(inline(c), styles['TableCellSmall']) for c in cells])
                index += 1
            count = len(data[0])
            table = Table(data, colWidths=[width / count] * count, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e4eff2')),
                ('LINEBELOW', (0, 0), (-1, 0), .7, colors.HexColor('#167d9a')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f7f8')])]))
            story.extend([table, Spacer(1, 12)])
        elif line.startswith('    '):
            code = []
            while index < len(lines) and (lines[index].startswith('    ') or not lines[index].strip()):
                code.append(lines[index][4:] if lines[index].startswith('    ') else '')
                index += 1
            story.append(Preformatted('\n'.join(code).rstrip(), styles['EquationCode']))
        else:
            paragraph = []
            while index < len(lines) and lines[index].strip() and not lines[index].startswith(('#','| ','![','    ')):
                paragraph.append(lines[index].strip())
                index += 1
            story.append(Paragraph(inline(' '.join(paragraph)), styles['ReportBody']))
    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#b7c9d2'))
        canvas.line(margin, page_height - 31, page_width - margin, page_height - 31)
        canvas.setFillColor(colors.HexColor('#596676'))
        canvas.setFont('Helvetica', 8)
        canvas.drawString(margin, page_height - 23, 'IPM-ML / Reproducible numerical study')
        canvas.drawRightString(page_width - margin, 24, str(document.page))
        canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def build_report(results=ROOT / 'results', report_dir=ROOT / 'report'):
    markdown = build_markdown(results, report_dir)
    pdf = Path(report_dir) / 'report.pdf'
    render_pdf(markdown, pdf)
    return markdown, pdf


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'report')
    args = parser.parse_args()
    for path in build_report(args.results, args.report_dir):
        print(path)
