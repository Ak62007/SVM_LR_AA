# Milestone checks

The commands below are run before each milestone is marked complete. Detailed
numerical evidence is retained in results/, not inferred from a test count.

1. Theory: `python -m pytest tests/test_theory.py` passed (1 test).
   Checked stable equation numbers and the numerical/theoretical scope statements.

2. Problems and data: `python -m pytest tests/test_derivatives.py tests/test_theory.py`.
   Objective and barrier gradients/Hessians checked by central finite differences;
   Jacobian products, strict starts, prescribed conditioning, and training-only
   real-data standardization checked.

3. Baselines: `python -m pytest tests/test_baselines.py` passed (4 tests).
   CLARABEL, SCS, and sklearn objectives and predictions agree on both problems,
   with separable and nonseparable examples. SAGA avoids intercept penalization.

4. Barrier on A: targeted SVM barrier reference and prediction tests passed.
   Checked strict starts, m/tau, recovered finite dual feasibility, objective
   agreement, and counted certificate recovery. A large-C cancellation failure
   led to the affine slack recurrence in equation (27). Raw residuals remain
   visible; recovered residuals are not substituted into the raw history.

5. Primal-dual on A: targeted SVM primal-dual tests passed. Checked one Newton
   direction per iteration, raw stopping residuals, full versus reduced
   directions, finite-difference KKT Jacobian, and counted failed solves.

6. Extension to B: `python -m pytest` passed (44 tests).
   Both objectives match CLARABEL to 1e-6 relative, final KKT under 1e-6,
   gaps below 1e-8, prediction agreement >=99% (100% for separable tests),
   duplicate points, constants, C/lambda=1e-4 and 1e4, and coordinate invariance.
   The full and reduced systems agree to 2e-10 in the direction tests.

7. Experiments 01-04: full profile completed, with targeted experiment-contract
   tests passing. Custom objectives agree with CLARABEL and predictions agree
   with sklearn on synthetic, separable, and real data. Histories distinguish
   nominal, surrogate, certified, and reference-objective quantities. Iteration
   units and setup/solve/diagnostic timing are separate.

8. Experiments 05-07: full profile completed; integration and solver checks
   pass. All required sensitivity cases validate. The size study retains
   iteration-limit returns at the strict stopping floor in two distinct
   primal-dual configurations (one repeated in both slices). SAGA's real-data
   epoch cap is retained. The analytic SVM self-concordance bound is checked
   separately from the ordinary Armijo and logistic traces.

9. Report and README: the complete reproduction command has run successfully.
   The report includes every required section and all theory derivations;
   its PDF is parsed by tests and visually reviewed after Poppler rendering.
   Figure titles, problem-specific size curves, and absolute-value formulas
   were corrected during visual QA. Final test counts and exact source hashes
   are recorded in results/verification.txt and results/metadata.json.
