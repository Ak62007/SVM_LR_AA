# Reproduction milestone checks

- Theory: passed. Numbered derivations and numerical scope checked.
- Problems and data: passed. Finite-difference gradients/Hessians, tails, conditioning, and training-only scaling checked.
- Baselines: passed. Matched primal objectives, intercepts, and predictions checked.
- Barrier on SVM: passed. Reference objectives, raw/recovered KKT, gap, and edge cases checked.
- Primal-dual on SVM: passed. Raw stopping, reference objectives, and edge cases checked.
- Both problems: passed. Full/reduced Newton directions, KKT Jacobian, accounting, predictions, and affine invariance checked.
- Experiments 01-04: passed. Correctness, convergence, timing units, and barrier tuning contracts checked.
- Experiments 05-07: passed. Conditioning outputs, reduced/full equivalence, and analytic self-concordance bound checked; full-size failure statuses retained.
- Report and README: passed. PDF parsing, complete sections, equation appendix, all smoke experiments, and final full test suite checked.
