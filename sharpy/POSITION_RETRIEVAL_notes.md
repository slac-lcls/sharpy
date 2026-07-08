# Position Retrieval — Investigation Notes

Reference: arXiv:1209.4924 (Marchesini et al., Inverse Problems 2013), Section IV.

## Verdict: always use `method="diag"`

`position_solve_coupled` was investigated exhaustively across six scenarios (Findings 12–17).
**No scenario was found where it outperforms `position_solve_diag`.** It actively diverges
in the full AP loop. Use `method="diag"` everywhere.

---

## Finding 12 — Resonance condition

`position_solve_coupled` Hessian goes indefinite when `r2 * step ≥ 1.0` (sharp threshold).
Pre-resonance instability already at `r2 * step ≥ ~0.7` — coupled diverges catastrophically
while diagonal fails gracefully. Reproducer: `lam_sweep.py`.

## Finding 13 — Capture range

Same capture basin as diagonal for `r2·step ≤ 0.60`. Basin ≈ `1/(r2·nx)` px (Taylor validity).
Prior claim of "faster convergence on correlated drift" (hand-off notes) was NOT confirmed.

## Finding 14 — Full capture range benchmark (`position_capture_range_test.py`)

Swept i.i.d. and ramp shifts across r2 ∈ {0.06, 0.10, 0.15, 0.20}, max|ξ| 0.1–2.0 px.
Result: coupled has NO capture-range or convergence advantage on either pattern.
- Ramp does not help coupled; capture is limited by absolute per-frame shift, not pairwise diff.
- At r2·step=0.80, coupled diverges at max|ξ|=1.0 px while diagonal succeeds (0.049 px).

## Finding 15 — Timing (`test_coupled_timing.py`, nnx=48)

spsolve = 98.2% of wall time (4608×4608 system, nnz=260352).
GPU braket (`zQQz2.cu`) = 0.3%.
**Fix implemented:** replaced spsolve with GPU CG (`cupyx.scipy.sparse.linalg.cg`, maxiter=500).
Note: CuPy 14 does not accept `tol` kwarg — omit it.

## Finding 16 — Long-range drift (`test_longrange_drift.py`)

1D scan geometry: N frames, step=16 (50% overlap), linear accumulating drift (NBR_STEP=0.10 px/frame).
Geometry rationale: pairwise neighbor diff = 0.10 px (small), global drift = N×0.10 px (grows with N).

| N   | total drift | diag result  | ok? | coupled result | ok? |
|-----|-------------|--------------|-----|----------------|-----|
| 10  | 0.90 px     | 3.7e-04      | YES | 3.7e-04        | YES |
| 20  | 1.90 px     | ~0 (conv)    | YES | ~0 (conv)      | YES |
| 50  | 4.90 px     | converges    | YES | converges      | YES |
| 100 | 9.90 px     | partial conv | NO  | diverges       | NO  |
| 200 | 19.9 px     | —            | NO  | diverges worse | NO  |

Diagonal and coupled are **identical** at small N. Coupled diverges first at large N (CG hits maxiter=500
on the ill-conditioned 200×200 sparse system). Hypothesis "coupled better for long-range drift" refuted.

## Finding 17 — E2E in `Alternating_projections_position` (`test_coupled_e2e.py`)

N=50, step=16, NBR_STEP=0.10 px, total_drift=4.90 px, 200 AP iters, position_start=20.

| Init case         | Method  | dr_init | dr_final | verdict            |
|-------------------|---------|---------|----------|--------------------|
| blind (ξ=0)       | diag    | 1.443   | 1.441    | stuck (needs warm start) |
| blind (ξ=0)       | coupled | 1.443   | 1.443    | stuck              |
| oracle (ξ=truth)  | **diag**| 0.000   | **0.013**| converges ✓        |
| oracle (ξ=truth)  | coupled | 0.000   | 0.197    | diverges from truth|
| noisy (+0.23 px)  | diag    | 0.228   | 0.229    | stable plateau     |
| noisy (+0.23 px)  | coupled | 0.228   | 0.910    | diverges 4×        |

Cold-start: neither solver self-bootstraps from ξ=0 at this drift scale.
**Bootstrap path: `drift_fit=True`** in `Alternating_projections_position` (committed 2c8630b),
then `method="diag"` for sub-pixel refinement.

## MATLAB verification (`test_H_compare.py`)

Python `position_solve_coupled` is a correct port of MATLAB `fit_shift.m` / `framesmul4.c` (`dotp` kernel):
- `ab = conj(ba)` to machine precision (2.8e-18) → `2·Re(ab+ba) = 2·Re(ba)` = MATLAB formula ✓
- `taylor_shift_probe` ↔ MATLAB `Taylor_shift`: identical ✓
- H is symmetric PSD (verified eigenvalues) ✓

MATLAB `framesmul4.c` latent bug: `dyi` cast as `size_t` (unsigned). Negative wrapped `dy`
(possible in 2D hex scans when `iyt[col] > iyt[row]`) → huge size_t → loop underflow.
Does NOT affect 1D scans (ty=0 → dyi=0 always).

---

## Open work

1. **hand-off-to-diag refinement after drift_fit** (`drift_fit=True` bootstraps to ~1 px; diag refines to ~0.01 px — test this pipeline end-to-end)
2. **Fig 7 reproduction** (`position_fig7.py` scaffold exists)
3. **Prelocalize bootstrap** (`prelocalize=True` + `diag_method="exact"`, Section III.A, basin-free coarse stage)
4. Model-constrained jitter refine after drift fit
