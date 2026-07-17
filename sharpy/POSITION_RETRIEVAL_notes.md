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

## Finding 18 — Fig 7 reproduction & prelocalize/exact investigation (2026-07-08)

**Script:** `position_fig7.py` (fixed `position_simulate.py` GPU compat bugs: `cropmat` numpy→cupy,
`make_probe` cupy→numpy, `map_frames`/`shift_probe_fourier` xp conversion).

**Geometry:** 16×16=256 frames, nx=32, step=3.5, r1=0.075, r2=0.255, contrast=4.1 (gold balls,
high-contrast). Taylor capture basin: ~1/(r2·nx) = 1/(0.255·32) ≈ **0.12 px**.

**`diag_method="taylor"` (baseline):**

| AMP (px) | xi RMS (px) | eps_xi @1000 it | Absolute error | verdict |
|----------|-------------|-----------------|----------------|---------|
| 0.05     | 0.029       | 0.127           | ~7 mpx         | converging (log) |
| 0.10     | 0.058       | 0.121           | ~7 mpx         | converging (log) |
| 0.20     | 0.116       | 0.107           | ~17 mpx        | converging (log) |
| 0.40     | 0.231       | 0.143           | ~86 mpx        | converging (log) |

eps_xi is RELATIVE: `sqrt(||xi_hat - xi_truth||² / ||xi_truth||²)`. At AMP=0.10, 7 mpx absolute is
excellent. Convergence is logarithmic (0.34@200→0.093@2000 iters, still decreasing). Not a gauge-
freedom issue (global offset < 0.001 px). Paper's near-zero eps_xi requires ~5000+ iters or larger
AMP to get good normalization.

**`prelocalize=True` (sub-pixel errors):** harmful — false-positive integer detections corrupt the
map, eps_xi sticks at 0.55 vs 0.14 without. `prelocalize` is ONLY for integer-level errors (> 0.5 px)
where `multiplex_prelocalize` can match data against the integer candidate comb. At 2 px errors the
image stays too flat/blurry (no object contrast) for the comb to score, so it also fails.

**`diag_method="exact"` bug & fix (`Solvers.py:probe_and_norm`):**

Root cause: `probe_and_norm` always used `taylor_shift_probe` to build the AP frames
(`frames = obj * P_taylor(xi)`), but `position_solve_diag(method="exact")` re-linearized
with the Fourier-exact probe `P_exact(xi)`. The residual `zu = frames - P_exact(xi)*psi`
contained a spurious `(P_taylor - P_exact)*obj` term that drove xi away from truth → diverged.

**Fix:** `probe_and_norm` now uses `shift_probe_fourier(dp["O"], xi_x, xi_y)` when
`diag_method="exact"`, so frames and the position solver use a consistent probe model.

Post-fix results (AMP=0.40, 1000 iters):

| diag_method | eps_xi @1000 | img_mse @1000 | vs taylor |
|-------------|-------------|--------------|-----------|
| taylor      | 0.136       | 4.7e-3       | baseline  |
| **exact**   | **0.123**   | **2.8e-3**   | 10% better xi, 40% better img |

At AMP=0.10 (within Taylor basin) they're equal — the truncation error is negligible
there. Advantage of `exact` grows with AMP (outside the Taylor basin).

---

## Finding 19 — MATLAB code analysis + RAAR solver + 4-way comparison (2026-07-08)

### MATLAB code findings (`horse_shoe_x1.m` + `doraar0_shift.m`)

1. **Outer loop: RAAR, not AP.** `doraar0_shift.m` uses RAAR (Luke 2005) with β=0.75:
   `xno_new = (1-2β)(Pf(x) - Po(Pf(x))) + β·xno`.  AP was never the MATLAB solver.
2. **Position solver: coupled, not diagonal.** `fit_shift.m` builds the full 2k×2k block
   system. But coupled uses Taylor shift (finite-difference `gradient()`) for data AND solve.
3. **Probe parameters:** `r1=.025*nx*3=0.075*nx`, `r2=.085*nx*3=0.255*nx` → R1=0.075, R2=0.255.
   `position_fig7.py` was using R2=0.19 (wrong); corrected to R2=0.255.
4. **Phase sync / probe update: disabled** for Fig 7. `options=[1/eps,1/eps,1/eps,1]` means
   probe, phase, I0 updates all fire every ~4.5e15 iters (never). Position updates every iter.
5. **Step clip: already present** in our Python (max_step=0.5 px, since eb9d2f4). NOT the cause
   of coupled divergence in AP.
6. **Model consistency:** MATLAB generates data using Taylor shift (finite-difference derivatives),
   same model the solver inverts → machine-precision convergence. Our `position_simulate.py`
   generates data with `shift_probe_fourier` (exact Fourier shift) → inherent model mismatch
   even with `diag_method="exact"` → logarithmic convergence only.

### RAAR_position implementation

New function `Solvers.RAAR_position` (separate from `Alternating_projections_position`):
- Same interface as AP version; adds `beta=0.75` parameter
- Maintains `xo` (overlap part) + `xno` (non-overlap part); `x = xo + xno`
- Each iter: `xm = Pf(x)`, `xo_new = Po(xm)`, `xno_new = (1-2β)(xm-xo_new) + β·xno`
- Verified by `position_raar_debug.py` (5/5 tests PASS):
  formula, convergence, diag position, coupled position, β=0 limit

### 4-way comparison: AP vs RAAR × diag(exact) vs coupled

**Setup:** seed=0, 256 frames (16×16), nx=32, step=3.5, R2=0.255, CONTRAST=4.1,
A=0.40 px/component (uniform [-A,A]), mean|ξ|=0.308 px = 2.52 res_el, position_start=100.

| Method | eps_xi @100 | eps_xi @2000 | img_mse @2000 |
|---|---|---|---|
| AP + diag (Taylor) | 0.973 | 0.115 | 4.4e-3 |
| AP + diag + exact | 0.973 | 0.092 | 1.1e-3 |
| RAAR + coupled | 0.964 | 0.082 | 4.3e-3 |
| **RAAR + diag + exact** | **0.976** | **0.061** | **6.1e-4** |

**Winner: RAAR + diag + exact.** 47% lower eps_xi, 7× lower img_mse vs AP+diag baseline.
RAAR+coupled without exact re-linearization ≈ AP+diag+exact — `diag_method="exact"` matters
more than switching diag→coupled.

**Why paper converges to 1e-10 but we don't:**
- MATLAB stores ε²_ξ (no sqrt): paper's "1e-10" = ε_ξ=1e-5 in our convention (still far)
- MATLAB generates data with Taylor shift (model-consistent) → zero residual at solution
- Our exact Fourier-shift simulation is a harder, more realistic test; convergence is
  logarithmic and limited by image quality, not solver correctness

---

## Finding 20 — Model-consistent simulation matches paper (2026-07-08)

**Script:** `position_taylor_match.py` + `position_simulate.py` (`shift_model` param added).

**Root cause of logarithmic vs machine-precision convergence:**
MATLAB `Poverlap_branch2` generates data with `Taylor_shift` (finite-difference gradients)
— the exact same model the solver inverts. Our default `shift_model="fourier"` uses the
exact band-limited Fourier phase ramp, which the Taylor solver can only approximate
(0.8% relative error at 0.4 px). Neither `diag_method="exact"` nor `diag_method="taylor"`
can fully cancel this mismatch → logarithmic floor.

**`shift_model="taylor"` in `position_simulate.py`:** generates data with `taylor_shift_probe`
(same 2nd-order Taylor model as the solver). With RAAR β=0.75 + diag + taylor:

| iter | eps_xi | img_mse |
|---|---|---|
| 100  | 1.000 | 7.9e-2 |
| 500  | 0.131 | 2.5e-3 |
| 1000 | 0.089 | 1.4e-3 |
| 2000 | 0.042 | 5.3e-4 |
| 5000 | 2.1e-3 | 5.1e-5 |
| 8000 | 7.8e-6 | 1.8e-7 |
| 10000 | **2.5e-7** | **1.1e-8** |

**Two-phase convergence:**
- Phase 1 (iters 100–3000): slow logarithmic decay — image still too blurry to give a
  sharp reference for position refinement; eps_xi halves every ~1000 iters.
- Phase 2 (iters 3000–10000): **superlinear** — image sharp enough; position and image
  errors enter quadratic convergence together. eps_xi drops 4 decades in 5000 iters.

Paper's "1e-10" is ε²_ξ (MATLAB stores no-sqrt metric). Our ε_ξ=2.5e-7 at 10k iters →
ε²=6e-14, below paper's claim. **Full reproduction confirmed.**

**Fourier vs Taylor simulation comparison at 2000 iters (RAAR+diag, matched solver):**

| shift_model | diag_method | eps_xi @2000 | img_mse @2000 |
|---|---|---|---|
| fourier | exact   | 0.061 | 6.1e-4 |
| taylor  | taylor  | 0.042 | 5.3e-4 |

Taylor-consistent converges ~30% faster to the 2000-iter checkpoint, and then unlocks
superlinear convergence past ~3000 iters that the Fourier simulation cannot reach.

---

## Finding 21 — Coupled vs diag: why oracle image ≠ AP loop (2026-07-08)

**Source:** `position_compare.py` / `position_exp.ipynb` (oracle test) vs Findings 12–17 (AP loop).

### The apparent contradiction

`position_compare.py` (notebook) shows coupled reaching **1e-16** in 15 iters while diag reaches 1e-11
— coupled looks clearly superior. But Findings 12–17 show coupled diverging in every AP-loop test.

### Why they differ: oracle image vs estimated image

**`position_compare.py` setup:**
- True image held **fixed** (ground truth passed directly to position solver)
- Data generated with Taylor shift (model-consistent)
- Pure position-solver benchmark, no outer loop
- Uses r2=0.20, step=4 → r2·step=0.80 (below resonance at 1.0)

**AP/RAAR loop setup (position_fig7, Findings 16–17):**
- Image is estimated jointly; starts as all-ones
- Image is blurry and phase-corrupted for the first ~3000 iters (phase 1)
- Position updates start at iter 100 into a wrong image

### Root cause

Coupled's full 2k×2k Hessian has off-diagonal blocks `∑_i conj(ψ_row)·ψ_col` that
encode cross-frame correlations **through the current image estimate**. With the oracle
image those terms are correct → super-accurate Newton step → 1e-16 in 15 iters.

In the AP loop the image estimate is blurry/phase-wrong → off-diagonal entries encode
incorrect correlations → corrupted Newton direction → divergence.

Diag ignores the off-diagonal blocks entirely (per-frame 2×2 only) → immune to image-
estimation error → stable throughout.

Additionally, `position_fig7` uses r2=0.255 → r2·step=0.893 (near resonance at 1.0),
so even small image errors push the Hessian toward indefiniteness.

### Conclusion

**Coupled is provably better when the image is known.** In the joint-estimation problem
the image is unknown and estimated poorly for thousands of iters; coupled's extra terms
are harmful rather than helpful until the image converges. Diag's robustness to that
corrupted image is exactly what makes it the right choice for the iterative outer loop.

The MATLAB paper's coupled solver works because MATLAB never jointly estimates the image
in `doraar0_shift.m` — it uses a warm-started, nearly-converged image. Our Python tests
start from scratch (img=ones), which is a much harder regime for coupled.

---

## Finding 22 — ADMM position solver: non-convexity kills dual variables (2026-07-15)

### Setup

Implemented `ADMM_position` in `Solvers.py` following the Boyd ADMM split:

```
min_{ψ, Z} (||FFT(Sⱼψ · p)||² - d_j)²  s.t.  Z = FFT(Sⱼψ · p)
```

Z-update (proximal for Fourier magnitude constraint): `Z_mag = (2√d + ρ|V|)/(2 + ρ)`  
ψ-update (least-squares overlap): `ψ = Σⱼ conj(pⱼ)·ifft(Z_j + U_j) / ||p||²`  
Dual update: `U += dual_step · (FFT(Sⱼψ·p) − Z)`

### Findings

**Bug 1 — Wrong proximal formula.** Initial implementation used `(ρ|V|+√d)/(ρ+1)`. Correct derivation of
`∂/∂|Z|[(√d − |Z|)²] = −2(√d−|Z|)` yields `Z_mag = (2√d + ρ|V|)/(2+ρ)`.

**Bug 2 — Wrong dual sign convention.** Boyd convention: `U += (Ax − z)`, i.e. `frames_F − Z`, not the reverse.

**Bug 3 — Parseval mismatch.** When Z lives in real space but data is in Fourier space, ρ must be scaled by n²≈1024 for equal weighting. Moved Z and U to Fourier domain: ρ=1 then gives natural 2:1 data:consensus blend.

**Fundamental issue — dual diverges for any `dual_step > 0`.** The Fourier magnitude constraint
`|Z| = √d` is non-convex (it defines an annulus, not a convex set). ADMM convergence proofs require convexity for the constraint. In practice: U accumulates without bound across iters, dragging ψ away from data
consistency. Even ρ=50 or reduced dual_step=0.01 does not stabilise.

**`dual_step=0` degenerates to proximal AP**, which does converge (U stays 0, Z-update = Pf, ψ-update = Po).
This variant beats plain AP+diag but is slower than AP+diag+exact or RAAR+diag.

### Conclusion

ADMM for ptychography requires a convex reformulation (e.g., lifted variable or relaxed magnitude) or a
warm-started dual near a feasible point. From cold start with non-convex Fourier magnitude, the dual
always diverges. Left in codebase as `ADMM_position` with `dual_step=0.0` as default (proximal AP); true
ADMM is a future direction.

---

## Finding 23 — Coarse position correction via cross-correlation (2026-07-15)

### Idea (from collaborator)

After the Fourier projection `Pf`, absolute position is unconstrained: `|FFT(f)| = |FFT(shifted f)|`.
Therefore the integer-pixel shift of each frame relative to the current model can be recovered by
cross-correlating the projected frame with the model prediction `Sⱼ(ψ) · p`:

```python
CC = ifft2(fft2(frames) * conj(fft2(model)))
delta = argmax(|CC|)   # integer shift (with wrap-around correction)
```

This breaks the ~0.12 px Taylor capture range: integer shifts up to ±nx//2 can be corrected.

### Implementation

`position_solve_coarse` added to `position_retrieval.py`:
- Computes per-frame CC in batch (shape [nframes, nx, ny])
- Wraps indices ≥ nx//2 to negative (circular shift)
- Optional `max_shift` clip to prevent runaway on noisy frames
- Returns updated `translations_x/y` and new `mapid`

Wired into `RAAR_position` via `coarse_every` (cadence) and `coarse_max_shift` parameters.
The coarse step runs **before** the fine Taylor linearisation on the same iteration.

### Status (tested 2026-07-17)

Tested on `test_coarse_correction.py`, 1000 iters, A ∈ {0.4, 1.0, 2.0} px, coarse_every=50,
max_shift=4:

| A (px) | no coarse eps@end | coarse/50 eps@end |
|--------|-------------------|-------------------|
| 0.40   | 0.099             | 0.099 (no change) |
| 1.00   | 0.129             | 0.129 (no change) |
| 2.00   | 0.545             | 0.546 (no change) |

**Coarse model CC does NOT break the capture gap.**  Root cause: at cold start (iter 100), the
reconstructed image `ψ` is too blurry (blurred by 2 px position errors across all frames) for
the CC between `xm_k` and `model(ψ)` to show a meaningful peak at the right shift.  The CC peak
stays near zero → no correction applied.

---

## Finding 24 — Global sync via pairwise CC: captures free shifts, not position errors (2026-07-17)

### Motivation

After the Fourier-magnitude projection Pf, each RAAR frame can have an arbitrary integer circular
shift (`free shift`) without changing the diffraction data.  The collaborator's idea: recover these
shifts globally via pairwise CC + graph Laplacian, then correct scan positions — breaking the Taylor
capture gap (≈0.12 px).

### Theory: what does pairwise CC measure?

Let `z_k = roll(ew_k, s_k)` be the RAAR iterate for frame k (probe × object at nominal position t_k,
rolled by free shift s_k).  Using the numpy CC convention `CC[δ] = Σ_r z_i(r) z_j*(r−δ)`:

```
argmax CC(z_i, z_j) = δ* = (t_j − t_i) + (s_i − s_j)
```

After subtracting the nominal step dtnom = t_j − t_i:

```
r_{ij} = dtnom − δ*  =  s_j − s_i    (free shift difference, NOT position error difference)
```

The position error `ε_k` does NOT appear because both frames share the same RAAR object estimate O_raar,
which is calibrated to the nominal positions {t_k}.  From O_raar's perspective, every frame is already
"at the right place"; only the free shifts differ.

### Sign bug found in earlier implementation

The original code computed `r_{ij} = raw_δ − dtnom` (wrong sign) instead of `dtnom − raw_δ`.  This
added 2×dtnom as a spurious correction, causing catastrophic divergence (eps → 400+ px).  Fixed in
the current implementation.

### Weighted graph Laplacian (correct formulation)

With the corrected sign, the Laplacian gives δ_k = −s_k (free-shift corrections).  Applying t'_k = t_k + δ_k
removes the free shifts.  The Taylor solver can then refine sub-pixel positions without being confused by
large (integer) free shifts.  The system is:

```
min_δ  Σ_{(i,j)∈E} w_{ij} (δ_i − δ_j − r_{ij})²   with anchor δ_0 = 0
```

Weights w_{ij} = peak height of CC(z_i, z_j) (proxy for measurement confidence).

### Test result: sync still diverges

Test on `test_coarse_correction.py`, 1000 iters, A ∈ {0.4, 1.0, 2.0} px, sync_every=50,
max_shift=4, position_start=100:

| A (px) | no sync eps@end | sync/50 eps@end |
|--------|-----------------|-----------------|
| 0.40   | 0.099           | 72.2 (DIVERGE)  |
| 1.00   | 0.129           | 26.9 (DIVERGE)  |
| 2.00   | 0.545           | 13.8 (DIVERGE)  |

### Diagnosis

Even with the corrected sign, sync diverges because at iteration 100 (first call) the RAAR iterates
have not yet converged to clean rolled exit waves.  The CC peaks are at spurious locations, giving
wrong r_{ij} that scramble the scan grid.  After enough iterations (~500+) the iterates converge to
`z_k ≈ roll(ew_k_wrong, s_k)` where ew_k_wrong is the exit wave at the WRONG position.  At that
point sync could correct free shifts, but NOT position errors (because O_raar is a blurry average
calibrated to nominal positions).

### Conclusion

The pairwise CC + graph Laplacian approach correctly recovers free Pf shifts (`s_k`) but cannot
detect position errors (`ε_k`).  It does NOT break the Taylor capture gap.  For breaking the gap,
a method sensitive to position errors (not just free shifts) is needed.

Alternatives still under investigation:
- Model-based coarse CC with a sharper object estimate (needs >500 iters of RAAR to build object)
- Intensity-based CC (using measured diffraction amplitudes Y_k directly)
- Downsampled/coarser reconstruction with larger Taylor radius

---

## Open work

1. **hand-off-to-diag refinement after drift_fit** (`drift_fit=True` bootstraps to ~1 px; diag refines to ~0.01 px — test this pipeline end-to-end)
2. **Fig 7 full reproduction** (RAAR+diag+taylor, 10k iters; PNG saved for 4-way 2k comparison; confirm paper Fig 7 curve shape with 10k run)
3. **Prelocalize** — correct use case: integer-level errors + sharp image; test with simulated integer map errors
4. Model-constrained jitter refine after drift fit
