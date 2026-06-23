# Sync eigensolver: why cupy `eigsh` sign-flips and power iteration doesn't

Notes for Yuan on the phase-synchronization eigensolver behavior (the localized
"blob"/sign-flip you saw in the poster `eigsh` sync vs. the clean power-iteration
result). Summary of an investigation on Perlmutter (A100), with reproducers you
can run. — June 2026

## TL;DR

1. **The static-matrix flip is real, and we now know the cause and can reproduce it
   on demand.** `cupy` `eigsh` returns a *sign-flipped* top eigenvector (π jumps in
   the per-frame phase `ω = v/|v|`, reshaped to the scan grid = your localized blob)
   whenever **two conditions hold together**:
   - **(a) a near-degenerate top cluster** of the overlap-graph Gramian — eig₂/eig₁
     within ~1 % of 1, a Fiedler/graph-Laplacian effect that *tightens ~1/N* with the
     number of frames and with *less overlap*; and
   - **(b) a non-flat consensus** — i.e. the top eigenvector is **not** the flat
     all-ones vector, so `v0 = ones` is a *poor* start. This happens when the
     **object is contrasty** (strong absorption / dark regions) → per-frame energy
     `|z_i|` is non-uniform → the consensus concentrates on the bright frames.

   Under-converged Krylov (`eigsh`, few iterations) then returns a
   *consensus + Fiedler* **mixture** → the π flip. **Power iteration is immune**: from
   `ones` it *amplifies* toward the smooth consensus and never diagonalizes, so it
   stays flip-free at every step (even while still far from converged in L₂).
   Slogan: **amplify, don't diagonalize.**

2. **This explains why a fresh gold-balls sim never reproduced it.** The standard
   object `exp(0.69·(-1+0.5j)·img/63)` is *weak contrast* (amplitude 0.5–1) →
   nearly uniform frame energy → flat consensus → `ones ≈ mode 1` → `eigsh` stays
   clean even at the same tiny gap. You need to **crank the contrast** to make the
   consensus non-flat.

3. **But it does NOT cause an in-loop reconstruction catastrophe with the current
   code** — not even blind, not even with high contrast and low overlap. The AP loop
   self-heals a momentary flip. So the poster's NMSE 0.19 → 0.56 blow-up is **not**
   caused by object contrast/overlap with today's code; it appears tied to the
   **old `cupy`-eigsh + gauge-handling path** that's since been replaced. The
   committed default (warm-started power iteration) is robust.

## The real failing matrix: `saveH.npz`

`sharpy/saveH.npz` is the actual 4096×4096 (= your 64×64 scan) complex64 Hermitian
overlap Gramian `H = W Wᴴ` from the poster run, saved in SciPy CSR format. It is the
ground-truth object for this study. Its signature:

| quantity | value |
|---|---|
| eig₂/eig₁ | 0.991 (near-degenerate cluster; modes 2,3,… are nodal Fiedler modes) |
| `\|⟨v₁, ones⟩\|²` | **0.18** (consensus is NOT flat → `ones` is a poor `eigsh` start) |
| Rayleigh(ones)/eig₁ | 0.84 |
| per-frame degree (diag) | 0.10–0.24, **center suppressed** (= dark object region) |
| cupy eigsh flip (maxiter 5–100) | 0.10–0.24 (flips, persists) |
| power-iteration flip | 0.000 (clean) |

## Reproducers (in `sharpy/`)

- **`synth_eigsh_fail.py`** — CPU/laptop, no GPU. Builds a *synthetic* overlap Gramian
  `H = diag(s) K diag(s)` (RBF overlap kernel on a G×G scan grid; `s` = per-frame
  energy) and shows **uniform `s` → no flip** vs. **structured `s` (dark blobs) →
  flip**, using a short-Lanczos emulation of under-converged `eigsh`. The structured
  case matches `saveH.npz` almost exactly (overlap 0.16, Rayleigh/eig₁ 0.80, flip 0.2).
  ```
  G=64 SIGMA=1.6 R=4 CONTRAST=0.85 python synth_eigsh_fail.py
  ```
- **`synth_eigsh_fail_gpu.py`** — Perlmutter, confirms on the **real `cupy` eigsh**
  (k=1, ncv=3, maxiter sweep) vs GPU power iteration, on the synthetic H *and* on
  `saveH.npz`. Verified result: uniform 0.000, structured 0.10–0.16, real saveH
  0.10–0.24; power 0.000 everywhere.
  ```
  source ~/sharpy-venv/bin/activate
  srun -A lcls_g -C gpu -q interactive -N1 -n1 --gpus 1 -t 00:10:00 python -u synth_eigsh_fail_gpu.py
  ```
- **`blind_sync_cmp.py`** — in-loop blind reconstruction (refine_illumination=True),
  comparing no-sync / power / PPM / `eigsh`. New `CONTRAST` env. Result: across
  CONTRAST 0.69→3.5 and overlap 69→25 %, `eigsh ≈ power`, both beat no-sync; no
  eigsh-specific divergence (low overlap breaks *all* methods equally).
  ```
  CONTRAST=3.5 SHARPY_STEP=5 python -u blind_sync_cmp.py
  ```

(Other untracked diagnostics in `sharpy/` from the study: `realH_cupy.py`,
`realH_orth.py`, `sign_flip_test.py`, `fiedler_test.py`, `overlap_sweep.py`,
`eigsh_flip_map.py`, `phase_sync_test.py`, `ppm_sync_cmp.py`.)

## The open question for you

We could **not** reproduce the actual *in-loop* 0.19/0.56 catastrophe with current
code. The remaining unknown is **what the poster's custom eigensolver returned**:

- Did the poster's short/custom Arnoldi (the patched `cupy` eigsh that took `v0`, or
  the custom ~2-iteration Arnoldi) diagonalize the projected `T` and return the
  **top Ritz vector**, or did it return the **last Krylov/Arnoldi vector** (≈ the
  Fiedler residual after the consensus is projected out → a pure sign flip)?
- And what was the **gauge handling** (the small-`|ω|` masking — the commented
  `abs(omega) < thresh -> 1` lines in `Eigensolver`)?

With the exact call + gauge code we can close the in-loop reproduction; the static
flip is fully understood and reproducible above.

## Practical recommendation

Use the committed **warm-started power-iteration `Eigensolver`** (commit `1e7ed37`):
clean gauge, cheap, no flip. `eigsh` is the better *eigenvalue/L₂* solver but its
*phase* is fragile under-convergence on the near-degenerate cluster — and only the
phase enters synchronization. The structural cure for the low-overlap/large-N regime
(where the Fiedler cluster tightens and even sync stops helping) is hierarchical/
tiled (multigrid) synchronization, not a different eigensolver.
