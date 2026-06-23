# Position retrieval — state of play & what to try next

Hand-off notes for Yuan picking up scan-position retrieval. Summary of the
investigation so far (sharpy `refine_illumination` branch). These are working
notes — grep the current tree for exact function names before relying on a
file:line. — June 2026

Position retrieval = **Section IV of arXiv:1209.4924** ("Augmented projections for
ptychographic imaging", Marchesini et al., Inverse Problems 2013): the pairwise
Gramian objective is extended to unknown per-frame shifts `Δx_(i)` by Taylor-
expanding the probe about each nominal position and solving a Gauss–Newton step.
See report Section V (`gramian-overleaf/main.tex`) for the write-up and the
"Open directions" box. MATLAB reference implementation:
`~/Downloads/matlab/Matlab_Archive` (`fit_shift.m` = the Eq.(27) core,
`doraar0_shift.m` driver, `horse_shoe_x{,1}.m` = the Fig 7 reproduction).

## What's implemented (sharpy `position_retrieval.py`)

- **Three solvers**, all on a probe-shift parameterization (band-limited probe
  shifts more accurately than the high-freq object — *don't* shift the object):
  - `position_solve_diag` — per-frame Section IV Gauss–Newton. `method="taylor"`
    (default) or `method="exact"` (band-limited Fourier-shifted probe + derivs).
  - `position_solve_coupled` — full off-diagonal Eq.(27) (neighbour couplings).
  - `position_solve_gradient` — Guizar-Sicairos & Fienup 2008 steepest-descent
    baseline (data-misfit gradient; use the **γ=1/2 amplitude** metric, not γ=1).
- **Probe derivatives** `probe_derivatives(method="fourier")` (spectral, default,
  3–90× more accurate sub-pixel than `"fd"`), `shift_probe_fourier` (exact shift).
- **Apodization** `apodize_probe`/`apodization_mask` (probe ~zero on outer half →
  un-aliased intensity); `position_simulate(apodize=True)` default.
- **Joint loop** `Solvers.Alternating_projections_position(replan=…, backend=…)`:
  position update wired into AP, with optional integer re-planning of the map.
- **CPU kernels** mirror the GPU `zQQz`/`zQQz2.cu`: Numba (`_braket*`) and OpenMP
  C (`zqqz_omp.c`/`zqqz_cpu.py`); selector `set_kernel_backend("auto"|"python"|
  "numba"|"omp")` / `SHARPY_KERNEL` env. All backends agree ~1e-15.
- Sim/utilities: `position_simulate.py` (`simulate_to_h5`/`load_h5`),
  `position_compare.py`, and the `position_capture_*`/`position_drift_test.py`
  diagnostics. `position_fig7.py` = the Fig 7 reproduction scaffold.

## Key findings (the important ones)

1. **Solver choice.** *Diagonal* is the practical winner — fast (~15 iters, <1 s
   CPU) and robust to both smooth and zone-plate probes. *Coupled* is most
   accurate on a smooth probe (≈1e-16) **but diverges on the zone-plate** — a
   conditioning issue; it **needs Levenberg–Marquardt damping** (add λI to H) to
   be usable with realistic LCLS probes. *Gradient* is robust but ~13× more iters
   (it's plain steepest descent here; CG+line-search would narrow the gap).

2. **Coupled wins on *correlated* drift.** For slow drift (random-walk in scan
   order) coupled converges **2–7× faster** than on i.i.d. errors of equal size;
   the diagonal solver gets no such benefit. So coupled (with LM damping) is the
   method for drift-dominated data; diagonal suffices for uncorrelated jitter.

3. **Capture range = ABSOLUTE per-frame shift, not relative/pairwise.** Recovery
   is gated by `max|Δx_(i)|` vs `~1/k_max` (probe band-limit / lens aperture), not
   by neighbour differences. Decisive control: a smooth ramp with *small* relative
   step but *large* absolute fails, while i.i.d. with *larger* relative but smaller
   absolute succeeds. **Observability ≠ capturability**: a 16-px ramp is loudly
   rejected by the data (`eps_F` huge) but uncapturable from `Δx=0`.

4. **Exact re-linearization extends the basin** (`method="exact"`): replace the
   Taylor model with the exact band-limited shifted probe (+ shifted derivatives)
   at each estimate → machine precision in-range and capture ~1.5 → ~2.5 px. Both
   Taylor and exact still fail at ~3–4 px = fundamental per-frame non-convexity.

5. **Re-planning extends REPRESENTATION, not CAPTURE.** Migrating the integer part
   of `Δx` into the map (`Alternating_projections_position(replan=True)`; needs an
   apodized probe; probe `+1` in x ≡ map `translations_y −1`, x↔y transposed) lets
   you *hold* a large shift as integer-map + sub-pixel residual, but does **not**
   help you *reach* it from zero. (A standalone fixed-frame remap is ill-founded —
   the probe↔map equivalence is a real-space translation that only closes through
   the intensity/data projection, i.e. inside the AP loop.)

6. **"Pure pairwise" gives the SAME capture as image-referenced.** A position solver
   built purely on pairwise frame differences (no image) is the same `(I−P_W)z`
   residual with only a `W*W` reweighting (variance identity), so it forms the same
   consensus implicitly → capture is still absolute/lens-limited, *not* relative.
   Don't build a separate pairwise position solver for drift capture.

7. **Joint probe+position gotcha:** `refine_illumination`'s naive `Σ conj(O_n) z_n`
   **blurs the probe** when frames carry per-frame shifts (it averages mis-
   registered probes). Un-shift each frame by `−Δx_n` to a common probe frame
   *before* the probe-update sum (~6× sharper probe at ~0.35 px spread).

## What to try next (suggested for Yuan)

1. **LM damping on the coupled solver** — the single missing piece to make the
   correlated-drift advantage usable with realistic zone-plate probes (#1, #2).
2. **Parametric drift fit + warm-start** (capture extender, validated as a sketch):
   fit a low-dim model (1 scalar/dim = scan magnification, or 2×2 affine) by a
   GLOBAL grid/line search on the total data misfit `eps_F` (a sharp bowl, no
   per-frame basin limit), then warm-start the per-frame exact refiner. Beats the
   lens basin for large drift; not yet productionized.
3. **Coarse cross-correlation pre-alignment** — register each frame to a coarse
   reconstruction for the integer/large part, then refine sub-pixel (the only cure
   beyond ~3 px).
4. **Faithful Fig 7 reproduction** — position + unknown image *jointly* from
   intensity-only data via RAAR/AP (truth only for the error metric ε_ξ). Scaffold:
   `position_fig7.py`; exact MATLAB setup (nx=32, step 3.5, 16×16=256 frames,
   zone-plate r1=0.075/r2=0.255, unknown ±2 px shifts) is in the notes.
5. **GPU kernel work** (backlog): generalize `zQQz2.cu` (left/right derivative
   probes), compute O11/O12/O22 in one pass, in-kernel Taylor probe shift,
   image-sized normalization. De-risk each on the Numba CPU twin first.
6. **Gradient → CG** for a fair head-to-head, and compare all three **under noise**
   (Fienup's γ/δ robustify the metric).

## Pointers

- Report: `gramian-overleaf/main.tex` Sec. V (+ the "Open directions" box).
- Code: `position_retrieval.py`, `Solvers.Alternating_projections_position`,
  `position_simulate.py`, `position_capture_*`/`position_drift_test.py`,
  `position_fig7.py`; kernels `src/zQQz2.cu`, `zqqz_omp.c`, `zqqz_cpu.py`.
- MATLAB reference: `~/Downloads/matlab/Matlab_Archive` (`fit_shift.m`,
  `doraar0_shift.m`, `horse_shoe_x{,1}.m`, `Common/zQQz.c`).
- Related: the sync-eigensolver study (`EIGSH_vs_POWER_sync_notes.md`) — the same
  overlap-graph Gramian machinery underlies both.
