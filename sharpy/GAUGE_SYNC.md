# Gauge alignment, resolution assessment, and group synchronization

Three pure-numpy utilities for comparing ptychographic/CDI reconstructions and for
synchronizing per-frame gauges (global phase, shift, tilt) across the overlap graph.
No GPU or external data; each module has a `__main__` self-test and a pytest suite.

| module | purpose |
|---|---|
| `gauge_align.py` | remove the gauge between two complex fields so they can be compared (FRC/NMSE/diff) |
| `patched_frc.py` | resolution **with an error bar** (patched FRC) + long-range fidelity (`lowband_frc`) |
| `group_sync.py`  | shift + tilt + phase **synchronization** over the overlap graph |

They compose: measure the pairwise relative gauge between overlapping frames with
`gauge_align`, synchronize it into globally-consistent per-frame gauges with
`group_sync`, and score the result with `patched_frc`.

## `gauge_align` — the universal gauge aligner

Two reconstructions (or a recon vs. ground truth, or two half-datasets) agree only up
to the phase-retrieval gauge freedoms. `align_gauge(A, B)` removes them all:

- **sub-pixel translation** — the single-step upsampled-DFT registration of
  Guizar-Sicairos, Thurman & Fienup, *Opt. Lett.* **33**, 156 (2008) [MGS08].
- **a polynomial phase** `p(r)` of degree `poly_order` — Marchesini, Chapman, Barty
  *et al.*, "Phase Aberrations in Diffraction Microscopy," arXiv:physics/0510033
  (2005): order 1 = linear ramp/tilt, order 2 adds defocus (2,0)+(0,2) & astigmatism
  (1,1), order 3 adds coma. Fit via wrap-free amplitude-weighted phase **gradients**
  (`conj(r_i)·r_{i+1}`), so it never wraps and drops the global-phase constant.
- **global phase + scale** — MGS08's global complex factor.

Insight: MGS08 registration applied in **both** Fourier-conjugate domains — a
real-space *shift* is a phase-slope in Fourier; a real-space *ramp* is a shift of the
Fourier **magnitude** — so the same routine, run on `|FFT|`, removes the object
phase-ramp gauge MGS08 omits. Shift and ramp couple (each corrupts the other's naive
estimate), so the ramp is first coarsened from the magnitude-spectrum shift.

`register_translation_masked(A, B, wA, wB)` is the **masked/weighted** variant
(Padfield, *IEEE TIP* 2012) for frames valid only where a weight is significant
(`w = probe amplitude × detector mask`): the weighted cross-correlation is normalized
by the weight **overlap**, so a partial footprint or a detector gap doesn't bias the
peak. For two frames sharing a common object the object *cancels* and it returns the
**probe's** relative shift — the position-error signal — even on a low-absorption
object (the structured probe is the reference).

**Minimal-gauge rule for FRC:** use `poly_order=1` only. In ptychography the low-q
phase is genuinely under-determined (transfer function suppressed at low q — Ophus,
arXiv:2309.05250), so pre-removing higher-order (defocus/astigmatism) phase would
erase the long-range signal you want to measure. `poly_order>=2` is for
*characterizing* aberrations, not for pre-FRC alignment.

## `patched_frc` — resolution with an error bar

`patched_frc(A, B)` tiles the (gauge-aligned) recon and reference into overlapping
subregions, computes an erf-apodized FRC per patch, and returns the **median ± std**
of the per-patch resolution. The error bar is a spatial-uniformity diagnostic
(instabilities degrade resolution *erratically*, not uniformly), and small patches
also cure the global-FRC Nyquist saturation. `lowband_frc(A, B)` is the mean FRC over
the lowest decile of spatial frequency = the long-range fidelity, which is where a
sync recovers quality — report it alongside the high-q resolution cutoff.

## `group_sync` — shift + tilt + phase synchronization

The per-frame gauge group is the Weyl-Heisenberg group (shift & tilt are conjugate
phase-space translations, their commutator is the phase). It **factors** because shift
and tilt are each abelian R²:

- `sync_translation` — weighted graph-Laplacian **least squares** (`L x = b`): the
  shift channel (translation synchronization), and the tilt channel in the Fourier
  coordinate.
- `sync_phase` — **U(1) angular synchronization** = top eigenvector of the Hermitian
  connection matrix (Singer, *ACHA* 2011).
- `sync_heisenberg` — both, iterating the ½(s×t) Heisenberg cocycle correction.

`build_overlap_graph(pos, radius)` gives the sparse edge set (overlapping pairs only,
O(N·k)). The pairwise relative gauges come from `gauge_align` on the overlaps.

Because the sync measures the pairwise **neighbor** difference and integrates it over
the graph, its capture range is set by the *relative* error, not the absolute — so it
is far larger for **correlated / drift** errors (thermal, stage creep, FEL pointing
wander) than for independent jitter, and larger than local (per-frame, vs-model)
refinement.

## Composition example

```python
import gauge_align as ga, group_sync as gs
edges = gs.build_overlap_graph(pos, radius)
dS = [ga.register_translation_masked(Ohat[k][ov], Ohat[l][ov], wk, wl) for k, l in edges]
corr = gs.sync_translation(edges, dS, len(pos))   # per-frame position correction
```
