"""Regression: the drift basis must be able to index a frame by WHEN, not only WHERE.

`Solvers.fit_drift_global` fits xi = B @ c, and before this test B was built
only from the nominal scan coordinate s = (translations - mean). That indexes a
frame by WHERE it sits. A drift that is a function of TIME is representable
only if the trajectory makes time a low-order polynomial in position, and two
common cases break that:

  1. SCAN ORDER. A serpentine raster reverses the fast axis every row, so a
     linear-in-time creep becomes a staircase along the slow axis plus a
     triangular wave along the fast one. The triangular part is out of reach.
     The size of the gap is the within-row increment, ~total_drift/n_rows, so it
     is negligible on a 40-row raster and dominant on a few-row one -- the test
     pins both ends, because a test that only showed the big number would
     misrepresent when this matters.

  2. NONLINEARITY IN TIME. Exponential settling -- what a stage or a thermal
     enclosure does after being disturbed -- is not reachable from s at all,
     and a SINGLE time column does not fix it either. That is the case that
     motivates a time ORDER (time2, time3, ...) rather than one t.

Oscillatory drift is out of reach of every model here; that is asserted too, so
nobody expects a global fit of a few scalars to catch vibration.

Most of the file is exact linear algebra on the basis (microseconds): span(B)
is a hard ceiling on what fit_drift_global can recover with that model, so the
projection residual is the honest statement of reachability, with no solver
noise on top. One end-to-end test then drives the real solver.

Runs on CPU (NumPy) by default; flip config.GPU for the CuPy path.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

if config.GPU:
    import cupy as cp

    xp = cp

    def tonp(a):
        return cp.asnumpy(a)
else:
    xp = np

    def tonp(a):
        return np.asarray(a)

from Solvers import drift_basis, fit_drift_global


# ---------------------------------------------------------------------------
# scan generation -- the ORDER is the point, so it is written out explicitly
# rather than taken from the default translation generator
# ---------------------------------------------------------------------------
def scan_order(rows, cols, step_x=1.0, step_y=1.0, serpentine=True):
    """Scan positions in ACQUISITION ORDER: frame i was measured i-th.

    Fast axis is x. `serpentine=True` reverses it on every odd row, which is
    how a raster is actually run (the stage does not fly back). `False` gives
    the same GRID in plain order -- the control that isolates the ordering.
    """
    tx, ty = [], []
    for j in range(rows):
        xs = range(cols) if (j % 2 == 0 or not serpentine) else range(cols - 1, -1, -1)
        for i in xs:
            tx.append(i * step_x)
            ty.append(j * step_y)
    return np.asarray(tx, float), np.asarray(ty, float)


def drift_profile(kind, nframes, peak_to_peak):
    """A per-frame drift of the given peak-to-peak amplitude, in acquisition order."""
    u = np.arange(nframes, dtype=float) / (nframes - 1)     # 0..1 through the scan
    f = {
        "linear": u,                             # thermal/mechanical creep
        "settling": 1.0 - np.exp(-u / 0.15),     # stage or enclosure settling
        "oscillatory": np.sin(2 * np.pi * 3 * u),  # vibration, 3 cycles
    }[kind]
    f = f - f.mean()
    return peak_to_peak * f / (f.max() - f.min())


def unreachable(model, tx, ty, xi):
    """rms of the part of `xi` that `model` cannot represent, in pixels.

    Least squares onto span(B) PLUS a constant column: the constant is the
    unobservable global-translation gauge, which fit_drift_global omits by
    design and shift_rmse removes from its metric. So this is exactly the best
    case for fit_drift_global with this model -- a floor no amount of
    grid-search effort can go below.
    """
    B = np.column_stack([np.ones(len(tx)), tonp(drift_basis(model, xp.asarray(tx),
                                                            xp.asarray(ty)))])
    coef, *_ = np.linalg.lstsq(B, xi, rcond=None)
    return float(np.std(xi - B @ coef))


P2P = 4.0  # peak-to-peak drift, px -- a few pixels, as drifts go


# ---------------------------------------------------------------------------
# 1. scan ORDER: what a serpentine raster costs a position-only basis
# ---------------------------------------------------------------------------
def test_serpentine_order_hides_a_linear_creep_from_the_position_basis():
    """Few rows: the position-only basis misses a linear creep; +time gets it.

    The control is the same GRID in plain raster order, where time IS affine in
    position and the old basis is already exact. If the control ever stops
    being exact, the mechanism is not the scan ordering and this test is
    measuring something else.
    """
    rows, cols = 4, 100
    xi = drift_profile("linear", rows * cols, P2P)

    tx, ty = scan_order(rows, cols, serpentine=True)
    assert unreachable("linear", tx, ty, xi) > 0.2
    assert unreachable("poly2", tx, ty, xi) > 0.2
    assert unreachable("linear+time", tx, ty, xi) < 1e-9
    assert unreachable("time", tx, ty, xi) < 1e-9

    cx, cy = scan_order(rows, cols, serpentine=False)
    assert unreachable("linear", cx, cy, xi) < 1e-9, (
        "plain raster order must already be exact -- the gap above is the "
        "fast-axis reversal, not the scan geometry"
    )


def test_the_gap_is_the_within_row_increment_so_many_rows_hides_it():
    """~total/n_rows: negligible on a 40-row raster, dominant on a few-row one.

    Pinned so the feature is not oversold: on an ordinary many-row raster the
    EXISTING basis already absorbs a linear creep to far below anything that
    matters, and the time column buys essentially nothing there.
    """
    got = {}
    for rows, cols in ((40, 40), (8, 200), (4, 100), (2, 800)):
        tx, ty = scan_order(rows, cols, serpentine=True)
        got[rows] = unreachable("linear", tx, ty,
                                drift_profile("linear", rows * cols, P2P))

    assert got[40] < 0.05          # a 40-row raster: already fine without time
    assert got[4] > 5 * got[40]    # few rows: the gap is real
    assert got[40] < got[8] < got[4] < got[2]

    # 2 rows is the degenerate case: the fold IS the bilinear term, so poly2 --
    # which has sx*sy -- represents it exactly. Few rows alone is not the story.
    tx, ty = scan_order(2, 800, serpentine=True)
    assert unreachable("poly2", tx, ty, drift_profile("linear", 1600, P2P)) < 1e-9


# ---------------------------------------------------------------------------
# 2. nonlinearity in TIME: the case a single t column does NOT fix
# ---------------------------------------------------------------------------
def test_settling_drift_needs_a_time_ORDER_not_merely_a_time_column():
    """Exponential settling defeats s, and defeats one t column just as badly."""
    rows, cols = 40, 40
    tx, ty = scan_order(rows, cols, serpentine=True)
    xi = drift_profile("settling", rows * cols, P2P)

    assert unreachable("linear", tx, ty, xi) > 0.4
    assert unreachable("poly2", tx, ty, xi) > 0.2
    # the point: ONE time column is no better than no time column at all here
    assert unreachable("linear+time", tx, ty, xi) > 0.4
    assert unreachable("time2", tx, ty, xi) > 0.2

    assert unreachable("time3", tx, ty, xi) > 0.05   # order 3 still short
    assert unreachable("time4", tx, ty, xi) < 0.05   # order 4 is where it lands
    assert unreachable("time6", tx, ty, xi) < 0.005

    # unlike the ordering gap, this one does not care about the geometry:
    # it is a property of the drift, not of the scan
    fx, fy = scan_order(6, 8, serpentine=True)
    assert unreachable("time4", fx, fy, drift_profile("settling", 48, P2P)) < 0.05


def test_oscillatory_drift_is_out_of_reach_of_every_model():
    """A handful of global scalars cannot represent vibration -- documented, not fixed."""
    rows, cols = 40, 40
    tx, ty = scan_order(rows, cols, serpentine=True)
    xi = drift_profile("oscillatory", rows * cols, P2P)
    for model in ("linear", "poly2", "time", "time3", "time6", "poly2+time6"):
        assert unreachable(model, tx, ty, xi) > 1.0, model


# ---------------------------------------------------------------------------
# 3. mechanism guards
# ---------------------------------------------------------------------------
def test_relative_residual_is_the_same_on_both_axes():
    """The fold lives in the REGRESSORS, not in which component of xi is fitted.

    B fits xi_x and xi_y independently with the same columns, so the residual
    FRACTION is a property of the basis and the acquisition order alone. Only
    the absolute error scales with the per-axis amplitude. (Stated because the
    opposite -- the slow axis faring better -- is the intuitive guess, and it
    is wrong.)
    """
    tx, ty = scan_order(4, 100, serpentine=True)
    f = drift_profile("linear", 400, P2P)
    for model in ("linear", "poly2"):
        rel = [unreachable(model, tx, ty, a * f) / np.std(a * f) for a in (1.0, 0.4)]
        assert abs(rel[0] - rel[1]) < 1e-9, model


def test_existing_models_are_bit_exact():
    """`linear` and `poly2` must be untouched: this change is purely additive."""
    rng = np.random.default_rng(0)
    tx = rng.normal(size=37) * 10 + 3
    ty = rng.normal(size=37) * 4 - 7

    def legacy(model, translations_x, translations_y):   # verbatim, pre-change
        sx = translations_x - np.mean(translations_x)
        sy = translations_y - np.mean(translations_y)
        scale = max(float(np.max(np.abs(sx))), float(np.max(np.abs(sy))), 1.0)
        sx = sx / scale
        sy = sy / scale
        cols = [sx, sy, sx * sx, sx * sy, sy * sy] if model == "poly2" else [sx, sy]
        return np.stack(cols, axis=1)

    for model in ("linear", "poly2"):
        new = tonp(drift_basis(model, xp.asarray(tx), xp.asarray(ty)))
        assert np.array_equal(new, legacy(model, tx, ty)), model


def test_time_columns_are_centered_normalized_and_ordered():
    tx, ty = scan_order(4, 25, serpentine=True)
    n = tx.size

    # "time" is exactly the centered, unit-normalized acquisition index
    t = np.arange(n, dtype=float)
    t = (t - t.mean()) / np.abs(t - t.mean()).max()
    assert np.allclose(tonp(drift_basis("time", xp.asarray(tx), xp.asarray(ty)))[:, 0], t)

    assert drift_basis("time", xp.asarray(tx), xp.asarray(ty)).shape[1] == 1
    assert drift_basis("time1", xp.asarray(tx), xp.asarray(ty)).shape[1] == 1
    assert drift_basis("time4", xp.asarray(tx), xp.asarray(ty)).shape[1] == 4
    assert drift_basis("linear+time2", xp.asarray(tx), xp.asarray(ty)).shape[1] == 4
    assert drift_basis("poly2+time3", xp.asarray(tx), xp.asarray(ty)).shape[1] == 8

    B = tonp(drift_basis("time4", xp.asarray(tx), xp.asarray(ty)))
    assert np.abs(B.mean(0)).max() < 1e-12      # no column carries the gauge
    assert np.allclose(np.abs(B).max(0), 1.0)   # drift_max means the same for each

    # order n spans the same functions as t..t^n, but is better conditioned --
    # which is what coordinate-descent grid search actually needs
    raw = np.stack([t ** m - (t ** m).mean() for m in range(1, 5)], axis=1)
    raw = raw / np.abs(raw).max(0)
    assert np.linalg.cond(B) < 0.5 * np.linalg.cond(raw)
    resid = raw - B @ np.linalg.lstsq(B, raw, rcond=None)[0]
    assert np.abs(resid).max() < 1e-10


def test_time_columns_are_orthogonalized_and_redundant_ones_dropped():
    """A time column the s columns already span is dropped, not renormalized.

    On a PLAIN raster t is exactly a combination of sx and sy, so "linear+time"
    must collapse to the two s columns; renormalizing a numerically-zero
    residual would hand the grid search a column of roundoff. Under serpentine
    order the same column is genuinely new and must survive.

    The projection is least squares against the whole accepted block, not
    sequential Gram-Schmidt: the poly2 columns are not mutually orthogonal, so
    projecting off one at a time would leave the time column correlated with
    them and would not detect redundancy.
    """
    for serpentine, expect in ((False, {"linear+time": 2, "poly2+time": 5,
                                        "linear+time3": 4, "poly2+time3": 7}),
                               (True, {"linear+time": 3, "poly2+time": 6,
                                       "linear+time3": 5, "poly2+time3": 8})):
        tx, ty = scan_order(4, 25, serpentine=serpentine)
        for model, k in expect.items():
            got = drift_basis(model, xp.asarray(tx), xp.asarray(ty)).shape[1]
            assert got == k, f"{model} {'serp' if serpentine else 'raster'}: {got} != {k}"

    tx, ty = scan_order(4, 25, serpentine=True)
    for model, nspace in (("linear+time3", 2), ("poly2+time3", 5)):
        B = tonp(drift_basis(model, xp.asarray(tx), xp.asarray(ty)))
        assert np.abs(B[:, :nspace].T @ B[:, nspace:]).max() < 1e-10, model


@pytest.mark.parametrize("model", ["banana", "", None, "time0", "linear+poly2", "Linear"])
def test_unknown_model_is_rejected(model):
    tx, ty = scan_order(2, 4)
    with pytest.raises(ValueError):
        drift_basis(model, xp.asarray(tx), xp.asarray(ty))


# ---------------------------------------------------------------------------
# 4. end to end: the real solver, on intensity data
# ---------------------------------------------------------------------------
# Few rows is what makes the ordering gap large, and few rows is also what
# makes the slow axis weakly determined, so the scan is stretched along the
# fast axis to keep the frame count (and hence the redundancy the grid search
# needs) up: 4 x 28 at step 2/14 gives 112 frames on a 56 x 56 image, the same
# redundancy class as position_capture_test.build_scene. Square by
# construction: Operators.map_frames flattens as x + y*Nx, which only agrees
# with an (Nx, Ny) image when Nx == Ny.
E2E = dict(rows=4, cols=28, step_x=2, step_y=14, nx=32)


def _scene(serpentine=True, **kw):
    from Operators import make_probe, map_frames, Splitc
    from position_retrieval import shift_probe_fourier, apodize_probe
    from position_simulate import transmission_object

    cfg = dict(E2E, **kw)
    nx = cfg["nx"]
    nimg = cfg["cols"] * cfg["step_x"]
    assert nimg == cfg["rows"] * cfg["step_y"], "image must be square"

    tx, ty = scan_order(cfg["rows"], cfg["cols"], cfg["step_x"], cfg["step_y"],
                        serpentine=serpentine)
    probe = make_probe(nx, nx, r1=0.075, r2=0.255)
    if isinstance(probe, tuple):
        probe = probe[0]
    probe = apodize_probe(np.asarray(probe / np.abs(probe).max(), np.complex64))
    truth = transmission_object(nimg, nimg, contrast=4.1)
    mapid = map_frames(tx, ty, nx, nx, nimg, nimg)

    def intensities(xi_x, xi_y):
        z = Splitc(truth, mapid) * shift_probe_fourier(probe, xi_x, xi_y)
        return (np.abs(np.fft.fft2(z)) ** 2).astype(np.float64)

    return tx, ty, probe, nx, nimg, intensities


def _fit_err(model, drift_x, drift_y):
    """Per-axis, gauge-removed error of fit_drift_global, in pixels."""
    tx, ty, probe, nx, nimg, intensities = _scene(serpentine=True)
    data = intensities(drift_x, drift_y)
    hat_x, hat_y = fit_drift_global(data, probe, tx, ty, nx, nx, nimg, nimg,
                                    model=model, drift_max=5.0)
    return (float(np.std(drift_x - tonp(hat_x))),
            float(np.std(drift_y - tonp(hat_y))))


def test_fit_drift_global_recovers_a_time_creep_a_position_basis_cannot():
    """The regression proper: same data, three models, on a serpentine raster.

    Reported per axis rather than pooled. Both axes carry the same creep at
    different amplitudes, so both must fail for the position-only models and
    both must be recovered by the time model -- the residual FRACTION is a
    property of the basis, not of the axis.

    The failures here (~2.2 px) are far worse than the 0.29 px that
    `unreachable` says is strictly out of reach: once the drift leaves span(B),
    the grid search is minimizing a misfit whose optimum is not near the truth,
    so it wanders well past the projection residual. Both numbers are real --
    0.29 px is what the model CANNOT reach, 2.2 px is what it actually does.
    """
    n = E2E["rows"] * E2E["cols"]
    creep = drift_profile("linear", n, P2P)
    drift_x, drift_y = 1.0 * creep, 0.4 * creep

    for model in ("linear", "poly2"):
        ex, ey = _fit_err(model, drift_x, drift_y)
        assert ex > 1.0 and ey > 1.0, f"{model}: {ex:.3f} {ey:.3f} -- expected failure"

    ex, ey = _fit_err("time", drift_x, drift_y)
    assert ex < 0.1 and ey < 0.1, f"time: {ex:.3f} {ey:.3f} -- expected recovery"


if __name__ == "__main__":
    print("reachability: rms of the drift that each model CANNOT represent (px)\n"
          f"drift {P2P} px peak-to-peak, serpentine raster unless noted\n")
    models = ["linear", "poly2", "time", "time2", "time3", "time4", "time6"]
    print(f"{'geometry':>12} {'drift':>12} " + " ".join(f"{m:>7}" for m in models))
    print("-" * (26 + 8 * len(models)))
    for rows, cols in ((40, 40), (8, 200), (4, 100), (2, 800)):
        tx, ty = scan_order(rows, cols, serpentine=True)
        for kind in ("linear", "settling", "oscillatory"):
            xi = drift_profile(kind, rows * cols, P2P)
            print(f"{f'{rows} x {cols}':>12} {kind:>12} "
                  + " ".join(f"{unreachable(m, tx, ty, xi):>7.3f}" for m in models))
    tx, ty = scan_order(40, 40, serpentine=False)
    xi = drift_profile("linear", 1600, P2P)
    print(f"{'40 x 40':>12} {'lin/raster':>12} "
          + " ".join(f"{unreachable(m, tx, ty, xi):>7.3f}" for m in models))

    print("\nend to end (fit_drift_global on intensity data), serpentine 4 x 28")
    n = E2E["rows"] * E2E["cols"]
    creep = drift_profile("linear", n, P2P)
    print(f"{'model':>8} {'err_x':>8} {'err_y':>8}   (truth: "
          f"{P2P:.1f} px p2p in x, {0.4 * P2P:.1f} in y)")
    for m in ("linear", "poly2", "time", "time2"):
        ex, ey = _fit_err(m, 1.0 * creep, 0.4 * creep)
        print(f"{m:>8} {ex:>8.3f} {ey:>8.3f}")
