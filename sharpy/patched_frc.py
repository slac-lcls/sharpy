"""Resolution WITH AN ERROR BAR via patched (subregion) FRC (SM's idea = Himanshu's 'Patched FRC ±σ').
Tile the (gauge-aligned) recon and ground truth into overlapping subregions, FRC each subregion vs
its GT patch, read a resolution at a threshold, and take the MEDIAN +/- STD across patches as the
resolution estimate and its uncertainty. Small patches also cure the global-FRC crossing saturation
(fewer ring samples -> higher, meaningful 2-sigma threshold).

Standalone (erf apodization + 2-sigma FRC inline). Run on a saved fel_recovery.npz (clean/uncorrected/
recovered A,B arrays) or import patched_frc(A,B).

  python -u patched_frc.py [fel_recovery.npz]   env: PATCH(96) STRIDE(48) THR(2sig|0.5)
"""
import numpy as np


def _erf(x):
    t = 1.0 / (1.0 + 0.3275911 * np.abs(x))
    return np.sign(x) * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x))


def erf_window(n, taper):
    x = np.arange(n); s = max(taper / 3.0, 1.0)
    return 0.25 * (1 + _erf((x - taper) / s)) * (1 + _erf((n - 1 - x - taper) / s))


def _frc1(a, b, taper):
    win = np.outer(erf_window(a.shape[0], taper), erf_window(a.shape[1], taper))
    FA = np.fft.fftshift(np.fft.fft2(a * win)); FB = np.fft.fftshift(np.fft.fft2(b * win))
    H, W = a.shape; cy, cx = H // 2, W // 2
    rr = np.hypot(*np.mgrid[-cy:H - cy, -cx:W - cx]).astype(int)
    nb = min(cy, cx)
    fa = FA.ravel(); fb = FB.ravel(); rrr = rr.ravel()
    num = np.zeros(nb, complex); da = np.zeros(nb); db = np.zeros(nb); nq = np.zeros(nb)
    for r in range(nb):
        m = rrr == r; nq[r] = m.sum()
        num[r] = (fa[m] * np.conj(fb[m])).sum(); da[r] = (np.abs(fa[m]) ** 2).sum(); db[r] = (np.abs(fb[m]) ** 2).sum()
    f = np.real(num) / (np.sqrt(da * db) + 1e-12)
    thr = 2.0 / np.sqrt(np.maximum(nq / 2.0, 1))
    return np.arange(nb) / (2.0 * nb), f, thr


def _res(freq, f, thr):
    fs = np.convolve(f, np.ones(5) / 5, "same")
    thr_arr = thr if np.ndim(thr) else np.full_like(fs, thr)
    above = fs > thr_arr
    if not above[:3].any():
        return np.nan
    for i in range(2, len(fs)):
        if above[i - 1] and not above[i]:
            return freq[i]
    return freq[-1]


def lowband_frc(A, B, frac=0.1, taper_frac=0.25):
    """LONG-RANGE fidelity = mean FRC over the lowest `frac` of the spatial-frequency range. In
    ptychography the low-q (long-range) phase is the LEAST-constrained quantity -- the transfer
    function is suppressed at low q (Ophus, arXiv:2309.05250) = the connection-Laplacian low-q modes
    (the Gramian/sync regime). So this low-band FRC, NOT the high-q resolution cutoff, is where sync /
    probe recovery lifts quality; report it alongside patched_frc's resolution. A,B must be gauge-
    aligned with the MINIMAL gauge only (global phase + shift + linear ramp = align_gauge poly_order=1);
    removing higher-order (defocus/astigmatism) phase would erase the very low-q signal this measures."""
    taper = int(min(A.shape) * taper_frac)
    freq, f, thr = _frc1(A, B, taper)
    m = (freq > 0) & (freq < frac * freq[-1])              # lowest decile, skip DC
    return float(f[m].mean())


def patched_frc(A, B, patch=96, stride=48, thr="2sig", taper_frac=0.25):
    """A=recon, B=GT (already globally gauge-aligned). Returns dict(median, std, p16, p84, res[], n).
    thr='2sig' (per-ring curve) or a float (e.g. 0.5). Resolution reported in cyc/px (higher=better).
    NB: report lowband_frc() TOO -- the low-q (long-range) fidelity is the sync-relevant quantity that
    the high-q resolution cutoff misses."""
    H, W = A.shape; taper = int(patch * taper_frac); out = []
    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):
            freq, f, t2 = _frc1(A[y:y + patch, x:x + patch], B[y:y + patch, x:x + patch], taper)
            out.append(_res(freq, f, t2 if thr == "2sig" else float(thr)))
    res = np.array(out); res = res[~np.isnan(res)]
    return dict(median=float(np.median(res)), std=float(res.std()),
                p16=float(np.percentile(res, 16)), p84=float(np.percentile(res, 84)),
                res=res, n=len(res))


def _band_limited_pair(cutoff, noise=0.3, H=384, W=384, seed=0):
    """synthetic (GT, recon) pair: GT band-limited to `cutoff` (cyc/px) + white noise on the recon.
    FRC decorrelates (drops) past the band edge -> patched-FRC resolution ~ cutoff. Resolution loss
    needs DECORRELATION (noise), not mere attenuation (FRC normalizes amplitude out)."""
    rng = np.random.default_rng(seed)
    fy, fx = np.meshgrid(np.fft.fftfreq(H), np.fft.fftfreq(W), indexing="ij"); q = np.hypot(fy, fx)
    F = np.fft.fft2(rng.standard_normal((H, W)) + 1j * rng.standard_normal((H, W)))
    A = np.fft.ifft2(F * (q < cutoff))                                   # GT, band-limited to cutoff
    n = rng.standard_normal((H, W)) + 1j * rng.standard_normal((H, W))
    B = A + noise * np.std(A) * n / np.std(n)                            # recon = GT + white noise
    return A, B


if __name__ == "__main__":
    for cut in (0.35, 0.15):
        A, B = _band_limited_pair(cut)
        r = patched_frc(A, B, patch=96, stride=48)
        pm = 1 / (2 * r["median"]); ps = pm * (r["std"] / max(r["median"], 1e-9))
        print(f"GT band-limited to {cut:.2f} cyc/px: patched-FRC resolution {r['median']:.3f}±{r['std']:.3f} "
              f"cyc/px  (period {pm:.2f}±{ps:.2f} px, n={r['n']})")
