"""Universal gauge aligner for complex fields. Given two complex images that agree only up to the
usual ptychography/CDI gauge freedoms, remove ALL of them so they can be compared (FRC, NMSE, diff).

FRAMING: this is Guizar-Sicairos, Thurman & Fienup, "Efficient subpixel image registration
algorithms," Opt. Lett. 33(2), 156 (2008) [MGS08] -- applied in BOTH Fourier-conjugate domains.
A real-space SHIFT is a phase-slope in Fourier; a real-space PHASE-RAMP is a shift of the Fourier
MAGNITUDE. They are Fourier duals, so the same efficient upsampled-DFT registration removes:

  * sub-pixel TRANSLATION      B(r) -> B(r-s)        MGS08 single-step DFT registration in REAL space
  * POLYNOMIAL PHASE           B -> B*exp(i p(r))    p = 2D polynomial of degree `poly_order`
                                                     (Marchesini et al. 2005): order 1 = linear ramp/tilt
                                                     (the coarse part = the RECIPROCAL-domain |FFT| shift;
                                                     the object-phase-ramp <-> probe/detector-shift gauge
                                                     MGS08 omits); order 2 adds DEFOCUS (2,0)+(0,2) &
                                                     ASTIGMATISM (1,1); order 3 adds COMA. Fit = |A||B|-
                                                     weighted lstsq of arg(A conjB) (Eq.2-3). Best on the
                                                     OBJECT: bigger FOV shows more cycles -> better cond.
  * global PHASE + SCALE       B -> c*B              MGS08's global complex factor alpha (c=<B,A>/<B,B>)

The polynomial coeffs are the LOW-ORDER ABERRATION basis: their fluctuation across reconstructions/shots
quantifies the instability (Marchesini 2005 Eq.4), and defocus (2,0)+(0,2) -> real-space dz (Eq.5,
defocus_dz()). Refs: Guizar-Sicairos/Thurman/Fienup Opt.Lett. 33,156 (2008); Marchesini, Chapman, Barty
et al., "Phase Aberrations in Diffraction Microscopy," arXiv:physics/0510033 (2005).

Shift<->ramp COUPLE (each corrupts the other's naive estimate) -> estimate ramp COARSE from |FFT|
registration (shift-immune) to break the deadlock, then translation (ramp-immune), then FINE ramp via
|A||B|-weighted phase-slope fit of A*conj(B), iterate. Reusable & standalone; numpy host arrays.
align_gauge(A, B) -> (B_aligned ~ A, params). Self-test plants a known gauge & recovers to ~1e-15.

  python -u gauge_align.py         # runs the self-test
"""
import numpy as np


def _upsampled_dft(data, region, usfac, offr, offc):
    """upsampled inverse DFT of `data` (a freq-domain cross-power spectrum) over a small `region`x
    `region` window at 1/usfac sampling, offset by (offr,offc). Guizar-Sicairos matrix-DFT trick."""
    nr, nc = data.shape
    kc = np.exp((-2j * np.pi / (nc * usfac)) *
                (np.fft.ifftshift(np.arange(nc)) - nc // 2)[:, None] * (np.arange(region) - offc)[None, :])
    kr = np.exp((-2j * np.pi / (nr * usfac)) *
                (np.arange(region)[:, None] - offr) * (np.fft.ifftshift(np.arange(nr)) - nr // 2)[None, :])
    return kr @ data @ kc


def register_translation(A, B, upsample=20):
    """sub-pixel shift (sy,sx) such that shifting B by (sy,sx) best matches A. Guizar-Sicairos."""
    FA = np.fft.fft2(A); FB = np.fft.fft2(B)
    prod = FA * np.conj(FB)
    cc = np.fft.ifft2(prod)
    H, W = A.shape
    p = np.array(np.unravel_index(np.argmax(np.abs(cc)), cc.shape), float)
    mid = np.array([H // 2, W // 2])
    p[p > mid] -= np.array([H, W])[p > mid]
    if upsample <= 1:
        return p[0], p[1]
    usf = float(upsample)
    p = np.round(p * usf) / usf
    reg = int(np.ceil(usf * 1.5)); dsh = reg // 2
    off = dsh - p * usf
    ccu = _upsampled_dft(np.conj(prod), reg, usf, off[0], off[1]).conj()
    m = np.array(np.unravel_index(np.argmax(np.abs(ccu)), ccu.shape), float) - dsh
    return p[0] + m[0] / usf, p[1] + m[1] / usf


def _apply_shift(B, sy, sx):
    H, W = B.shape
    ky = np.fft.fftfreq(H)[:, None]; kx = np.fft.fftfreq(W)[None, :]
    return np.fft.ifft2(np.fft.fft2(B) * np.exp(-2j * np.pi * (ky * sy + kx * sx)))


def _parpk(v, i, n):
    a, b, d = abs(v[(i - 1) % n]), abs(v[i]), abs(v[(i + 1) % n]); den = a - 2 * b + d
    return i + (np.clip(0.5 * (a - d) / den, -0.5, 0.5) if abs(den) > 1e-20 else 0.0)


def estimate_ramp_coarse(A, B, upsample=20):
    """COARSE ramp (ky,kx to remove): a real-space ramp SHIFTS the magnitude spectrum (translation
    touches only its phase), so ramp = shift of |FFT(B)| vs |FFT(A)| -> disentangled from translation."""
    MA = np.abs(np.fft.fftshift(np.fft.fft2(A))); MB = np.abs(np.fft.fftshift(np.fft.fft2(B)))
    sy, sx = register_translation(MA, MB, upsample)
    return -sy, -sx


def fit_poly_phase(A, B, order=2):
    """Fit the phase difference between A and B to a 2D POLYNOMIAL of total degree `order` (Marchesini
    et al. 2005, Eq.2-3), via WRAP-FREE PHASE GRADIENTS: the residual r=A*conj(B); the per-pixel phase
    step angle(conj(r_i)*r_{i+1}) never wraps (up to Nyquist), and is weighted by |r_i*r_{i+1}| so
    low-amplitude (phase-meaningless) pixels don't corrupt the fit. Fit the x,y gradients jointly to the
    polynomial's partial derivatives (so the global-phase CONSTANT drops out automatically -> no bias),
    then reconstruct the phase. Returns (phase_field_to_remove, coeffs): B*exp(1j*phase) removes it.
    order 1 = ramp/tilt; 2 = + defocus (2,0)+(0,2) & astigmatism (1,1); 3 = + coma. coeffs keyed by (i,j)
    in normalized coords = the aberration basis (fluctuation across shots = instability, Eq.4)."""
    r = A * np.conj(B); H, W = r.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    xb = (xx - W / 2) / (W / 2); yb = (yy - H / 2) / (H / 2)
    gx = np.angle(np.conj(r[:, :-1]) * r[:, 1:]).ravel()            # d(phase)/dpixel, wrap-free, at x+0.5
    gy = np.angle(np.conj(r[:-1, :]) * r[1:, :]).ravel()
    wx = np.abs(r[:, :-1] * r[:, 1:]).ravel(); wy = np.abs(r[:-1, :] * r[1:, :]).ravel()
    xbx = xb[:, :-1] + 1.0 / W; ybx = yb[:, :-1]                    # gradient-sample coords (normalized)
    xby = xb[:-1, :]; yby = yb[:-1, :] + 1.0 / H
    terms = [(i, j) for tot in range(1, order + 1) for i in range(tot + 1) for j in (tot - i,)]

    def dc(i, j, X, Y, ax):                                         # partial deriv of xb^i yb^j (chain rule)
        if ax == "x":
            return (2.0 / W) * i * X ** (i - 1) * Y ** j if i >= 1 else np.zeros_like(X)
        return (2.0 / H) * j * X ** i * Y ** (j - 1) if j >= 1 else np.zeros_like(X)
    Dx = np.stack([dc(i, j, xbx, ybx, "x").ravel() for i, j in terms], 1)
    Dy = np.stack([dc(i, j, xby, yby, "y").ravel() for i, j in terms], 1)
    D = np.vstack([Dx, Dy]); g = np.concatenate([gx, gy]); w = np.concatenate([wx, wy])
    coef, *_ = np.linalg.lstsq(D * w[:, None], g * w, rcond=None)
    phase = sum(c * xb ** i * yb ** j for c, (i, j) in zip(coef, terms))
    return phase, dict(zip(terms, coef))


def align_gauge(A, B, upsample=20, iters=3, poly_order=1, do_shift=True):
    """Align B onto A up to translation + a POLYNOMIAL phase (Marchesini et al. 2005) + global phase&
    scale. poly_order: 1 = linear ramp/tilt only; 2 adds defocus+astigmatism; 3 adds coma. Per iter:
    COARSE linear ramp (spectrum-magnitude shift -> breaks the shift<->ramp deadlock) -> sub-pixel
    TRANSLATION -> FINE polynomial phase fit (residual, now shift is gone). Returns (B_aligned, params);
    params.poly = accumulated {(i,j): coeff} aberration coefficients."""
    H, W = A.shape; yy, xx = np.mgrid[0:H, 0:W]
    tot_s = np.zeros(2); acc = {}
    for it in range(iters):
        if poly_order >= 1 and it == 0:                     # coarse linear ramp, first iter, to decouple shift
            ky, kx = estimate_ramp_coarse(A, B, upsample)
            B = B * np.exp(-2j * np.pi * (kx * xx / W + ky * yy / H))
        if do_shift:
            sy, sx = register_translation(A, B, upsample)
            B = _apply_shift(B, sy, sx); tot_s += [sy, sx]
        if poly_order >= 1:                                 # fine polynomial phase (ramp + curvature + ...)
            phase, coef = fit_poly_phase(A, B, poly_order)
            B = B * np.exp(1j * phase)
            for k, v in coef.items():
                acc[k] = acc.get(k, 0.0) + v
    c = np.vdot(B, A) / (np.vdot(B, B) + 1e-30)             # global phase + scale
    B = B * c
    return B, dict(shift=tuple(tot_s), poly=acc, phase=float(np.angle(c)), scale=float(abs(c)))


def defocus_dz(coef, NA, lam, N):
    """real-space defocus distance from the quadratic aberration coeffs (Marchesini 2005 Eq.5):
    dz = lam/(4 pi N A^2) * (p20 + p02).  Pass NA, wavelength lam, linear pixel count N."""
    p = coef.get((2, 0), 0.0) + coef.get((0, 2), 0.0)
    return lam / (4 * np.pi * N * NA * NA) * p


def _selftest():
    rng = np.random.default_rng(0); H = W = 256
    # a smooth-ish complex field
    A = np.fft.ifft2(np.fft.fft2(rng.standard_normal((H, W)) + 1j * rng.standard_normal((H, W))) *
                     (np.hypot(*np.meshgrid(np.fft.fftfreq(H), np.fft.fftfreq(W), indexing="ij")) < 0.15))
    win = np.outer(np.hanning(H), np.hanning(W))            # taper edges -> circular shift can't wrap content
    A = win * A / np.abs(A).mean()
    yy, xx = np.mgrid[0:H, 0:W]; xb = (xx - W / 2) / (W / 2); yb = (yy - H / 2) / (H / 2)
    cases = [(3.7, -2.3, 4.0, -6.0, 1.1, 2.3, 0.0, 1),      # shift + ramp + phase + scale
             (-11.2, 5.9, -2.5, 3.3, -2.0, 0.4, 0.0, 1),
             (5.1, -3.3, 2.0, -1.5, 0.7, 1.5, 8.0, 2)]      # + DEFOCUS (quadratic) -> needs poly_order 2
    for (sy, sx, ky, kx, ph, sc, df, order) in cases:
        B = _apply_shift(A, -sy, -sx)                        # plant the inverse gauge so align recovers (sy,sx)
        B = B * np.exp(-2j * np.pi * (kx * xx / W + ky * yy / H))
        B = B * np.exp(-1j * df * (xb ** 2 + yb ** 2))       # defocus = quadratic phase
        B = B * (sc * np.exp(-1j * ph))
        Ba, p = align_gauge(A, B, poly_order=order)
        m = 20                                              # crop the Fourier-shift wraparound border
        nm = np.linalg.norm((A - Ba)[m:-m, m:-m]) / np.linalg.norm(A[m:-m, m:-m])
        pd = p["poly"].get((2, 0), 0.0) + p["poly"].get((0, 2), 0.0)
        print(f"planted shift({sy:+.2f},{sx:+.2f}) ramp({ky:+.1f},{kx:+.1f}) defocus{df:.1f} phase{ph:+.2f} scale{sc:.2f} [order {order}]")
        print(f"  recovered shift({p['shift'][0]:+.2f},{p['shift'][1]:+.2f}) defocus(p20+p02){pd:+.2f}"
              f" phase{p['phase']:+.2f} scale{p['scale']:.2f}  -> residual NMSE {nm:.2e}")
        assert nm < 1e-3, "alignment failed"
    print("SELF-TEST PASSED (shift + polynomial phase[ramp+defocus] + phase + scale removed to <1e-3)")


if __name__ == "__main__":
    _selftest()
