#!/usr/bin/env python3
"""Inverse-iteration (A4) vs power vs eigsh for U(1) phase synchronization.

Tests the conjecture (Marchesini, arXiv:1209.4924 Appendix A, Eq. A4) that solving
the CONSTRAINED graph-Laplacian system  L w = alpha*1  with  sum(w)=const  -- i.e.
ONE step of INVERSE ITERATION on the Laplacian from w0 = ones -- beats power
iteration on the complementary adjacency, precisely in the regime where power is
slow: a strongly VARYING low-frequency object phase (so the answer is far from the
constant start).

Notation (bridging the two documents -- the letter "H" is FLIPPED between them):
  * report/sharpy:  H = Gramian/adjacency (overlap inner products);  Laplacian = D - H
  * paper (A4):     H = the whole Laplacian;  the adjacency is (I - H)
Here: W = real overlap weights, A = W .* exp(i(theta_i-theta_j)) = connection adjacency,
D = diag(row sums of W), Laplacian L = D - A, normalized adjacency Hn = D^-1/2 A D^-1/2
(top eigenvalue 1; sharpy power-iterates this), normalized Laplacian Ln = I - Hn.
The sync answer phi = exp(i*theta) is the null vector of L (Ln) and the top
eigenvector of Hn -- same operator, opposite ends of the spectrum.

The point (gap inversion): the near-degenerate TOP gap of Hn (lambda2/lambda1 ~ .99,
which makes power crawl) is a near-ZERO BOTTOM eigenvalue of Ln -- inverse iteration
mu_min/mu_2 ~ 0 lands it in ~1 solve, and the 1/(mu+eps) amplification overcomes a
SMALL <ones,phi> overlap that power cannot overcome in a few matvecs.

  ENV: G=64 SIGMA=1.6 R=4 CONTRAST=0.85 EPS=1e-4 SEED=0 python inverse_iter_sync_test.py
"""
import os, time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, spsolve, splu

G        = int(os.environ.get("G", 64))            # scan grid side; #frames = G^2
SIGMA    = float(os.environ.get("SIGMA", 1.6))     # overlap-kernel width
R        = int(os.environ.get("R", 4))             # kernel cutoff radius (frames)
CONTRAST = float(os.environ.get("CONTRAST", 0.85)) # absorbing-blob depth (non-uniform energy)
EPS      = float(os.environ.get("EPS", 1e-4))      # Laplacian shift for the inverse solve
SEED     = int(os.environ.get("SEED", 0))
N = G * G


def minority_sign(v):
    """Per-frame gauge-flip metric: fraction of w/|w| on the minority side after
    removing the global phase. ~0 = clean, ~0.5 = sign-flipping (Fiedler) mode."""
    ph = v / (np.abs(v) + 1e-30)
    g = ph.sum()
    ph = ph * np.conj(g) / (np.abs(g) + 1e-30)
    return float(np.mean(ph.real < 0))


def phase_err_rms(w, phi):
    """RMS per-frame phase error (rad) vs truth phi, after best global-phase align."""
    wn = w / (np.abs(w) + 1e-30)
    g = np.vdot(phi, wn)                       # global phase alignment
    wn = wn * np.conj(g) / (np.abs(g) + 1e-30)
    err = np.angle(wn * np.conj(phi))
    return float(np.sqrt(np.mean(err ** 2)))


def alignment(w, phi):
    """Mean |<w/|w|, phi>| in [0,1]; 1 = perfectly synchronized."""
    wn = w / (np.abs(w) + 1e-30)
    return float(np.abs(np.vdot(phi, wn)) / N)


def frame_weights(contrast):
    if contrast <= 0:
        return np.ones(N)
    xs, ys = np.meshgrid(np.linspace(0, 1, G), np.linspace(0, 1, G), indexing="ij")
    blob = np.zeros((G, G))
    for cx, cy, a in [(0.5, 0.5, 0.9), (0.3, 0.7, 0.7), (0.75, 0.35, 0.6)]:
        blob += a * np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / 0.02))
    return np.clip(1.0 - contrast * np.clip(blob, 0, 1), 0.1, 1.0).ravel()


def low_freq_phase(cycles):
    """Smooth radial-quadratic object phase with corner-to-corner swing cycles*2pi."""
    xs, ys = np.meshgrid(np.linspace(-1, 1, G), np.linspace(-1, 1, G), indexing="ij")
    r2 = (xs ** 2 + ys ** 2) / 2.0           # 0 at center, 1 at corner
    return (cycles * 2 * np.pi * r2).ravel()


def build_ops(s, theta):
    """Return (Hn, Ln, phi_true): normalized adjacency, normalized Laplacian, truth."""
    xs, ys = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    xs = xs.ravel(); ys = ys.ravel()
    ri, ci, wv, pv = [], [], [], []
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            d2 = dx * dx + dy * dy
            if d2 > R * R:
                continue
            xn, yn = xs + dx, ys + dy
            ok = (xn >= 0) & (xn < G) & (yn >= 0) & (yn < G)
            i = np.where(ok)[0]
            j = xn[ok] + yn[ok] * G
            w = s[i] * s[j] * np.exp(-d2 / (SIGMA * SIGMA))
            ri.append(i); ci.append(j); wv.append(w)
            pv.append(np.exp(1j * (theta[i] - theta[j])))      # connection phase
    ri = np.concatenate(ri); ci = np.concatenate(ci)
    wv = np.concatenate(wv); pv = np.concatenate(pv)
    W = sparse.csr_matrix((wv, (ri, ci)), shape=(N, N))               # real weights
    A = sparse.csr_matrix((wv * pv, (ri, ci)), shape=(N, N)).astype(complex)  # adjacency
    d = np.asarray(W.sum(axis=1)).ravel()                            # degree
    Dm12 = sparse.diags(1.0 / np.sqrt(d))
    Hn = (Dm12 @ A @ Dm12).tocsr()                                  # normalized adjacency
    Ln = (sparse.identity(N, dtype=complex, format="csr") - Hn).tocsr()  # I - Hn
    phi = np.exp(1j * theta)
    return Hn, Ln, phi


def power(Hn, it, v0=None):
    v = (np.ones(N, complex) if v0 is None else v0.copy())
    v /= np.linalg.norm(v)
    for _ in range(it):
        v = Hn @ v; v /= np.linalg.norm(v)
    return v


def inverse_iter(Ln, steps=1):
    """Inverse iteration on the (shifted) normalized Laplacian from ones -- the A4
    constrained solve. Factor (Ln+eps I) once, back-solve `steps` times."""
    M = (Ln + EPS * sparse.identity(N, dtype=complex, format="csr")).tocsc()
    lu = splu(M)                                  # one factorization (reusable across AP iters)
    v = np.ones(N, complex)
    for _ in range(steps):
        v = lu.solve(v); v /= np.linalg.norm(v)
    return v


def timed(fn):
    t0 = time.perf_counter(); out = fn(); return out, (time.perf_counter() - t0) * 1e3  # ms


def run(cycles):
    s = frame_weights(CONTRAST)
    theta = low_freq_phase(cycles)
    Hn, Ln, phi = build_ops(s, theta)

    # reference spectrum / gap from a dense solve (small N only)
    wsp = np.sort(eigsh(Hn, k=4, which="LA", maxiter=20000, tol=0,
                        return_eigenvectors=False))[::-1]
    gap = wsp[1] / wsp[0]
    ov = abs(np.vdot(np.ones(N) / np.sqrt(N), phi * np.sqrt(np.asarray(Hn.sum(0)).ravel()*0+1))) ** 2  # <ones,phi>
    ov = abs(np.vdot(np.ones(N, complex), phi)) ** 2 / N  # |<ones,phi>|^2 / N  (start overlap, phase only)

    print(f"\n=== cycles={cycles}  (corner swing {cycles}*2pi)  N={N}  "
          f"eig2/eig1={gap:.4f}   |<ones,phi>|^2/N={ov:.3f} ===")
    print(f"   {'method':<28}{'phase_err(rad)':>15}{'align':>9}{'flip':>7}{'ms':>9}")

    def show(name, v, ms):
        print(f"   {name:<28}{phase_err_rms(v, phi):>15.2e}{alignment(v, phi):>9.3f}"
              f"{minority_sign(v):>7.2f}{ms:>9.1f}")

    for it in (5, 20, 50, 100):
        v, ms = timed(lambda it=it: power(Hn, it))
        show(f"power(ones, {it})", v, ms)

    for mi in (20, 100):
        def _e(mi=mi):
            return eigsh(Hn, k=1, which="LA", v0=np.ones(N), ncv=3, maxiter=mi, tol=0)[1][:, 0]
        try:
            v, ms = timed(_e); show(f"eigsh(Hn, maxiter={mi})", v, ms)
        except Exception as e:
            print(f"   eigsh(maxiter={mi}) -> {type(e).__name__}")

    for steps in (1, 2):
        v, ms = timed(lambda steps=steps: inverse_iter(Ln, steps))
        show(f"inv-iter on Ln ({steps} solve)", v, ms)


if __name__ == "__main__":
    np.random.seed(SEED)
    print(f"Inverse-iteration (A4) vs power vs eigsh  "
          f"(G={G}, sigma={SIGMA}, R={R}, contrast={CONTRAST}, eps={EPS})")
    for c in (0.0, 0.5, 1.0, 2.0):
        run(c)
    print("\nExpected: flat phase (cycles=0) -> all fine. As the phase varies (cycles>=1),")
    print("<ones,phi> drops, power PLATEAUS within budget, but 1 Laplacian solve still nails it.")
