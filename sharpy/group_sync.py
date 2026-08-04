"""Group synchronization on the ptychography overlap graph: recover a per-frame gauge (global phase,
and -- extending the classic angular sync -- a 2D SHIFT and 2D TILT) that is globally consistent with
the pairwise relative gauges measured between overlapping frames (e.g. by gauge_align). The gauge group
is the Weyl-Heisenberg group (shift & tilt are conjugate phase-space translations, their commutator is
the phase); it factors because shift and tilt are each ABELIAN R^2:

  * shift-sync  = translation synchronization  = weighted graph-Laplacian LEAST SQUARES  (L x = b)
  * tilt-sync   = the same in the FOURIER (frequency) coordinate
  * phase-sync  = U(1) angular synchronization = top eigenvector of the Hermitian connection matrix

and the small s x t -> phase (Heisenberg) coupling is folded in by a few iterations. Only OVERLAPPING
frame pairs are edges (sparse), so this is O(N*k), not O(N^2). The value over local (per-frame)
refinement: the capture range is set by the pairwise NEIGHBOR difference, not the absolute error, so it
is far larger for CORRELATED/DRIFT errors (thermal, stage creep, FEL pointing wander).

Refs: A. Singer, "Angular synchronization by eigenvectors and semidefinite programming," ACHA 30 (2011)
(U(1)); Singer & Wu, "Vector diffusion maps and the connection Laplacian," CPAM 65 (2012). Pure numpy.
"""
import numpy as np


def build_overlap_graph(pos, radius):
    """edges = frame pairs whose scan positions are within `radius` (overlapping footprints). pos (N,2)."""
    pos = np.asarray(pos, float); N = len(pos)
    edges = []
    for k in range(N):
        d = np.hypot(pos[k, 0] - pos[k + 1:, 0], pos[k, 1] - pos[k + 1:, 1])
        for j in np.where(d < radius)[0]:
            edges.append((k, k + 1 + int(j)))
    return edges


def sync_translation(edges, delta, N, w=None, anchor=0):
    """R^d synchronization: recover per-node x_k (up to a global constant) from pairwise x_k - x_l =
    delta_kl, by weighted graph-Laplacian least squares (L x = b). delta (E,d), edges list of (k,l).
    Returns (N,d) anchored so x[anchor]=0. (Use for the SHIFT channel; and, on Fourier-domain deltas,
    the TILT channel.)"""
    delta = np.atleast_2d(np.asarray(delta, float)); d = delta.shape[1]
    if w is None:
        w = np.ones(len(edges))
    L = np.zeros((N, N)); b = np.zeros((N, d))
    for (k, l), dkl, wkl in zip(edges, delta, w):
        L[k, k] += wkl; L[l, l] += wkl; L[k, l] -= wkl; L[l, k] -= wkl
        b[k] += wkl * dkl; b[l] -= wkl * dkl
    idx = [i for i in range(N) if i != anchor]
    x = np.zeros((N, d))
    x[idx] = np.linalg.solve(L[np.ix_(idx, idx)], b[idx])      # L singular (null=const) -> pin anchor
    return x - x[anchor]


def sync_phase(edges, dtheta, N, w=None, anchor=0):
    """U(1) angular synchronization: recover theta_k (up to a global phase) from pairwise theta_k -
    theta_l = dtheta_kl, as the top eigenvector of the Hermitian connection matrix H_kl = w e^{i dtheta}."""
    if w is None:
        w = np.ones(len(edges))
    H = np.zeros((N, N), complex)
    for (k, l), dt, wkl in zip(edges, dtheta, w):
        H[k, l] += wkl * np.exp(1j * dt); H[l, k] += wkl * np.exp(-1j * dt)
    vals, vecs = np.linalg.eigh(H)
    v = vecs[:, int(np.argmax(vals))]
    return np.angle(v * np.exp(-1j * np.angle(v[anchor])))


def sync_heisenberg(edges, dS, dT, dTheta, N, w=None, iters=3):
    """Full shift+tilt+phase sync. dS,dT (E,2) pairwise shift/tilt differences, dTheta (E,) phase. Shift
    & tilt by sync_translation, phase by sync_phase, iterating the 1/2 (s x t) Heisenberg cocycle (2D
    cross product = scalar). Returns (s (N,2), t (N,2), theta (N,))."""
    s = sync_translation(edges, dS, N, w); t = sync_translation(edges, dT, N, w)
    cross = lambda a, b: a[0] * b[1] - a[1] * b[0]
    theta = np.zeros(N)
    for _ in range(iters):
        dTh = np.array([dTheta[e] - 0.5 * (cross(s[k], t[k]) - cross(s[l], t[l]))
                        for e, (k, l) in enumerate(edges)])
        theta = sync_phase(edges, dTh, N, w)
    return s, t, theta
