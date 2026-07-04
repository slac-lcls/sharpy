"""Matrix-free shift-invert Lanczos: does feeding eigsh the invit inverse operator
(L+eI)^-1 via CG cure the degenerate-cluster ambiguity that plain eigsh(H) has?

CPU/scipy (authoritative). Build a degenerate truth-frame Gramian, then recover the
consensus phase omega by:
  (a) plain eigsh(H, LM)          -- run across random v0 seeds -> SPREAD = the bug
  (b) shift-invert eigsh(Minv,LM) -- Minv=CG solve of (L+eI); v0=ones -> stable?
  (c) invit (1-2 CG from ones)    -- the current fix
Metric: |<omega, omega_ref>| where omega_ref = dense-eigh bottom of L (ground truth).
"""
import os, sys, time
sys.path.insert(0, "/Users/smarches/git/sharpy/sharpy")
os.environ.setdefault("NX", "16")
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, cg, LinearOperator
import sync_bandpass_test as T
from Operators import Illuminate_frames, Gramiam_calc, xp

K = int(os.environ.get("K", 80))
EPS = float(os.environ.get("EPS", 1e-3))
ctx = T.build(K)
tf = Illuminate_frames(T.Splitc(ctx["truth"], ctx["mapid"]), ctx["probe"])
fl = Illuminate_frames(tf, np.conj(ctx["probe"]))
H = Gramiam_calc(fl, fl * ctx["inorm_split"], ctx["Gramiam"], ctx["frames_norm"]).tocsr()
n = H.shape[0]
print(f"K={K}: {n} frames; building normalized Laplacian...")

absH = H.copy(); absH.data = np.abs(absH.data)
d = np.maximum(np.asarray(absH.sum(1)).ravel().real, 1e-30)
s = (1.0/np.sqrt(d)).astype(H.dtype)
Hn = sp.diags(s) @ H @ sp.diags(s)
L = sp.identity(n, dtype=H.dtype) - Hn
M = (L + EPS*sp.identity(n, dtype=H.dtype)).tocsr()

def omega(v):
    w = v/(np.abs(v)+1e-30); ss = np.conj(w.sum()); ss/= (abs(ss)+1e-30); return (w*ss)

# ground truth: dense-eigh smallest mode of L (the consensus)
lam, V = np.linalg.eigh(L.toarray())
oref = omega(V[:, 0])                 # smallest Laplacian eigenvalue = consensus
gap = float(lam[1]-lam[0])
print(f"Laplacian bottom gap lam2-lam1 = {gap:.2e}")

def align(o): return float(abs(np.vdot(oref, o))/n)

# (a) plain eigsh(H) across random seeds
print("\n(a) plain eigsh(H, LM) across 5 random starts:")
al = []
for seed in range(5):
    rng = np.random.default_rng(seed)
    v0 = (rng.standard_normal(n)+1j*rng.standard_normal(n)).astype(H.dtype)
    la, Va = eigsh(H.astype(np.complex128), k=1, which="LM", v0=v0, ncv=20, maxiter=300)
    a = align(omega(Va[:,0])); al.append(a)
print(f"    align: {[f'{a:.3f}' for a in al]}  spread {max(al)-min(al):.3f}")

# (b) matrix-free shift-invert eigsh: Minv x = cg((L+eI), x); v0=ones
def minv_mv(x):
    y,_ = cg(M, x, rtol=1e-6, maxiter=500); return y
Minv = LinearOperator((n,n), matvec=minv_mv, dtype=H.dtype)
print("\n(b) matrix-free shift-invert eigsh(Minv, LM), v0=ones, 3 runs:")
al=[]; t0=time.perf_counter()
for r in range(3):
    lb, Vb = eigsh(Minv, k=1, which="LM", v0=np.ones(n, dtype=H.dtype), ncv=20, maxiter=300)
    al.append(align(omega(Vb[:,0])))
dt=(time.perf_counter()-t0)/3
print(f"    align: {[f'{a:.3f}' for a in al]}  spread {max(al)-min(al):.3f}  {1e3*dt:.0f}ms/solve")

# (c) invit: 2 CG solves of (L+eI)x=1 from ones
print("\n(c) invit (2 CG solves of (L+eI)x=1 from ones):")
x = np.ones(n, dtype=H.dtype)
t0=time.perf_counter()
for _ in range(2):
    x,_ = cg(M, x, rtol=1e-8, maxiter=500); x/=np.linalg.norm(x)
dt=time.perf_counter()-t0
print(f"    align: {align(omega(x)):.3f}  {1e3*dt:.0f}ms")
