"""3-tier streaming OD2P (thread #4, Stage B): stream tile frame-data through
DISK(memmap) -> HOST(LRU cache) -> DEVICE, one fetch per tile per OUTER step reused for
K_local AP iters. Proves (on the Mac, correctness + I/O accounting; real timings -> GPU):
  (1) the streamed recon is NUMERICALLY IDENTICAL to the in-RAM OD2P (Stage A) -- memmap
      returns the same bytes, so tiering is transparent;
  (2) K_local REUSE: data is read once per tile per outer and reused K_local iters, so disk
      traffic is amortized 1/K_local vs naive per-iteration streaming (the roofline lever --
      per-iter streaming is transfer-bound, gpu-memory-scaling memory);
  (3) the HOST tier turns disk into COLD-LOAD-ONCE when the working set fits (disk_reads =
      n_tiles total); when host < dataset, disk re-reads scale with n_outer -> exactly the
      regime where the K_local reuse must be large enough to pay (the disk-tier "makes sense"
      condition).

CPU / numpy authoritative.  /opt/anaconda3/bin/python3 od2p_stream_test.py
env: NX(16) K(24) STEPD(4) TILEK(4) HALO(1) NOUT(10) KLOC(10) HOSTCAP(big|small)
"""
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NX", "16")

import numpy as np
import sync_bandpass_test as T
from Operators import (Illuminate_frames, Splitc, Overlapc, Project_data,
                       Precondition_calc, xp)
from od2p_coarse_only_test import make_tiles, coarse_gauge, od2p_run

nx = T.nx
K = int(os.environ.get("K", 24))
TILEK = int(os.environ.get("TILEK", 4))
HALO = int(os.environ.get("HALO", 1))
NOUT = int(os.environ.get("NOUT", 10))
KLOC = int(os.environ.get("KLOC", 10))


class TieredStore:
    """DISK(np.memmap) -> HOST(LRU dict, capacity host_cap tiles) -> DEVICE(xp.asarray).
    Instruments bytes/reads per tier so the reuse + cold-load behavior is measurable."""

    def __init__(self, data_np, path, host_cap):
        self.mm = np.memmap(path, dtype=np.float32, mode="w+", shape=data_np.shape)
        self.mm[:] = data_np
        self.mm.flush()
        self.host_cap = host_cap                 # max tiles held in host RAM (None = unbounded)
        self.cache = OrderedDict()               # tile_id -> host ndarray
        self.stat = dict(disk_reads=0, disk_MB=0.0, host_hits=0, host_MB=0.0,
                         dev_MB=0.0, dev_xfers=0)

    def fetch(self, tile_id, idx):
        if tile_id in self.cache:                # HOST hit
            self.cache.move_to_end(tile_id)
            arr = self.cache[tile_id]
            self.stat["host_hits"] += 1
            self.stat["host_MB"] += arr.nbytes / 1e6
        else:                                    # DISK read -> cache (LRU evict)
            arr = np.array(self.mm[idx])
            self.stat["disk_reads"] += 1
            self.stat["disk_MB"] += arr.nbytes / 1e6
            self.cache[tile_id] = arr
            if self.host_cap is not None and len(self.cache) > self.host_cap:
                self.cache.popitem(last=False)
        dev = xp.asarray(arr)                     # stage to DEVICE (PCIe on real GPU)
        self.stat["dev_xfers"] += 1
        self.stat["dev_MB"] += arr.nbytes / 1e6
        return dev


def od2p_stream_run(ctx, tiles, store, mode, n_out, k_loc):
    """Same block-Jacobi OD2P as od2p_run, but each tile's data is FETCHED through the
    tiered store ONCE per outer and reused for k_loc local iters."""
    probe, cprobe = ctx["probe"], ctx["cprobe"]
    Nx, Ny = ctx["Nx"], ctx["Ny"]
    gobj = xp.ones((Ny, Nx), dtype=xp.complex64)
    covs = [t[3] for t in tiles]
    den = sum(covs); den = xp.where(xp.abs(den) < 1e-30, xp.complex64(1), den)
    for _ in range(n_out):
        objs = []
        for tid, (idx, mapid_t, norm_safe, cov_t) in enumerate(tiles):
            data_t = store.fetch(tid, idx)                 # ONE fetch, reused k_loc iters
            fr = Illuminate_frames(Splitc(gobj, mapid_t), probe)
            for _ in range(k_loc):
                fr, _ = Project_data(fr, data_t)
                obj_t = Overlapc(Illuminate_frames(fr, cprobe), Nx, Ny, mapid_t) / norm_safe
                fr = Illuminate_frames(Splitc(obj_t, mapid_t), probe)
            objs.append(obj_t)
        if mode == "coarse":
            g = coarse_gauge(objs, covs)
            objs = [o * xp.conj(g[k]) for k, o in enumerate(objs)]
        gobj = sum(c * o for c, o in zip(covs, objs)) / den
    return T.band_err(ctx, gobj)


if __name__ == "__main__":
    import tempfile
    T.STEPD = int(os.environ.get("STEPD", 4))
    T.NUMITER = 5
    ctx = T.build(K)
    data = ctx["data"]
    ctx["frames_norm"] = Precondition_calc(data, bw=ctx["Gramiam"]["bw"])
    tiles = make_tiles(ctx, TILEK, HALO)
    ntiles = len(tiles)
    data_np = np.asarray(data).astype(np.float32)
    frame_MB = data_np[0].nbytes / 1e6
    tot_MB = data_np.nbytes / 1e6

    # in-RAM reference (Stage A)
    ref = od2p_run(ctx, tiles, data, "coarse", NOUT, KLOC)

    tmp = tempfile.mkdtemp()
    print(f"K={K} ({ctx['nframes']} fr x {nx}), TILEK={TILEK} -> {ntiles} tiles, "
          f"NOUT={NOUT} KLOC={KLOC}; dataset {tot_MB:.2f} MB ({frame_MB*1e3:.1f} KB/frame)")
    print(f"in-RAM OD2P (Stage A):        lo={ref[0]:.5f} hi={ref[1]:.5f}")
    for cap_label, host_cap in (("host=BIG (fits all)", None), ("host=SMALL (1 tile)", 1)):
        store = TieredStore(data_np, os.path.join(tmp, f"d_{host_cap}.dat"), host_cap)
        out = od2p_stream_run(ctx, tiles, store, "coarse", NOUT, KLOC)
        s = store.stat
        match = "IDENTICAL" if abs(out[0] - ref[0]) < 1e-6 and abs(out[1] - ref[1]) < 1e-6 else "DIFFERS"
        naive_MB = ntiles * NOUT * KLOC * (data_np[tiles[0][0]].nbytes / 1e6)  # per-iter streaming
        reuse = (s["dev_MB"] * KLOC) / max(s["disk_MB"], 1e-9)
        print(f"\n[{cap_label}]  streamed lo={out[0]:.5f} hi={out[1]:.5f}  -> {match} to in-RAM")
        print(f"   DISK reads={s['disk_reads']:>4}  {s['disk_MB']:>7.2f} MB   |   "
              f"HOST hits={s['host_hits']:>4}  {s['host_MB']:>7.2f} MB   |   "
              f"DEVICE xfers={s['dev_xfers']:>4}  {s['dev_MB']:>7.2f} MB")
        print(f"   disk traffic {s['disk_MB']:.2f} MB vs naive per-iter-streaming "
              f"{naive_MB:.2f} MB  = {naive_MB / max(s['disk_MB'], 1e-9):.0f}x less "
              f"(K_local reuse={KLOC})")
    print("\nEXPECT: streamed recon IDENTICAL to in-RAM (tiering transparent). host=BIG -> disk "
          "read ONCE per tile (cold load), rest HOST hits. host=SMALL(<dataset) -> disk re-read "
          "every outer -> disk traffic scales with NOUT, and the K_local reuse is what keeps it "
          "affordable (naive per-iter streaming moves K_local x more). Real BW/timing -> GPU.")
