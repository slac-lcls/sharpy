# -*- coding: utf-8 -*-
import numpy as np
import copy
import h5py
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from Operators import Split_Overlap_plan
import Solvers
from Operators import get_times, reset_times, normalize_times
import config
from Operators import Gramiam_plan, Replicate_frame

Alternating_projections = Solvers.Alternating_projections
reset_times()
GPU = config.GPU #False
sync = True
debug = True
if GPU:
    import cupy as cp
    xp = cp
    print("using GPU")
else:
    xp = np
    print("using CPU")


##################################################
# input data
fname_in = "simulation_small.h5"

import sys
# Retrieve the value of 'fname' from the command-line arguments
#fname_in = sys.argv[1] if len(sys.argv) > 1 else None

fid = h5py.File(fname_in, "r")

data = xp.array(fid["data"], dtype=xp.float32)
illumination = xp.array(fid["probe"], dtype=xp.complex64)

translations = xp.array(fid["translations"])
#print('translations1', type(translations),translations.dtype) #float128 dtype
nframes, nx, ny = data.shape
resolution = (
    xp.float32(fid["wavelength"])
    * xp.float32(fid["detector_distance"])
    / (nx * xp.float32(fid["detector_pixel_size"]))
)
translations = translations / resolution
truth = xp.array(fid["truth"], dtype=xp.complex64)

translations_x = xp.array(fid["translations_x"]) #load the real and imag of dtype int64
translations_y = xp.array(fid["translations_y"])
fid.close()

#################################################

#translations_x = xp.real(translations)
#translations_y = xp.imag(translations)

# get the image extent (with wrap-around)
Nx = int(xp.ceil(xp.max(translations_x) - xp.min(translations_x)))
Ny = Nx

#
if GPU:
    mempool = cp.get_default_memory_pool()
    print(
        "loaded data, memory used, and total:",
        mempool.used_bytes(),
        mempool.total_bytes(),
    )
    print(
        "normalized by data.nbytes memory used and total normalized:",
        mempool.used_bytes() / data.nbytes,
        mempool.total_bytes() / data.nbytes,
    )
    print("data size", data.nbytes)
    print("----")


Split, Overlap = Split_Overlap_plan(translations_x, translations_y, nx, ny, Nx, Ny)

if GPU:
    mempool = cp.get_default_memory_pool()
    print(
        "Split and Overlap, memory used, and total:",
        mempool.used_bytes(),
        mempool.total_bytes(),
    )
    print(
        "normalized by data.nbytes memory used and total normalized:",
        mempool.used_bytes() / data.nbytes,
        mempool.total_bytes() / data.nbytes,
    )
    print("data size", data.nbytes)
    print("----")

if sync == True:
   #calculate the preconditioner here
    Gplan = Gramiam_plan(translations_x,translations_y,nframes,nx,ny,Nx,Ny, bw = 0)
    #print('plan',Gplan['col'],Gplan['row'],Gplan['dx'],Gplan['dy'],Gplan['val'])
    if Gplan['col'].size == 0:
        sync = False
else: 
    Gplan = None



#img_initial = xp.ones((Nx, Ny), dtype=xp.complex64)
img_initial = xp.ones((Nx, Ny), dtype=xp.complex64) #Need to match datatype in operators fft Plan2D
############################
# reconstruct
refine_illumination = False
maxiter = 30
# residuals_interval = np.inf
residuals_interval = 1

t0 = timer()
print("geometry: img size:", (Nx, Ny), "frames:", (nx, ny, nframes))
print(
    "not refining illumination, starting with good one, maxiter:",
    maxiter,
)

img4, frames, illum, residuals_AP = Alternating_projections(
    sync,
    img_initial,
    Gplan,
    illumination,
    Overlap,
    Split,
    data,
    refine_illumination,
    maxiter,
    normalization=None,
    img_truth =truth,
    residuals_interval = residuals_interval
)

print("total time:", timer() - t0)

############################


## print results
timers = get_times()


tt_ops = copy.deepcopy(timers)
tt_solver = copy.deepcopy(Solvers.get_times())

print(timers)


tots = normalize_times()
print("total time operators benchmarked:", tots)

print("after normalization")
timersn = get_times()
print(timersn)

tt = Solvers.get_times()
print("solver timers:\n", tt)
tots = Solvers.normalize_times(tt["solver_loop"])

print("solver timers after normalization by", tots, ":\n", Solvers.get_times())

print("mse timings:", tt["mse_step"] + tt["mse_truth"] + tt_ops["mse_data"])
print(
    "ops timings:",
    tt["ProxD"]
    + tt["Overlap"]
    + tt["illuminate&split"]
    + tt["refine_illumination"]
    + tt["copies"],
)


tots = (
    tt["ProxD"]
    + tt["Overlap"]
    + tt["illuminate&split"]
    + tt["refine_illumination"]
    + tt["copies"]
)
tots = Solvers.normalize_times(tots)
print(
    "solver timers after normalization by total prox ops",
    tots,
    ":\n",
    Solvers.get_times(),
)


# calculate mse
nrm0 = xp.linalg.norm(truth)
if residuals_AP.size > 0:
    nmse4 = residuals_AP[-1, 0]
else:
    nmse4 = np.NaN

if GPU:
    truth = truth.get()
    img = img4.get()
    residuals_AP = residuals_AP.get()
else:
    img = img4

fig = plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.title("Ground truth", fontsize=10)
plt.imshow(abs(truth), cmap="gray")
plt.colorbar()
plt.subplot(1, 3, 2)
plt.title("Alternating Projections\n Sync:%2.2g" % (nmse4), fontsize=10)
plt.imshow(abs(img), cmap="gray")
plt.colorbar()
plt.subplot(1, 3, 3)
plt.title("Difference")
plt.imshow(abs(truth) - abs(img), cmap="jet")
plt.colorbar()
plt.show()

##

# make a new figure with residuals
fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10, 10))
axs[0].semilogy(residuals_AP[:, 0])
axs[0].set_title("|img-f truth| (f=phase scalar)")
axs[1].semilogy(residuals_AP[:, 1])
axs[1].set_title("||frames|-data|")
axs[2].semilogy(residuals_AP[:, 2])
axs[2].set_title("frames overlapped")
plt.show()

print(
    "solver tot (seconds) :",
    tt_solver["solver_tot"],
    "loop:",
    tt_solver["solver_loop"],
    "seconds",
)
print("proxD :", tt_solver["ProxD"] / tt_solver["solver_loop"] * 100, "%")
print("Overlap :", tt_solver["Overlap"] / tt_solver["solver_loop"] * 100, "%")
print(
    "illuminate and split :",
    tt_solver["illuminate&split"] / tt_solver["solver_loop"] * 100,
    "%",
)
print(
    "mse:",
    (tt_solver["mse_step"] + tt_solver["mse_truth"] + tt_ops["mse_data"])
    / tt_solver["solver_loop"]
    * 100,
    "%",
)
print(
    "proxD (propagate(ffts), mse, prox)%:",
    (
        tt_ops["Propagate"] / tt_ops["Data_prox_tot"] * 100,
        tt_ops["mse_data"] / tt_ops["Data_prox_tot"] * 100,
        tt_ops["Prox_data"] / tt_ops["Data_prox_tot"] * 100,
    ),
)
