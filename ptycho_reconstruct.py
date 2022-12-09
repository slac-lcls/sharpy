# -*- coding: utf-8 -*-
import numpy as np
import cupy as cp

import h5py
import matplotlib.pyplot as plt

# from Operators import Split_plan, Overlap_plan #, Project_data
from Operators import Split_Overlap_plan  # , Project_data
import Solvers

Alternating_projections = Solvers.Alternating_projections


from Operators import get_times, reset_times, normalize_times

reset_times()

from timeit import default_timer as timer

import config

GPU = config.GPU

if GPU:
    xp = cp
    print("using GPU")
else:
    xp = np
    print("using CPU")


##################################################
# input data
fname_in = "simulation.h5"
fid = h5py.File(fname_in, "r")

data = xp.array(fid["data"], dtype=xp.float32)
illumination = xp.array(fid["probe"], dtype=xp.complex64)

translations = xp.array(fid["translations"])

nframes, nx, ny = data.shape
resolution = (
    xp.float32(fid["wavelength"])
    * xp.float32(fid["detector_distance"])
    / (nx * xp.float32(fid["detector_pixel_size"]))
)

translations = translations / resolution

truth = xp.array(fid["truth"], dtype=xp.complex64)
fid.close()

#################################################

translations_x = xp.real(translations)
translations_y = xp.imag(translations)

# get the image extent (with wrap-around)
# Nx,Ny = truth.shape
Nx = xp.int(xp.ceil(xp.max(translations_x) - xp.min(translations_x)))
Ny = Nx


#%%
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

# Split = Split_plan(translations_x,translations_y,nx,ny,Nx,Ny)
# Overlap = Overlap_plan(translations_x,translations_y,nx,ny,Nx,Ny)

img_initial = xp.ones((Nx, Ny), dtype=xp.complex64)
############################
# reconstruct

maxiter = 100
t0 = timer()
print("geometry: img size:", (Nx, Ny), "frames:", (nx, ny, nframes))
print(
    "not refining illumination, starting with good one, maxiter:",
    maxiter,
)
residuals_interval = 1
residuals_interval = np.inf


img4, frames, illum, residuals_AP = Alternating_projections(
    img_initial,
    illumination,
    Overlap,
    Split,
    data,
    refine_illumination=False,
    maxiter=maxiter,
    img_truth=truth,
    residuals_interval=residuals_interval,
)
print("total time:", timer() - t0)

#%%

"""

lens_true=xp.fft.fft2(xp.fft.fftshift(illumination))
lens_init=xp.abs(lens_true) # remove true phase
illum_init=xp.fft.fftshift(xp.fft.ifft2(lens_init))


img3,frames, residuals_AP3 = Alternating_projections(img_initial, illumination, Overlap, Split, data, refine_illumination = False, maxiter = maxiter, img_truth = truth)

print('not refining illumination, starting with the wrong one')
img3,frames, residuals_AP3 = Alternating_projections(img_initial, illum_init, Overlap, Split, data, refine_illumination = False, maxiter = maxiter, img_truth = truth)


print('refining illumination')
img2,frames, residuals_AP2 = Alternating_projections(img_initial, illum_init, Overlap, Split, data, refine_illumination = True, maxiter = maxiter, img_truth = truth)

"""

############################


## print results
timers = get_times()

import copy

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
# nmse4=mse_calc(truth,img4)/nrm0
if residuals_AP.size > 0:
    nmse4 = residuals_AP[-1, 0]
else:
    nmse4 = np.NaN

fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 10))

if GPU:
    truth = truth.get()
    img = img4.get()
    residuals_AP = residuals_AP.get()

else:
    img = img4


axs[0].set_title("Truth", fontsize=10)
axs[0].imshow(abs(truth), cmap="gray")
axs[1].set_title("Alternating Projections no Sync:%2.2g" % (nmse4), fontsize=10)
axs[1].imshow(abs(img), cmap="gray")

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


# tt_solver['solver_tot']
# tt_solver['loop']
#%%
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
