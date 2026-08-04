# -*- coding: utf-8 -*-
import numpy as np
import copy
import h5py
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from Operators import Split_Overlap_plan, make_probe,make_translations,cropmat
from Operators_Tile import  Tile_plan_c,map_tiles, Gplan_sub,Gplan_tile_c, Alternating_projections_tiles
import Solvers
from Operators import get_times, reset_times, normalize_times
import config
from Operators import Gramiam_plan, Replicate_frame
from wrap_ops import overlap_cuda,split_cuda
import  Operators_Tile

reset_times()
GPU = config.GPU #False
sync = True
tile = True

if GPU:
    import cupy as cp
    xp = cp
    print("using GPU")
else:
    xp = np
    print("using CPU")


# input data

import sys
# Retrieve the value of 'fname' from the command-line arguments
#fname_in = sys.argv[1] if len(sys.argv) > 1 else None

fid = h5py.File(fname_in, "r")

frames_data = xp.array(fid["data"], dtype=xp.float32)
illumination = xp.array(fid["probe"], dtype=xp.complex64)

translations = xp.array(fid["translations"])
#print('translations1', type(translations),translations.dtype) #float128 dtype
nframes, nx, ny = frames_data.shape

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

# get the image extent (with wrap-around)
Nx = int(xp.ceil(xp.max(translations_x) - xp.min(translations_x)))
Ny = Nx
NTx = 2 #number of Tiles in x direction
NTy = NTx

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
        mempool.used_bytes() / frames_data.nbytes,
        mempool.total_bytes() / frames_data.nbytes,
    )
    print("data size", frames_data.nbytes)
    print("----")

#get sync plan
if sync == True:
    if GPU:
        if tile == True:
            #Get the shift of the tiles and groupid
            Tiles_plan = Tile_plan_c(NTx,NTy,translations_x, translations_y, nx, ny, Nx, Ny)
            #mapid_tiles, tiles_sizes = map_tiles(translations_x, translations_y, nx, ny, Nx, Ny,NTx,NTy,Tiles_plan) #may not need mapid_tiles
            tiles_sizes = map_tiles(translations_x, translations_y, nx, ny, Nx, Ny,NTx,NTy,Tiles_plan) #no need for v2
            
            #Get the gramiam_plan for each tile
            groupid = Tiles_plan['groupid']
            Gplan_tiles = Gplan_sub(groupid,translations_x,translations_y,nx,ny,Nx,Ny,bw = 0)
            Gplan = Gramiam_plan(translations_x,translations_y,nframes,nx,ny,Nx,Ny, bw = 0)
            #print('col is col[mask] instead')
            
            #get the Gplan to sync between tiles
            translations_tx,translations_ty=xp.meshgrid(Tiles_plan['shift_Tx'][0:-1],Tiles_plan['shift_Ty'][0:-1]) #maybe incorporate in tile_plan_c
            translations_tx = translations_tx.ravel()
            translations_ty = translations_ty.ravel()
            Gplan_Tiles=Gplan_tile_c(translations_tx,translations_ty,NTx * NTy,tiles_sizes[:,0].astype(int),tiles_sizes[:,1].astype(int),Nx,Ny,btw=0) 
           


    else:
        Gplan = Gramiam_plan(translations_x,translations_y,nframes,nx,ny,Nx,Ny, bw = 0)
        
        if Gplan['col'].size == 0:
            sync = False
else: 
    Gplan = None

    
############################
# reconstruct
refine_illumination = False
maxiter = 20
residuals_interval = 1
img_initial = xp.ones((Nx, Ny), dtype=xp.complex64) #Need to match datatype in operators fft Plan2D
#img_initial = truth +0

# threshold to check if things match within numerical accuracy
thres = np.finfo(truth.dtype).eps * 1e2



t0 = timer()
print("geometry: img size:", (Nx, Ny), "frames:", (nx, ny, nframes),"tiles:",(NTx,NTy))
print(
    "not refining illumination, starting with good one, maxiter:",
    maxiter,
)



img4, frames, illum, residuals_AP = Alternating_projections_tiles(
    sync,
    img_initial,
    NTx,
    NTy,
    Gplan_tiles,
    Gplan_Tiles,
    Gplan,
    groupid,
    illumination,
    translations_x,
    translations_y,
    translations_tx,
    translations_ty,
    overlap_cuda,
    split_cuda,
    frames_data,
    refine_illumination,
    maxiter,
    normalization = None,
    img_truth = truth,
    residuals_interval = 1,
    num_iter = 10)
    

print("total time:", timer() - t0)

############################


## print results
timers = get_times()


tt_ops = copy.deepcopy(timers)
#tt_solver = copy.deepcopy(Solvers.get_times())
tt_solver = copy.deepcopy(Operators_Tile.get_times())

print(timers)


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
