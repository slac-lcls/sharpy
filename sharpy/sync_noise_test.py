#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 20:47:00 2023

@author: yuan
"""

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

if GPU:
    import cupy as cp
    xp = cp
    print("using GPU")
else:
    xp = np
    print("using CPU")


##################################################
# input data
fname_in = "simulation.h5"
fid = h5py.File(fname_in, "r")

#data = xp.array(fid["data"], dtype=xp.float32) 
data = xp.array(fid["data"], dtype=xp.float64) 
data0 = copy.deepcopy(data) #make a copy of the noiseless data
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
Nx = xp.int(xp.ceil(xp.max(translations_x) - xp.min(translations_x)))
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
   Gplan = Gramiam_plan(translations_x,translations_y,nframes,nx,ny,Nx,Ny)

else: 
   Gplan = None



img_initial = xp.ones((Nx, Ny), dtype=xp.complex64)
############################
# reconstruct
refine_illumination = False
maxiter = 100
# residuals_interval = np.inf
residuals_interval = 1

t0 = timer()
print("geometry: img size:", (Nx, Ny), "frames:", (nx, ny, nframes))
print(
    "not refining illumination, starting with good one, maxiter:",
    maxiter,
)

nrm0 = xp.linalg.norm(truth)
test_result = {'SNR':[],'img':[],'frames':[],'illum':[],'residuals_AP':[],'nmse':[]}
for nl in range(-3,2):
    noise = 10**nl * xp.random.poisson(1, xp.shape(data0))
    data_in = data0 + noise
    img4, frames, illum, residuals_AP = Alternating_projections(
    sync,
    img_initial,
    Gplan,
    illumination,
    Overlap,
    Split,
    data_in,
    refine_illumination,
    maxiter,
    normalization=None,
    img_truth =truth,
    residuals_interval = residuals_interval
    )
    print("total time:", timer() - t0)
    
    # calculate mse
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

    test_result['SNR'].append(xp.linalg.norm(truth.ravel()) / xp.linalg.norm(noise.ravel()))
    test_result['img'].append(img)
    test_result['frames'].append(frames)
    test_result['illum'].append(illum)
    test_result['residuals_AP'].append(residuals_AP)
    test_result['nmse'].append(nmse4)
    
############################

#make a reconstruction plot
#fig = plt.figure(figsize=(30, 10),dpi=1200)
fig = plt.figure(figsize=(30, 10))
nn = len(test_result['SNR'])
for i in range(nn):
    plt.subplot(2, nn, i+1)
    plt.title("AP Sync \n SNR MSE : (% 2.2g ,% 2.2g)" % (test_result['SNR'][i],test_result['nmse'][i]), fontsize=20)
    plt.imshow(abs(test_result['img'][i]), cmap="gray")
    plt.subplot(2, nn, nn + i +1)
    plt.title("Difference", fontsize=20)
    plt.imshow(abs(truth) - abs(test_result['img'][i]), cmap="jet")
    plt.colorbar(location="bottom")
    
plt.show()

##

# make a new figure with residuals
#fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10, 10),dpi=1200)
fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10, 10))
for i in range(nn) :
    axs[0].semilogy(xp.asarray(test_result['residuals_AP'][i][:, 0]),label = {'SNR',round(test_result['SNR'][i],2)})
    axs[0].set_title("|img-f truth| (f=phase scalar)")
    axs[0].legend(loc = 'upper right')
    axs[1].semilogy(test_result['residuals_AP'][i][:, 1])
    axs[1].set_title("||frames|-data|")
    axs[2].semilogy(test_result['residuals_AP'][i][:, 2])
    axs[2].set_title("frames overlapped")
plt.show()

