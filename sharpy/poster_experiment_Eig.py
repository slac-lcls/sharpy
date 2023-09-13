# -*- coding: utf-8 -*-
import numpy as np
import copy
import h5py
import matplotlib.pyplot as plt
from timeit import default_timer as timer

#from Operators import Split_Overlap_plan
import Solvers
from Operators import get_times, reset_times, normalize_times
import config
from Operators import Gramiam_plan, Replicate_frame
from wrap_ops import overlap_cuda,split_cuda

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

if GPU:
    Alternating_projections = Solvers.Alternating_projections_c
else:
    Alternating_projections = Solvers.Alternating_projections
##################################################
# input data


import sys
# Retrieve the value of 'fname' from the command-line arguments
#fname_in = sys.argv[1] if len(sys.argv) > 1 else None
#fname_in = "poster_64x64.h5"
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


#Split, Overlap = Split_Overlap_plan(translations_x, translations_y, nx, ny, Nx, Ny)

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
    if GPU:
       #calculate the preconditioner here
        Gplan = Gramiam_plan(translations_x,translations_y,nframes,nx,ny,Nx,Ny, bw = 0)
        
        #print('plan',Gplan['col'],Gplan['row'],Gplan['dx'],Gplan['dy'],Gplan['val'])
    else:
        Gplan = Gramiam_plan(translations_x,translations_y,nframes,nx,ny,Nx,Ny, bw = 0)
        
    if Gplan['col'].size == 0:
        sync = False
else: 
    Gplan = None



#img_initial = xp.ones((Nx, Ny), dtype=xp.complex64)
img_initial = xp.ones((Nx, Ny), dtype=xp.complex64) #Need to match datatype in operators fft Plan2D

############################
# reconstruct
refine_illumination = False
maxiter = 4000
# residuals_interval = np.inf
residuals_interval = 1

t0 = timer()
print("geometry: img size:", (Nx, Ny), "frames:", (nx, ny, nframes))
print(
    "not refining illumination, starting with good one, maxiter:",
    maxiter,
)


#############################
##compare sync and no sync
#############################

t0 = timer()
residuals_sync = []

t0 = timer()
img_nosync, frames0, illum0, residuals_nosync = Alternating_projections(
        False,
        img_initial + 0,
        Gplan,
        illumination,
        translations_x,
        translations_y,
        overlap_cuda,
        split_cuda,
        data,
        refine_illumination,
        maxiter,
        normalization=None,
        img_truth =truth,
        residuals_interval = residuals_interval,
        sync_interval = 1,
        num_iter = 100,
    )
print("No Sync total time:", timer() - t0)


for ii in np.array([1,10,30,50,100,200]):
    reset_times()
    
    t0 = timer()
    img2, frames2, illum2, residuals_AP2 = Alternating_projections(
    True,
    img_initial +0,
    Gplan,
    illumination ,
    translations_x ,
    translations_y ,
    overlap_cuda,
    split_cuda,
    data ,
    refine_illumination,
    maxiter,
    normalization=None,
    img_truth =truth,
    residuals_interval = residuals_interval,
    sync_interval = 1,
    num_iter = ii
    )
    
    print("Sync total time:",ii ,timer() - t0)
    residuals_sync.append(residuals_AP2)
    
    ##
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



############################
# calculate mse
############################
'''
nrm0 = xp.linalg.norm(truth)
if residuals_AP0.size > 0:
    nmse4_0 = residuals_AP0[-1, 0] #nmse for no sync
    nmse4_1 = residuals_AP1[-1, 0] #nmse for sync
else:
    nmse4 = np.NaN


if GPU:
    truth = truth.get()
    img0 = img0.get() #reconstruction for no sync
    img1 = img1.get() #reconstruction for nyc
    residuals_AP0 = residuals_AP0.get() #convergence for no sync
    residuals_AP1 = residuals_AP1.get() #convergence for sync
else:
    img = img4

## Reconstruction plot

fig = plt.figure(figsize=(10, 14))
plt.subplot(1, 3, 1)
plt.axis('off')
plt.title("Ground truth", fontsize=10)
plt.imshow(abs(truth), cmap="gray")
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(1, 3, 2)
plt.axis('off')
plt.title("Alternating Projections\n No Sync:%2.2g" % (nmse4_0), fontsize=10)
plt.imshow(abs(img0), cmap="gray")
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(1, 3, 3)
plt.axis('off')
#plt.title("Difference")
#plt.imshow(abs(truth) - abs(img), cmap="jet")
plt.title("Alternating Projections\n Sync:%2.2g" % (nmse4_1), fontsize=10)
plt.imshow(abs(img1), cmap="gray")
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()
'''

# make a new figure with residuals
nn = np.array([1,10,30,50,100,200])
fig,axs = plt.subplots(nrows = 1, ncols = 3, sharex = True, figsize=(30, 10))
for i in range(len(nn)):
    axs[0].semilogy((residuals_sync[i][:, 0]).get(), label = {'num_power_it',nn[i]})
    axs[0].set_title("|img-f truth| (f=phase scalar)")
    
    axs[1].semilogy((residuals_sync[i][:, 1]).get(), label = {'num_power_it',nn[i]})
    axs[1].set_title("||frames|-data|")
    
    axs[2].semilogy((residuals_sync[i][:, 2]).get(), label = {'num_power_it',nn[i]})
    axs[2].set_title("frames overlapped")


axs[0].semilogy((residuals_nosync[:,0]).get(), '--',label = {'no sync'})
axs[0].legend()
axs[1].semilogy((residuals_nosync[:,1]).get(), '--',label = {'no sync'})
axs[1].legend()
axs[2].semilogy((residuals_nosync[:,2]).get(), '--',label = {'no sync'})
axs[2].legend()



plt.show()
