# -*- coding: utf-8 -*-
import numpy as np
import cupy as cp

xp=cp


import h5py
import matplotlib.pyplot as plt
# from Operators import Split_plan, Overlap_plan #, Project_data 
from Operators import Split_Overlap_plan #, Project_data 
from Solvers import Alternating_projections
from Operators import get_times 


##################################################
# input data
fname_in = 'simulation.h5'
fid= h5py.File(fname_in, "r")

data=xp.array(fid['data'], dtype = xp.float32)
illumination=xp.array(fid['probe'], dtype = xp.complex64)
translations=xp.array(fid['translations'])

nframes,nx,ny=data.shape
resolution = xp.float32( fid['wavelength']) * xp.float32(fid['detector_distance']) / (nx*  xp.float32(fid['detector_pixel_size'] ))

translations=translations/resolution

truth=xp.array(fid['truth'], dtype = xp.complex64)
fid.close()

#################################################

translations_x = xp.real(translations)
translations_y = xp.imag(translations)

# get the image extent (with wrap-around)
# Nx,Ny = truth.shape
Nx = xp.int(xp.ceil(xp.max(translations_x)-xp.min(translations_x)))
Ny = Nx


#%%
# import cupyx.scipy.sparse as sparse
# from Operators import map_frames

# mapid=map_frames(translations_x,translations_y,nx,ny,Nx,Ny)
# col = xp.arange(mapid.size)
# val = xp.ones((mapid.size),dtype=np.float32)
# SS=sparse.coo_matrix((val.ravel(),(mapid.ravel(),col.ravel())))
# SST=sparse.coo_matrix((val.ravel(),(col.ravel(),mapid.ravel())))
# SS=sparse.csr_matrix(SS)
# SST=sparse.csr_matrix(SST)

# def Overlap(frames,SS, shape):
#   output = SS*frames.ravel()
#   output.shape=shape
#   return output
# def Split(img,SST, shape):
#   output = SST*img.ravel()
#   output.shape=shape
#   return output
  
#%%

Split, Overlap = Split_Overlap_plan(translations_x,translations_y,nx,ny,Nx,Ny)

#Split = Split_plan(translations_x,translations_y,nx,ny,Nx,Ny)
#Overlap = Overlap_plan(translations_x,translations_y,nx,ny,Nx,Ny)
 
img_initial = xp.ones((Nx,Ny))
############################
# reconstruct

maxiter=100
img4,frames, residuals_AP = Alternating_projections(img_initial, illumination, Overlap, Split, data, refine_illumination = False, maxiter = maxiter, img_truth = truth)

#%%


lens_true=xp.fft.fft2(xp.fft.fftshift(illumination))
lens_init=xp.abs(lens_true) # remove true phase
illum_init=xp.fft.fftshift(xp.fft.ifft2(lens_init))


print('not refining illumination, starting with the wrong one')
img3,frames, residuals_AP3 = Alternating_projections(img_initial, illum_init, Overlap, Split, data, refine_illumination = False, maxiter = maxiter, img_truth = truth)

print('refining illumination')
img2,frames, residuals_AP2 = Alternating_projections(img_initial, illum_init, Overlap, Split, data, refine_illumination = True, maxiter = maxiter, img_truth = truth)


############################


## print results
timers=get_times()
print(timers)

#calculate mse
nrm0=xp.linalg.norm(truth)
# nmse4=mse_calc(truth,img4)/nrm0
nmse4=residuals_AP[maxiter-1,0]

fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True,figsize=(10,10))

axs[0].set_title('Truth',fontsize=10)
axs[0].imshow(abs(truth), cmap = 'gray')
axs[1].set_title('Alternating Projections no Sync:%2.2g' %( nmse4),fontsize=10)
axs[1].imshow(abs(img4),cmap = 'gray')

plt.show()

##

# make a new figure with residuals
fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True,figsize=(10,10))
axs[0].semilogy(residuals_AP[:,0])
axs[0].set_title('|img-f truth| (f=phase scalar)')
axs[1].semilogy(residuals_AP[:,1])
axs[1].set_title('||frames|-data|')

axs[2].semilogy(residuals_AP[:,2])
axs[2].set_title('frames overlapped')

