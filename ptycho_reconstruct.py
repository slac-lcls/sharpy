# -*- coding: utf-8 -*-
import numpy as np
import h5py
import matplotlib.pyplot as plt
from Operators import Split_plan, Overlap_plan #, Project_data 
from Solvers import Alternating_projections
from Operators import get_times 


##################################################
# input data
fname_in = 'simulation.h5'
fid= h5py.File(fname_in, "r")

data=np.array(fid['data'], dtype = np.float32)
illumination=np.array(fid['probe'], dtype = np.complex64)
translations=np.array(fid['translations'])

nframes,nx,ny=data.shape
resolution = np.float32( fid['wavelength']) * np.float32(fid['detector_distance']) / (nx*  np.float32(fid['detector_pixel_size'] ))

translations=translations/resolution

truth=np.array(fid['truth'], dtype = np.complex64)
fid.close()

#################################################

translations_x = np.real(translations)
translations_y = np.imag(translations)

# get the image extent (with wrap-around)
# Nx,Ny = truth.shape
Nx = np.int(np.ceil(np.max(translations_x)-np.min(translations_x)))
Ny = Nx


#%%

Split = Split_plan(translations_x,translations_y,nx,ny,Nx,Ny)
Overlap = Overlap_plan(translations_x,translations_y,nx,ny,Nx,Ny)
 
img_initial = np.ones((Nx,Ny))
############################
# reconstruct

maxiter=100
img4,frames, residuals_AP = Alternating_projections(img_initial, illumination, Overlap, Split, data, maxiter = maxiter, img_truth = truth)

############################


## print results
timers=get_times()
print(timers)

#calculate mse
nrm0=np.linalg.norm(truth)
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

