#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 21:03:21 2023

@author: yuan
"""
#test on the synchronization without AP

import numpy as np
import h5py
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from Operators import Split_Overlap_plan, Illuminate_frames
from Operators import get_times, reset_times, normalize_times
import config
from Operators import Gramiam_plan, Replicate_frame, synchronize_frames_c
from Operators import common_scale

reset_times()
GPU = config.GPU #False

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
#get translations
translations_x = xp.real(translations)
translations_y = xp.imag(translations)

# get the image extent (with wrap-around)
Nx = xp.int(xp.ceil(xp.max(translations_x) - xp.min(translations_x)))
Ny = Nx

#define operators
Split, Overlap = Split_Overlap_plan(translations_x, translations_y, nx, ny, Nx, Ny)
Gplan = Gramiam_plan(translations_x,translations_y,nframes,nx,ny,Nx,Ny)

#
frames=Illuminate_frames(Split(truth),illumination) #check
    
# generate normalization
normalization=Overlap(Replicate_frame(np.abs(illumination)**2,nframes)) #check
    
#get back the original image
img=Overlap(Illuminate_frames(frames,np.conj(illumination)))/normalization
   
#randomize framewise phases
phases=np.exp(1j*np.random.random((nframes,1,1))*2*np.pi)
frames_rand=frames*phases
img1=Overlap(Illuminate_frames(frames_rand,np.conj(illumination)))/normalization
   
#phase sync for rand frame
inormalization_split = Split(1/normalization)
frames_norm=np.linalg.norm(frames,axis=(1,2))

import scipy as sp
Gplan['Preconditioner']=sp.sparse.diags(1/frames_norm)
omega=synchronize_frames_c(frames_rand, illumination, inormalization_split, Gplan)
frames_sync=frames_rand*omega
img2=Overlap(Illuminate_frames(frames_sync,np.conj(illumination)))/normalization
    
#account for common scaling
img2 *= common_scale(img2,truth)
plt.imshow(np.abs(img2)) 
print('SYNC MSE:',np.linalg.norm(img2-truth)  ) 