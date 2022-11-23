# -*- coding: utf-8 -*-
import numpy as np
import h5py

from Operators import  Split_plan 
from Operators import Illuminate_frames 
from Operators import cropmat, make_probe, make_translations #, map_frames

# define simulation dimensions (frames, step, image)
nx=128 # frame size
Dx=5 # Step size
nnx=16 # number of frames in x direction
Nx = Dx*nnx

# same thing for y
ny=nx 
nny=nnx;
Dy=Dx;
nframes=nnx*nny; #total number of frames

Ny = Dy*nny

# we need: image, illumination, scan pattern

# make the illumination (astigmatic i.e. fx, fy different from 0)
illumination=make_probe(nx,ny, r1 = .03 , r2 = 0.06, fx = +20, fy = -20)
#############################
# create translations (in pixels) using close packing
translations_x,translations_y=make_translations(Dx,Dy,nnx,nny,Nx,Ny)


# load an image with Pillow and make it complex
from PIL import Image
img0=np.array(Image.open('data/gold_balls.png'),np.float32)/63.
# img0=img0+1j # simple complex
img0=np.exp(.69*(-1+.5*1j)*img0) # more realistic

# set image dimension:
Nx = np.int(np.ceil(np.max(translations_x)-np.min(translations_x)))
Ny = Nx

truth=cropmat(img0,[Nx,Ny])

# threshold to check if things match within numerical accuracy
thres = np.finfo(truth.dtype).eps *1e2




##################
Split = Split_plan(translations_x,translations_y,nx,ny,Nx,Ny)
# generate frames from truth:
frames=Illuminate_frames(Split(truth),illumination) #check

## keep the data fftshifted
frames_data = np.abs(np.fft.fft2(frames))**2 #squared magnitude from the truth

lens = np.fft.fft2(illumination)
lens_aperture = np.abs(lens)/np.max(np.abs(lens)) >.1


# make up some Physical numbers for the experiment:
resolution = 1. # desired size of 1 pixels
wavelength = .1 # 
detector_pixel_size = 100. 

# this defines the detector distance (paraxial approx)
detector_distance = (nx * detector_pixel_size *  resolution) / wavelength 

# save data to file:
file_out = 'simulation.h5'
fid = h5py.File(file_out, 'w')
fid.create_dataset('truth', data=truth)
fid.create_dataset('data', data=frames_data)
fid.create_dataset('probe', data=illumination)
fid.create_dataset('lens_aperture', data=lens_aperture)

    

# use complex for 2D translations
translations = (translations_x+1j*translations_y)*resolution
fid.create_dataset('translations', data=translations)
fid.create_dataset('wavelength', data=wavelength)
fid.create_dataset('detector_pixel_size', data=detector_pixel_size)
fid.create_dataset('detector_distance', data=detector_distance)

fid.close()


