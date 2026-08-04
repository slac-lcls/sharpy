# -*- coding: utf-8 -*-
#!/cds/home/y/yn754/anaconda3/envs/sharpy-env/bin/python

import numpy as np
import cupy as cp
import h5py

from Operators import Split_plan 
from wrap_ops import split_cuda
from Operators import Illuminate_frames
from Operators import cropmat, make_probe, make_translations  # , map_frames
import config

GPU = config.GPU


# define simulation dimensions (frames, step, image)
#nx = 8  # frame size
#Dx = 5  # Step size
#nnx= 2 # number of frames in x direction

# define larger simulation dimensions (frames, step, image)
#nx = 64  # frame size
#Dx = 16  # Step size
#nnx=32 # number of frames in x direction

# nnx=40 # number of frames in x direction
#nnx = 80  # number of frames in x direction
# nnx=40 # number of frames in x direction

# nnx=64 # number of frames in x direction
#nnx = 100  # number of

# whynotworking
#nx = 128  # frame size
#Dx = 20  # Step size
#nnx=32 # number of frames in x direction
nx = 16
nnx = 16
Dx = 5

# same thing for y
ny = nx
nny = nnx
Dy = Dx
nframes = nnx * nny
# total number of frames

Nx = nnx * Dx
Ny = nny * Dy

# we need: image, illumination, scan pattern

# make the illumination (astigmatic i.e. fx, fy different from 0)
#illumination = make_probe(nx, ny, r1=0.03, r2=0.06, fx=+20, fy=-20)
illumination,lens_mask = make_probe(nx, ny, r1=0.025*3, r2=0.085*3, fx=+20, fy=-20) #old
#illumination,lens_mask = make_probe(nx, ny, r1=0.02*3, r2=0.06*3, fx=+20, fy=-20) #refine_illum #complex128
#illumination += illumination

# Set a random seed for reproducibility
np.random.seed(42)

# Generate Gaussian random noise
mean = 0  # Center the noise around 0.5
std_dev = 0.01  # Standard deviation
size = (nx, ny)  # Size of the noise array

# Generate noise and clip it to the range [0, 1]
noise = cp.asarray(np.clip(np.random.normal(mean, std_dev, size), 0, 1))

illumination += noise

#############################
# create translations (in pixels) using close packing
translations_x, translations_y = make_translations(Dx, Dy, nnx, nny, Nx, Ny)
print('type0',type(translations_x),translations_x.dtype)

# load an image with Pillow and make it complex
from PIL import Image

img0 = np.array(Image.open("../data/gold_balls.png"), np.float32) / 63.0
#img0=img0+1j # simple complex
#img0 = np.exp(0.69 * (-1 + 0.5 * 1j) * img0)  # more realistic
img0 = np.exp(0.69 * (-1 + 0.5 * 1j) * img0)  # increase for more contrast

# set image dimension:
Nx = int(np.ceil(np.max(translations_x) - np.min(translations_x)))
#why not Nx +=  Nx + nx
Ny = Nx

from time import sleep


if GPU:
    import cupy as xp

    img0 = xp.array(img0, dtype=xp.complex64)
    illumination = xp.array(illumination, dtype=xp.complex64)

truth = cp.ascontiguousarray(cropmat(img0, [Nx, Ny]))
#print(type(truth))

# threshold to check if things match within numerical accuracy
thres = np.finfo(truth.dtype).eps * 1e2


##################
Split = Split_plan(translations_x, translations_y, nx, ny, Nx, Ny)

# generate frames from truth:
  
frames0 = Illuminate_frames(Split(truth), illumination)  # check
frames = xp.zeros((nframes,nx,ny),dtype = xp.complex64)

translations = (translations_x + 1j * translations_y).astype(xp.complex64)

frames_out = xp.zeros(frames.shape,dtype = xp.complex64)
split_cuda(truth, frames_out, translations, illumination)  

#print('!!!',frames_out.shape)
frames = xp.reshape(frames_out,(frames.shape))
#print(xp.linalg.norm(frames-frames0))
## keep the data fftshifted
frames_data = np.abs(np.fft.fft2(frames)) ** 2  # squared magnitude from the truth

lens = np.fft.fft2(illumination)
lens_aperture = np.abs(lens) / np.max(np.abs(lens)) > 0.1


# make up some Physical numbers for the experiment:
resolution = 1.0  # desired size of 1 pixels
wavelength = 0.1  #
detector_pixel_size = 100.0


# this defines the detector distance (paraxial approx)
detector_distance = (nx * detector_pixel_size * resolution) / wavelength
# save data to file:
file_out = "refine_illum.h5"
fid = h5py.File(file_out, "w")


if GPU:
    truth = truth.get()
    frames_data = frames_data.get()
    illumination = illumination.get() #complex64
    lens_mask = lens_mask.get()
    lens_aperture = lens_aperture.get()
    translations_x = translations_x.get()
    translations_y = translations_y.get()
    

fid.create_dataset("truth", data=truth)
fid.create_dataset("data", data=frames_data)
fid.create_dataset("probe", data=illumination)
fid.create_dataset("lens_aperture", data=lens_aperture)
fid.create_dataset("lens_mask",data = lens_mask )

# use complex for 2D translations
translations = (translations_x + 1j * translations_y) * resolution
fid.create_dataset("translations", data=translations)
fid.create_dataset("translations_x",data=translations_x)
fid.create_dataset("translations_y",data=translations_y)

fid.create_dataset("wavelength", data=wavelength)
fid.create_dataset("detector_pixel_size", data=detector_pixel_size)
fid.create_dataset("detector_distance", data=detector_distance)


fid.close()
