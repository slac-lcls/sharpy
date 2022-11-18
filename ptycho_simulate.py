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

# make the illumination
illumination=make_probe(nx,ny)
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


# make up some Physical numbers for the experiment:
resolution = 1. # desired size of 1 pixels
wavelength = .1 # 
detector_pixel_size = 100. 
# this defines the detector distance
detector_distance = (nx * detector_pixel_size *  resolution) / wavelength 


# save data to file:
frames_data = np.abs(np.fft.fft2(frames))**2 #squared magnitude from the truth
file_out = 'simulation.h5'
fid = h5py.File(file_out, 'w')
fid.create_dataset('truth', data=truth)
fid.create_dataset('data', data=frames_data)
fid.create_dataset('probe', data=illumination)


# use complex for 2D
translations = (translations_x+1j*translations_y)*resolution
fid.create_dataset('translations', data=translations)
fid.create_dataset('wavelength', data=wavelength)
fid.create_dataset('detector_pixel_size', data=detector_pixel_size)
fid.create_dataset('detector_distance', data=detector_distance)

fid.close()


##### ----------------------------------------
# check if overlap is self-consistent with the split
test_flag = True
if test_flag:
    from Operators import  Overlap_plan, Replicate_frame, mse_calc #, frames_overlap, Stack_frames
    
    # used to check if things are self consistent
    Overlap = Overlap_plan(translations_x,translations_y,nx,ny,Nx,Ny)
    normalization=Overlap(Replicate_frame(np.abs(illumination)**2,nframes)) #check
    # recover the image from true frames, true illumination:
    img=Overlap(Illuminate_frames(frames,np.conj(illumination)))/normalization
    nmse0=mse_calc(truth,img)/np.linalg.norm(truth)
    # thres = np.finfo(img.dtype).eps *1e2
    good_enough = nmse0 < thres
    print('normalized mean square error is:', nmse0, '. Threshold:', thres, "; good enough:", good_enough)
##### ----------------------------------------

#%%
test_synch = True
if test_synch:
    print('testing synch!!!!!!!!!!!!')
    from Operators import  Overlap_plan, Replicate_frame, mse_calc #, frames_overlap, Stack_frames
    from Operators import Gramiam_plan,  synchronize_frames_c
    bw = 0
    Gplan = Gramiam_plan(translations_x,translations_y,nframes,nx,ny,Nx,Ny,bw) 
    
    #randomize framewise phases
    phases=np.exp(1j*np.random.random((nframes,1,1))*2*np.pi)

    frames_rand=frames*phases

    img2=Overlap(Illuminate_frames(frames_rand,np.conj(illumination)))/normalization
    nmse2ns = mse_calc(truth,img2)/np.linalg.norm(truth)
    # good_enough2 = nmse2 < thres
    # print('no-synch, normalized mean square error is:', nmse2ns) #, '. Threshold:', thres, "; good enough:", good_enough2 )


    frames_norm=np.linalg.norm(frames_rand,axis=(1,2))
    inormalization_split = Split(1/normalization)
    omega=synchronize_frames_c(frames_rand, illumination, inormalization_split, Gplan)
    
    #synchronize frames
    frames_sync=frames_rand*omega


    img2=Overlap(Illuminate_frames(frames_sync,np.conj(illumination)))/normalization
    nmse2 = mse_calc(truth,img2)/np.linalg.norm(truth)

    good_enough2 = nmse2 < thres
    print('with synch, normalized mean square error is:', nmse2, 'without', nmse2ns, '. Threshold:', thres, "; good enough:", good_enough2 )

    
    # # used to check if things are self consistent
    # Overlap = Overlap_plan(translations_x,translations_y,nx,ny,Nx,Ny)
    # normalization=Overlap(Replicate_frame(np.abs(illumination)**2,nframes)) #check



    pass


