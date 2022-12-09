#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 09:35:48 2022

@author: smarchesini
"""

import numpy as np
import h5py

from Operators import Split_plan, Overlap_plan  # , Project_data
from Operators import Illuminate_frames

# from Operators import cropmat, make_probe, make_translations #, map_frames


##################################################
# input data
fname_in = "simulation.h5"
fid = h5py.File(fname_in, "r")

data = np.array(fid["data"], dtype=np.float32)
illumination = np.array(fid["probe"], dtype=np.complex64)
translations = np.array(fid["translations"])

nframes, nx, ny = data.shape
resolution = (
    np.float32(fid["wavelength"])
    * np.float32(fid["detector_distance"])
    / (nx * np.float32(fid["detector_pixel_size"]))
)

translations = translations / resolution

truth = np.array(fid["truth"], dtype=np.complex64)
fid.close()

#################################################

translations_x = np.real(translations)
translations_y = np.imag(translations)

# get the image extent (with wrap-around)
# Nx,Ny = truth.shape
Nx = np.int(np.ceil(np.max(translations_x) - np.min(translations_x)))
Ny = Nx


#%%

Split = Split_plan(translations_x, translations_y, nx, ny, Nx, Ny)
Overlap = Overlap_plan(translations_x, translations_y, nx, ny, Nx, Ny)

frames = Illuminate_frames(Split(truth), illumination)  # check

# thres = np.finfo(frames.dtype).eps *1e2

##### ----------------------------------------
# check if overlap is self-consistent with the split
test_flag = True
if test_flag:
    from Operators import (
        Overlap_plan,
        Replicate_frame,
        mse_calc,
    )  # , frames_overlap, Stack_frames

    # used to check if things are self consistent
    Overlap = Overlap_plan(translations_x, translations_y, nx, ny, Nx, Ny)
    normalization = Overlap(
        Replicate_frame(np.abs(illumination) ** 2, nframes)
    )  # check
    # recover the image from true frames, true illumination:
    img = Overlap(Illuminate_frames(frames, np.conj(illumination))) / normalization
    nmse0 = mse_calc(truth, img) / np.linalg.norm(truth)
    thres = np.finfo(img.dtype).eps * 1e2
    good_enough = nmse0 < thres
    print(
        "normalized mean square error is:",
        nmse0,
        ". Threshold:",
        thres,
        "; good enough:",
        good_enough,
    )
##### ----------------------------------------

#%%
test_synch = True
if test_synch:
    print("testing synch!!!!!!!!!!!!")
    from Operators import (
        Overlap_plan,
        Replicate_frame,
        mse_calc,
    )  # , frames_overlap, Stack_frames
    from Operators import Gramiam_plan, synchronize_frames_c

    bw = 0
    Gplan = Gramiam_plan(translations_x, translations_y, nframes, nx, ny, Nx, Ny, bw)

    # randomize framewise phases
    phases = np.exp(1j * np.random.random((nframes, 1, 1)) * 2 * np.pi)

    frames_rand = frames * phases

    img2 = (
        Overlap(Illuminate_frames(frames_rand, np.conj(illumination))) / normalization
    )
    nmse2ns = mse_calc(truth, img2) / np.linalg.norm(truth)
    # good_enough2 = nmse2 < thres
    # print('no-synch, normalized mean square error is:', nmse2ns) #, '. Threshold:', thres, "; good enough:", good_enough2 )

    frames_norm = np.linalg.norm(frames_rand, axis=(1, 2))
    inormalization_split = Split(1 / normalization)
    omega = synchronize_frames_c(frames_rand, illumination, inormalization_split, Gplan)

    # synchronize frames
    frames_sync = frames_rand * omega

    img2 = (
        Overlap(Illuminate_frames(frames_sync, np.conj(illumination))) / normalization
    )
    nmse2 = mse_calc(truth, img2) / np.linalg.norm(truth)

    good_enough2 = nmse2 < thres
    print(
        "with synch, normalized mean square error is:",
        nmse2,
        "without",
        nmse2ns,
        ". Threshold:",
        thres,
        "; good enough:",
        good_enough2,
    )

    # # used to check if things are self consistent
    # Overlap = Overlap_plan(translations_x,translations_y,nx,ny,Nx,Ny)
    # normalization=Overlap(Replicate_frame(np.abs(illumination)**2,nframes)) #check

    pass
