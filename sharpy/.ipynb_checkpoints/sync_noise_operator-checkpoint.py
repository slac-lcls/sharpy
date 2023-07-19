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
from Operators import Gramiam_plan, Replicate_frame, circular_aperture, mse_calc

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
class DataProcessor:
    def __init__(self,fname_in):
        self.fname_in = fname_in
        
    def load_data(self):
        fid = h5py.File(self.fname_in, "r")
        #data = xp.array(fid["data"], dtype=xp.float32) 
        self.data = xp.array(fid["data"], dtype=xp.float64) 
        self.data0 = copy.deepcopy(self.data) #make a copy of the noiseless data
        self.illumination = xp.array(fid["probe"], dtype=xp.complex64)

        self.translations = xp.array(fid["translations"])

        self.nframes, self.nx, self.ny = self.data.shape
        self.resolution = (xp.float32(fid["wavelength"])
                      * xp.float32(fid["detector_distance"])
                      / (self.nx * xp.float32(fid["detector_pixel_size"]))
                     )

        self.translations = self.translations / self.resolution

        self.truth = xp.array(fid["truth"], dtype=xp.complex64)
        
        self.translations_x = xp.real(self.translations)
        self.translations_y = xp.imag(self.translations)

        # get the image extent (with wrap-around)
        self.Nx = xp.int(xp.ceil(xp.max(self.translations_x) - xp.min(self.translations_x)))
        self.Ny = self.Nx
        fid.close()
#################################################

def Get_Mappings(processor):
    Split, Overlap = Split_Overlap_plan(processor.translations_x, processor.translations_y, processor.nx,
                                        processor.ny, processor.Nx,processor.Ny)

    if sync == True:
       Gplan = Gramiam_plan(processor.translations_x,processor.translations_y,processor.nframes,processor.nx,processor.ny,processor.Nx,processor.Ny)

    else: 
       Gplan = None
    return Split,Overlap, Gplan


############################
# reconstruct

def Simulate_Noise(opts,sync,processor):
    t0 = timer()
    
    ###
    refine_illumination, maxiter, residuals_interval = opts['refine_illumination'], opts['maxiter'], opts['residuals_interval']
    data,data0,illumination,translations, nframes, nx, ny, resolution, truth, translations_x, translations_y, Nx, Ny= processor.data, processor.data0, processor.illumination, processor.translations, processor.nframes, processor.nx, processor.ny, processor.resolution, processor.truth, processor.translations_x, processor.translations_y, processor.Nx, processor.Ny
    ###
    
    print("geometry: img size:", (Nx, Ny), "frames:", (nx, ny, nframes))
    print( "not refining illumination, starting with good one, maxiter:", maxiter,)

    ###
    img_initial = xp.ones((Nx, Ny))
    nrm0 = xp.linalg.norm(truth)
    test_result = {'SNR':[],'img':[],'frames':[],'illum':[],'residuals_AP':[],'nmse':[]}
    Split,Overlap, Gplan = Get_Mappings(processor)
    ###
    
    noise_list = noise_generator(opts['noise_low'], opts['noise_high'], xp.shape(data0),opts['noise_type'])
    for nl in xp.arange(len(noise_list)):
        noise = noise_list[nl]
        data_in = data0 + noise
        #data_in = (xp.sqrt(data0) + noise)**2 #gaussian noise
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
    return test_result, img
   
############################
def noise_generator(noise_low,noise_high,length,noise_type):
    noise = []
    level = xp.arange(noise_low,noise_high,0.2)
    for nl in level:
        if noise_type == 'poisson':
            noise.append(10**nl * xp.random.poisson(1, length))   
        elif noise_type == 'gaussian':
            noise.append(10**nl * xp.random.randn(length))
    return noise
############################

def plotter(test_result0, test_result1, img0, image1,processor):
    #make a reconstruction plot
    #fig = plt.figure(figsize=(30, 10),dpi=1200)
    fig = plt.figure(figsize=(50, 20))
    nn = len(test_result0['SNR'])
    for i in range(nn):
        plt.subplot(4, nn, i+1)
        plt.title("AP Sync \n SNR MSE : (% 2.2g ,% 2.2g)" % (test_result1['SNR'][i],test_result1['nmse'][i]), fontsize=20)
        plt.imshow(abs(test_result1['img'][i]), cmap="gray")
        plt.subplot(4, nn, nn + i +1)
        plt.title("Difference", fontsize=20)
        plt.imshow(abs(processor.truth) - abs(test_result1['img'][i]), cmap="jet")
        plt.colorbar(location="bottom")
        
        plt.subplot(4, nn, 2*nn + i+1)
        plt.title("AP NoSync \n SNR MSE : (% 2.2g ,% 2.2g)" % (test_result0['SNR'][i],test_result0['nmse'][i]), fontsize=20)
        plt.imshow(abs(test_result0['img'][i]), cmap="gray")
        plt.subplot(4, nn, 3* nn + i +1)
        plt.title("Difference", fontsize=20)
        plt.imshow(abs(processor.truth) - abs(test_result0['img'][i]), cmap="jet")
        plt.colorbar(location="bottom")
    
    plt.show()

##

    # make a new figure with residuals
    #fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10, 10),dpi=1200)
    fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10, 20))
    for i in range(nn) :
        axs[0].semilogy(xp.asarray(test_result0['residuals_AP'][i][:, 0]),label = {'SNR',round(test_result0['SNR'][i],2)})
        axs[0].semilogy(xp.asarray(test_result1['residuals_AP'][i][:, 0]),'--',label = {'SNR',round(test_result1['SNR'][i],2)})
        axs[0].set_title("|img-f truth| (f=phase scalar)")
        axs[0].legend(loc = 'upper right')
        axs[1].semilogy(xp.asarray(test_result0['residuals_AP'][i][:, 3]))
        axs[1].semilogy(xp.asarray(test_result1['residuals_AP'][i][:, 3]),'--')
        axs[1].set_title("|truth|/|img-f truth| (f=phase scalar)")
        axs[2].semilogy(test_result0['residuals_AP'][i][:, 1])
        axs[2].semilogy(test_result1['residuals_AP'][i][:, 1],'--')
        axs[2].set_title("||frames|-data|")
        axs[3].semilogy(test_result0['residuals_AP'][i][:, 2])
        axs[3].semilogy(test_result1['residuals_AP'][i][:, 2],'--')
        axs[3].set_title("frames overlapped")
    plt.show()

def MSE_aperture(test_result0, test_result1,processor,radius = None):
  
    if radius != None:
        aperture = circular_aperture(radius, processor.truth)
        truth_low = xp.fft.ifft2(aperture * xp.fft.fft2(processor.truth))
    else:
        truth_low = processor.truth
      
    nmse0 = []
    nmse1 = []
    for i in range(len(test_result0['SNR'])):
        if radius != None:
            img0_low = xp.fft.ifft2(aperture * xp.fft.fft2(test_result0['img'][i]))
            img1_low = xp.fft.ifft2(aperture * xp.fft.fft2(test_result1['img'][i]))
            nmse0.append(mse_calc(truth_low, img0_low))
            nmse1.append(mse_calc(truth_low, img1_low))
        else:
            nmse0.append(mse_calc(processor.truth, test_result0['img'][i]))
            nmse1.append(mse_calc(processor.truth, test_result1['img'][i]))
    
    return nmse0, nmse1, truth_low

def plot_MSE_aperture(nmse0, nmse1,test_result0):
    fig, axs = plt.subplots(1,1,sharex=True, figsize=(10, 20))
    
    axs.semilogy(xp.asarray(test_result0['SNR']),xp.asarray(nmse0),label = {'SNR',round(test_result0['SNR'],2)})     
    axs.semilogy(xp.asarray(test_result0['SNR']),xp.asarray(nmse1),'--')
    axs.set_title("Comparison of MSE for Sync and NoSync")
    axs.legend(loc = 'upper right')   
    axs.set_xlabel('SNR')
    axs.grid(True)
    plt.show()
    