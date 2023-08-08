import numpy as np
from timeit import default_timer as timer
from Operators import (
    Illuminate_frames,
    Project_data,
    synchronize_frames_c,
    mse_calc,
    Precondition_calc
)
from Operators import Replicate_frame
from wrap_ops import overlap_cuda,split_cuda
import matplotlib.pyplot as plt

import config

GPU = config.GPU

if GPU:
    import cupy as cp
    xp = cp
    mempool = cp.get_default_memory_pool()
else:
    xp = np


eps0 = xp.float32(1e-2)

eps_illum = None


############################################
# timings
#
timers = {
    "solver_tot": 0,
    "solver_loop": 0,
    "ProxD": 0,
    "Overlap": 0,
    "Sync": 0,
    "illuminate&split": 0,
    "refine_illumination": 0,
    "mse_step": 0,
    "mse_truth": 0,
    "copies": 0,
    "loop_intrnl": 0,
    "solver_final": 0,
    "solver_init": 0,
}


def get_times():
    return timers


def reset_times():
    for keys in timers:
        timers[keys] = 0


def normalize_times(tot=None):
    if type(tot) == type(None):
        tot = 0
        for keys in timers:
            tot += timers[keys]
    try:
        for keys in timers:
            timers[keys] /= tot
    except:
        pass
    return tot


############################################


def refine_illumination_function(
    img, illumination, frames, Split, Overlap, lens_mask=None
):
    """
    refine_illumination based on

    Parameters
    ----------
    img : TYPE
        input image.
    illumination : TYPE
        initial illumination.
    frames : TYPE
        frames estimate.
    Split : TYPE
        Split operator.
    Overlap : TYPE
        overlap operator.
    lens_mask : TYPE, optional
        lens mask in F-space to remove grid pathology. The default is None.

    Returns
    -------
    illumination : TYPE
        refined illumination.
    normalization : TYPE
        refined normalization.

    """

    # eps_illum = None
    global eps_illum
    frames_split = Split(img)
    norm_frames = xp.mean(xp.abs(frames_split) ** 2, 0)
    if type(eps_illum) == type(None):
        eps_illum = xp.max(norm_frames) * eps0

    illumination = xp.sum(
        frames * xp.conj(Split(img)) + eps_illum * illumination, 0
    ) / (norm_frames + eps_illum)

    # apply mask to illumination
    if type(lens_mask) != type(None):
        illumination = xp.fft.fft2(illumination)
        illumination *= lens_mask
        illumination = xp.fft.ifft2(illumination)

    normalization = Overlap(
        Replicate_frame(xp.abs(illumination) ** 2, frames_split.shape[0])
    )  # check
    return illumination, normalization

'''
def Alternating_projections(
    sync,
    img,
    Gramiam,
    illumination,
    Overlap,
    Split,
    frames_data,
    refine_illumination,
    maxiter,
    normalization,
    img_truth,
    residuals_interval,
):
    """
    Parameters
    ----------
    sync : bool 
        synchronization
    img : 2d matrix
        reconstructed image.
    Gramiam:
        
    illumination : 2d matrix
        ukkynubatuib.
    Overlap : TYPE
        overlap operator.
    Split : TYPE
        DESCRIPTION.
    frames_data : TYPE
        Data.
    refine_illumination : bool, optional
        DESCRIPTION. The default is False.
    maxiter : TYPE, optional
        max iterations. The default is 100.
    normalization : 2d matrix, optional
        normalization, computed internally if None. The default is None.
    img_truth : TYPE, optional
        truth, used to compare . The default is None.

    Returns
    -------
    img : TYPE
        image.
    frames : TYPE
        frames.
    residuals : matrix shape (3, maxiter)
        normalized residuals:
            0) | img - f truth|;
            1) | |F(frames)| - |data||
            2) | frames_new - frames_old|


    """
    if GPU:
        print(
            "start AP memory used, and total:",
            mempool.used_bytes(),
            mempool.total_bytes(),
        )
        print(
            "normalized by data.nbytes memory used and total normalized:",
            mempool.used_bytes() / frames_data.nbytes,
            mempool.total_bytes() / frames_data.nbytes,
        )
        print("data size", frames_data.nbytes)
        print("----")

    t000 = timer()
    t00 = t000

    # we need the frames norm to normalize
    frames_norm_sum = xp.linalg.norm(xp.sqrt(frames_data))
    # renormalize the norm for the ifft2 space
    frames_norm_r = frames_norm_sum / xp.sqrt(xp.prod(xp.array(frames_data.shape[-2:])))

    # Prox_data = prox_data_plan(frames_data)

    if GPU:
        print(
            "after Prox_data, memory used, and total:",
            mempool.used_bytes(),
            mempool.total_bytes(),
        )
        print(
            "normalized by data.nbytes memory used and total normalized:",
            mempool.used_bytes() / frames_data.nbytes,
            mempool.total_bytes() / frames_data.nbytes,
        )
        print("----")

    # get the frames from the inital image
    frames = Illuminate_frames(Split(img), illumination)

    if GPU:
        print(
            "after frames initial, memory used, and total:",
            mempool.used_bytes(),
            mempool.total_bytes(),
        )
        print(
            "normalized by data.nbytes memory used and total normalized:",
            mempool.used_bytes() / frames_data.nbytes,
            mempool.total_bytes() / frames_data.nbytes,
        )
        print("----")

    nresiduals = int(np.ceil(maxiter / residuals_interval))

    # residuals = xp.zeros((maxiter,3),dtype=xp.float32)
    residuals = xp.zeros((nresiduals, 4), dtype=xp.float32)

    if type(img_truth) != type(None):
        nrm_truth = xp.linalg.norm(img_truth)
    if type(normalization) == type(None):
        nframes = xp.shape(frames_data)[0]
        normalization = Overlap(
            Replicate_frame(xp.abs(illumination) ** 2, nframes)
        )  # check

    # refine_illumination = 1
    # eps_illum = None
    # eps0 = 1e-2
    
    if sync == True:
        inormalization_split = Split(1/(normalization))
        #inormalization_split = Split(1/(normalization+1e-8))
        frames_norm = Precondition_calc(frames, bw=Gramiam['bw'])
        print('!!!frames_norm',frames_norm[0],frames_norm[1])
        #print('!!!frames_norm_shape',frames_norm.dtype,frames_norm.shape)
  
    timers["solver_init"] = timer() - t00
    t00 = timer()
    if GPU:
        print(
            "start loop, memory used, and total:",
            mempool.used_bytes(),
            mempool.total_bytes(),
        )
        print(
            "normalized by data.nbytes memory used and total normalized:",
            mempool.used_bytes() / frames_data.nbytes,
            mempool.total_bytes() / frames_data.nbytes,
        )
        print("----")

    compute_residuals = False
    
    for ii in xp.arange(maxiter):
        print(ii)
        # data projection
        t0 = timer()
        t0_loop = timer()
        # frames, mse_data = Project_data(frames,frames_data)
        if residuals_interval < np.inf:
            compute_residuals = not np.mod(ii, residuals_interval)

        # frames, mse_data = Prox_data(frames, compute_residuals )
        frames, mse_data = Project_data(
            frames, frames_data, compute_residuals=compute_residuals
        )
        # if GPU and ii<2:
        #     print('in loop, after Prox data memory used, and total normalized:', mempool.used_bytes()/frames_data.nbytes,mempool.total_bytes()/frames_data.nbytes )
        #     print('----')

        if compute_residuals:
            residuals[ii // residuals_interval, 1] = mse_data
        timers["ProxD"] += timer() - t0

        t0 = timer()

        frames_old = frames + 0.0  # make a copy
        timers["copies"] += timer() - t0

        # if GPU and ii<2:
        #     print('in loop, after copy memory used, and total normalized:', mempool.used_bytes()/frames_data.nbytes,mempool.total_bytes()/frames_data.nbytes )
        #     print('----')

        ####################
        # here goes the synchronization
        if sync==True:
            t0 = timer()
            omega=synchronize_frames_c(frames, illumination, frames_norm, inormalization_split, Gramiam)
            frames=frames*omega
            timers["Sync"] += timer() - t0
           
        ##################
        
        ##################
        # overlap projection
        t0 = timer()
        img = Overlap(Illuminate_frames(frames, xp.conj(illumination))) / normalization
        timers["Overlap"] += timer() - t0

        t0 = timer()

        if refine_illumination and ii > 5:
            print("refining illum")
            illumination, normalization = refine_illumination_function(
                img, illumination, frames, Split, Overlap, lens_mask=None
            )
        # else:
        #    print('not refining')

        timers["refine_illumination"] += timer() - t0

        t0 = timer()
        #print('3',type(frames.dtype))
        frames = Illuminate_frames(Split(img), illumination)
        #print('4',type(frames.dtype))
        timers["illuminate&split"] += timer() - t0

        # if GPU and ii<2:
        #     print('in loop, after split memory used, and total normalized:', mempool.used_bytes()/frames_data.nbytes,mempool.total_bytes()/frames_data.nbytes )
        #     print('----')

        t0 = timer()
        if compute_residuals:
            residuals[ii // residuals_interval, 2] = xp.linalg.norm(frames - frames_old)
            # residuals[ii//residuals_interval,2] = xp.inner((frames-frames_old).ravel(),(frames-frames_old).ravel())

        timers["mse_step"] += timer() - t0

        if type(img_truth) != type(None):
            t0 = timer()
            if compute_residuals:
                nmse0 = mse_calc(img_truth, img)
                residuals[ii // residuals_interval, 0] = nmse0
            timers["mse_truth"] += timer() - t0
        timers["loop_intrnl"] += timer() - t0_loop

    if GPU:
        print(
            "end of loop, memory used, and total/1e9:",
            mempool.used_bytes() / 1e9,
            mempool.total_bytes() / 1e9,
        )
        print(
            "normalized by data.nbytes memory used and total normalized:",
            mempool.used_bytes() / frames_data.nbytes,
            mempool.total_bytes() / frames_data.nbytes,
        )

        print("----")

    # finalize loop
    # normalize residuals
    timers["solver_loop"] = timer() - t00

    t0 = timer()

    residuals[:, 1] /= frames_norm_sum
    residuals[:, 2] /= frames_norm_r
    if type(img_truth) != type(None):
        residuals[:, 0] /= nrm_truth
        residuals[:,3] = 1/residuals[:,0] #|truth|/|img-truth| measures the SNR
    ##
    timers["solver_final"] += timer() - t0
    timers["solver_tot"] += timer() - t000
    # print('time sync:',time_sync)
    return img, frames, illumination, residuals
'''

def Alternating_projections_c(
    sync,
    img,
    Gramiam,
    illumination,
    translations_x,
    translations_y,
    overlap_cuda,
    split_cuda,
    frames_data,
    refine_illumination,
    maxiter,
    normalization,
    img_truth,
    residuals_interval,
):
    """
    Parameters
    ----------
    sync : bool 
        synchronization
    img : 2d matrix
        reconstructed image.
    Gramiam:
        
    illumination : 2d matrix
        ukkynubatuib.
    Overlap : TYPE
        overlap operator.
    Split : TYPE
        DESCRIPTION.
    frames_data : TYPE
        Data.
    refine_illumination : bool, optional
        DESCRIPTION. The default is False.
    maxiter : TYPE, optional
        max iterations. The default is 100.
    normalization : 2d matrix, optional
        normalization, computed internally if None. The default is None.
    img_truth : TYPE, optional
        truth, used to compare . The default is None.

    Returns
    -------
    img : TYPE
        image.
    frames : TYPE
        frames.
    residuals : matrix shape (3, maxiter)
        normalized residuals:
            0) | img - f truth|;
            1) | |F(frames)| - |data||
            2) | frames_new - frames_old|


    """
    if GPU:
        print(
            "start AP memory used, and total:",
            mempool.used_bytes(),
            mempool.total_bytes(),
        )
        print(
            "normalized by data.nbytes memory used and total normalized:",
            mempool.used_bytes() / frames_data.nbytes,
            mempool.total_bytes() / frames_data.nbytes,
        )
        print("data size", frames_data.nbytes)
        print("----")

    t000 = timer()
    t00 = t000

    # we need the frames norm to normalize
    frames_norm_sum = xp.linalg.norm(xp.sqrt(frames_data))
    # renormalize the norm for the ifft2 space
    frames_norm_r = frames_norm_sum / xp.sqrt(xp.prod(xp.array(frames_data.shape[-2:])))
    translations = (translations_x + 1j * translations_y).astype(np.complex64)
    
    # Prox_data = prox_data_plan(frames_data)

    if GPU:
        print(
            "after Prox_data, memory used, and total:",
            mempool.used_bytes(),
            mempool.total_bytes(),
        )
        print(
            "normalized by data.nbytes memory used and total normalized:",
            mempool.used_bytes() / frames_data.nbytes,
            mempool.total_bytes() / frames_data.nbytes,
        )
        print("----")

    # get the frames from the inital image
    if GPU:
        frames = xp.zeros(frames_data.shape,dtype = xp.complex64)
        split_cuda(img, frames, translations, illumination)
    else:
        frames = Illuminate_frames(Split(img), illumination)

    if GPU:
        print(
            "after frames initial, memory used, and total:",
            mempool.used_bytes(),
            mempool.total_bytes(),
        )
        print(
            "normalized by data.nbytes memory used and total normalized:",
            mempool.used_bytes() / frames_data.nbytes,
            mempool.total_bytes() / frames_data.nbytes,
        )
        print("----")

    nresiduals = int(np.ceil(maxiter / residuals_interval))

    # residuals = xp.zeros((maxiter,3),dtype=xp.float32)
    residuals = xp.zeros((nresiduals, 4), dtype=xp.float32)

    if type(img_truth) != type(None):
        nrm_truth = xp.linalg.norm(img_truth)
    if type(normalization) == type(None):
        nframes = xp.shape(frames_data)[0]
        #normalization = Overlap(Replicate_frame(xp.abs(illumination) ** 2, nframes)
        #print(illumination.shape,img_truth.shape)
        #print( translations.dtype,type( translations))
        normalization = xp.zeros(img_truth.shape,dtype = xp.complex64)
        overlap_cuda(normalization, 0, translations, illumination) 
        #print(normalization)
          # check type

    # refine_illumination = 1
    # eps_illum = None
    # eps0 = 1e-2
    
    if sync == True:
        reg = 1e-8
        #print(xp.minimum(normalization))
        if GPU:
            inormalization_split = xp.zeros(frames_data.shape,dtype = xp.complex64)
            split_cuda(1/(normalization+reg),inormalization_split,translations, 0)
            #print(inormalization_split)
        else:
            #inormalization_split = Split(1/(normalization+1e-8))
            inormalization_split = Split(1/(normalization))
        frames_norm = Precondition_calc(frames, bw=Gramiam['bw'])
        print('!!!frames_norm',frames_norm[0],frames_norm[1])
        #print('!!!frames_norm_shape',frames_norm.dtype,frames_norm.shape)
  
    timers["solver_init"] = timer() - t00
    t00 = timer()
    if GPU:
        print(
            "start loop, memory used, and total:",
            mempool.used_bytes(),
            mempool.total_bytes(),
        )
        print(
            "normalized by data.nbytes memory used and total normalized:",
            mempool.used_bytes() / frames_data.nbytes,
            mempool.total_bytes() / frames_data.nbytes,
        )
        print("----")

    compute_residuals = False

    for ii in xp.arange(maxiter):
        print(ii)
        # data projection
        t0 = timer()
        t0_loop = timer()
        # frames, mse_data = Project_data(frames,frames_data)
        if residuals_interval < np.inf:
            compute_residuals = not np.mod(ii, residuals_interval)

        # frames, mse_data = Prox_data(frames, compute_residuals )
        frames, mse_data = Project_data(
            frames, frames_data, compute_residuals=compute_residuals
        )
        # if GPU and ii<2:
        #     print('in loop, after Prox data memory used, and total normalized:', mempool.used_bytes()/frames_data.nbytes,mempool.total_bytes()/frames_data.nbytes )
        #     print('----')

        if compute_residuals:
            residuals[ii // residuals_interval, 1] = mse_data
        timers["ProxD"] += timer() - t0

        t0 = timer()

        frames_old = frames + 0.0  # make a copy
        timers["copies"] += timer() - t0

        # if GPU and ii<2:
        #     print('in loop, after copy memory used, and total normalized:', mempool.used_bytes()/frames_data.nbytes,mempool.total_bytes()/frames_data.nbytes )
        #     print('----')

        ####################
        # here goes the synchronization
        if sync==True:
            t0 = timer()
            omega=synchronize_frames_c(frames, illumination, frames_norm, inormalization_split, Gramiam, Gramiam['bw'])
            frames=frames*omega
            timers["Sync"] += timer() - t0
           
        ##################
        
        ##################
        # overlap projection
        t0 = timer()
        if GPU == False:
            img = Overlap(Illuminate_frames(frames, xp.conj(illumination))) / normalization
        else:
            img *= 0 
            overlap_cuda(img, frames,translations, illumination) 
            img = img/normalization
            print('Here', type(translations),(translations).dtype)
        timers["Overlap"] += timer() - t0

        t0 = timer()

        if refine_illumination and ii > 5:
            print("refining illum")
            illumination, normalization = refine_illumination_function(
                img, illumination, frames, Split, Overlap, lens_mask=None
            )
        # else:
        #    print('not refining')

        timers["refine_illumination"] += timer() - t0

        t0 = timer()
        #print('3',type(frames.dtype))
        frames = xp.zeros(frames_data.shape,dtype = xp.complex64)
        split_cuda(img,frames, translations,illumination)
   
        #print('4',type(frames.dtype))
        timers["illuminate&split"] += timer() - t0

        # if GPU and ii<2:
        #     print('in loop, after split memory used, and total normalized:', mempool.used_bytes()/frames_data.nbytes,mempool.total_bytes()/frames_data.nbytes )
        #     print('----')

        t0 = timer()
        if compute_residuals:
            residuals[ii // residuals_interval, 2] = xp.linalg.norm(frames - frames_old)
            # residuals[ii//residuals_interval,2] = xp.inner((frames-frames_old).ravel(),(frames-frames_old).ravel())

        timers["mse_step"] += timer() - t0

        if type(img_truth) != type(None):
            t0 = timer()
            if compute_residuals:
                nmse0 = mse_calc(img_truth, img)
                residuals[ii // residuals_interval, 0] = nmse0
            timers["mse_truth"] += timer() - t0
        timers["loop_intrnl"] += timer() - t0_loop

    if GPU:
        print(
            "end of loop, memory used, and total/1e9:",
            mempool.used_bytes() / 1e9,
            mempool.total_bytes() / 1e9,
        )
        print(
            "normalized by data.nbytes memory used and total normalized:",
            mempool.used_bytes() / frames_data.nbytes,
            mempool.total_bytes() / frames_data.nbytes,
        )

        print("----")

    # finalize loop
    # normalize residuals
    timers["solver_loop"] = timer() - t00

    t0 = timer()

    residuals[:, 1] /= frames_norm_sum
    residuals[:, 2] /= frames_norm_r
    if type(img_truth) != type(None):
        residuals[:, 0] /= nrm_truth
        residuals[:,3] = 1/residuals[:,0] #|truth|/|img-truth| measures the SNR
    ##
    timers["solver_final"] += timer() - t0
    timers["solver_tot"] += timer() - t000
    # print('time sync:',time_sync)
    return img, frames, illumination, residuals