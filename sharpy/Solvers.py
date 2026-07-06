"""
Ptycho solvers: the iterative reconstruction loops.

Each solver alternates a data constraint (Fourier-magnitude projection) with
an overlap/image update, optionally with Gramian phase synchronization and/or
position retrieval. See the Operators.py module docstring for the
naming/vocabulary glossary (Gramian / braket / zQQz) and the key operators.

Key solvers
-----------
  Alternating_projections        CPU AP loop (NumPy), optional Gramian sync.
  Alternating_projections_c      GPU AP loop (CuPy + CUDA Split/Overlap/zQQz),
                                 the production reconstruction with sync.
  Alternating_projections_position
                                 AP loop with scan-position retrieval
                                 (method="diag" | "coupled"); see
                                 position_retrieval.py.
  plot_intermediate              debug montage of intermediate images.

In operator terms each iteration is:
  Split -> Illuminate -> Propagate -> Project_data -> IPropagate
        -> [synchronize_frames_c] -> Overlap  (-> next iterate)
"""

import numpy as np
from timeit import default_timer as timer
from Operators import (
    make_probe,
    Illuminate_frames,
    Project_data,
    Propagate,
    synchronize_frames_c,
    mse_calc,
    Precondition_calc
)
from ap_accel import line_search as _line_search
from Operators import Replicate_frame, synchronize_illum_c,refine_illumination_pairwise,refine_illumination_function
import Operators  # for the live _FUSED_PROXD flag + _diffnorm (fused eps_S residual)
import config
if config.GPU:
    from wrap_ops import overlap_cuda,split_cuda
import matplotlib.pyplot as plt
from tqdm import tqdm

GPU = config.GPU

if GPU:
    import cupy as cp
    xp = cp
    mempool = cp.get_default_memory_pool()
else:
    xp = np


eps0 = xp.float32(1e-2)

eps_illum = None

reg = 1e-8 #for sync normalizations
reg_img = None #need implement
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
#old version that does not have the GPU calculations
############################################

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
    line_search=False,
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
    line_search : bool, optional
        Over-relax the AP step by the alpha that minimizes the Fourier data
        misfit (Newton line search, ap_accel.line_search; parameter-free,
        alpha~1.9). Default False (unchanged behaviour); ~1.8x fewer iterations.

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

    if line_search:
        frames_amp = xp.sqrt(frames_data)   # sqrt(I), for the data-misfit line search

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
    reg0 = 1e-6 #control reg
    
    if sync == True:
        inormalization_split = Split(1/(normalization))
        #inormalization_split = Split(1/(normalization+1e-8))
        #frames_norm = Precondition_calc(frames, bw=Gramiam['bw'])
        frames_norm = Precondition_calc(frames_data, bw=Gramiam['bw'])

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
        if line_search:
            z0 = frames + 0.0   # carried iterate at loop top (before P_a)
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

        if Operators._FUSED_PROXD:
            # reference, not a copy: every step below rebinds `frames` to a new
            # array (never mutates in place), so the old buffer stays alive via
            # this ref; skip it entirely when the eps_S residual isn't needed.
            frames_old = frames if compute_residuals else None
        else:
            frames_old = frames + 0.0  # make a copy
        timers["copies"] += timer() - t0

        # if GPU and ii<2:
        #     print('in loop, after copy memory used, and total normalized:', mempool.used_bytes()/frames_data.nbytes,mempool.total_bytes()/frames_data.nbytes )
        #     print('----')

        ####################
        # here goes the synchronization
        if sync==True:
            t0 = timer()
            frames_norm = Precondition_calc(frames_data, bw=Gramiam['bw']) ###????
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

        if refine_illumination:
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

        if line_search:
            # AP step Dz = P_Q P_a z0 - z0; over-relax by the alpha minimizing the
            # data misfit ||sqrt(I) - |Propagate(z0 + alpha Dz)|||^2 (Fourier,
            # 2 Newton steps from 1). Parameter-free; alpha~1.9 (over-relaxation).
            Dz = frames - z0
            # Propagate overwrites its input in place on GPU (fft_plan owrite=True),
            # so pass copies -- otherwise z0 and Dz get clobbered with their own
            # FFTs before the update below (CPU scipy.fftpack does not overwrite).
            alpha = _line_search(Propagate(z0 + 0.0), Propagate(Dz + 0.0), frames_amp)
            frames = z0 + alpha * Dz

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


#####
# With GPU calculations
# Below is to use pairwise update illumination new approach
####
def Alternating_projections_c(
    sync,
    img,
    Gramiam,
    illumination_truth,
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
    sync_interval=1,
    num_iter = 5
):
    """
    Parameters
    ----------
    sync : bool 
        synchronization
    img : 2d matrix
        reconstructed image.
    Gramiam:
        
    illumination_truth : 2d matrix
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
    eig_plan: dictionary for eigensolver, optional
    
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
    reg0 = 1e-08
    
    # we need the frames norm to normalize
    frames_norm_sum = xp.linalg.norm(xp.sqrt(frames_data))

    # renormalize the norm for the ifft2 space
    frames_norm_r = frames_norm_sum / xp.sqrt(xp.prod(xp.array(frames_data.shape[-2:])))
 
    translations = (translations_x + 1j * translations_y).astype(xp.complex64)
 
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
    
    #initial illumination
    if refine_illumination:
        nrm_illumination = xp.linalg.norm(illumination_truth)
        
        #starting guess 
        
        #illumination_start = xp.random.rand(frames_data.shape[1],frames_data.shape[2],dtype = xp.float32) + 1j * xp.random.rand(frames_data.shape[1],frames_data.shape[2],dtype = xp.float32)

  
        #w_initial = make_probe(frames_data.shape[1],frames_data.shape[2], r1=0.01*3, r2=0.09*3, fx=+18, fy=-18)# on CPU
        w_initial,lens_mask = make_probe(frames_data.shape[1],frames_data.shape[2], r1=0.025*3, r2=0.085*3, fx=+10, fy=-10)
        illumination_start = xp.array(w_initial, dtype=xp.complex64)
        
        '''
        illumination_start = illumination_truth + 0.1 * xp.random.rand(frames_data.shape[1],frames_data.shape[2],dtype = xp.float32) + 0.5 * 1j*xp.random.rand(frames_data.shape[1],frames_data.shape[2],dtype = xp.float32) 
        '''

        #illumination_start /= xp.linalg.norm(illumination_start) * nrm_illumination
        #eps_illum = 1e-8
        
    else:
        illumination_start = illumination_truth + 0
        illumination = illumination_start
    
    #if refine_illumination == False and type(normalization) == type(None) and sync == True:
    if type(normalization) == type(None):
        nframes = xp.shape(frames_data)[0]
        normalization = xp.zeros(img_truth.shape,dtype = xp.complex64)
        overlap_cuda(normalization, 0, translations, illumination_start + 0) 
 

    if sync == True:

        if GPU:
            inormalization_split = xp.zeros(frames_data.shape,dtype = xp.complex64)
            reg = reg0 * xp.max(xp.abs(normalization)) #for accuracy, choose reg0 = 1e-08
            split_cuda(1/(normalization+reg),inormalization_split,translations, 0)
            print('!!!inorm',np.isnan(inormalization_split).any())
            #print(inormalization_split)
        else:
            #inormalization_split = Split(1/(normalization+1e-8))
            inormalization_split = Split(1/(normalization))
        
    # get the frames from the inital image
    if GPU:
        frames = xp.zeros(frames_data.shape,dtype = xp.complex64)
        split_cuda(img, frames, translations, illumination_start)
            
    else:
        frames = Illuminate_frames(Split(img), illumination_start)
    
    frames, mse_data = Project_data(
            frames, frames_data, compute_residuals=False
        )
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
    
    if sync == True:
        frames_norm = Precondition_calc(frames, bw=Gramiam['bw'])
        
    
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

    compute_residuals = True
    iii = True #if print
    images =[] #initialize the dictionary to store all intermediate results
    illuminations = [] #store intermediate illuminations
    
    for ii in tqdm(range(maxiter)):

        # data projection
        t0 = timer()
        t0_loop = timer()
        
        #############
        # plot results
        if ii%100 == 0:
            current_image = abs(img.get())
            images.append(current_image) 
        if ii%100 == 0:
            current_illumination = abs(illumination_start.get())
            illuminations.append(current_illumination) 
        
        ####################
        # here goes update frames
        t0 = timer()
        
        frames = xp.zeros(frames_data.shape,dtype = xp.complex64)
        split_cuda(img,frames, translations,illumination_start + 0)
  
        timers["illuminate&split"] += timer() - t0
        
        frames, mse_data = Project_data(
            frames, frames_data, compute_residuals=compute_residuals
        )
        
        if refine_illumination and sync:
            tic = timer()
            frames_norm = Precondition_calc(frames, bw=Gramiam['bw']) #illumination changes the norm
            
        if residuals_interval < np.inf:
            compute_residuals = not np.mod(ii, residuals_interval)
        
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
            if ii%sync_interval == 0:
                omega = synchronize_frames_c(frames, illumination_start, frames_norm, inormalization_split, Gramiam, Gramiam['bw'],num_iter)
                frames = frames * omega
            timers["Sync"] += timer() - t0
        
        ##################
        # overlap projection
        t0 = timer()
        if GPU == False:
            img = Overlap(Illuminate_frames(frames, xp.conj(illumination_start))) / normalization
        else: 
            img0 = img * 0 
            overlap_cuda(img0, frames,translations, illumination_start) 
            if refine_illumination:
                tic = timer()
                reg_img = reg0 * xp.max(xp.abs(normalization))
                img = (img0 + reg_img * img) / (normalization + reg_img * xp.eye(normalization.shape[0],normalization.shape[1], dtype = normalization.dtype))
                #img = (img0 + reg_img * img) / (normalization + reg_img)
                #img = img0 / (normalization+reg_img)
            else:
                reg = reg0 * xp.max(xp.abs(normalization))
                img = img0/(normalization+reg)

          
        timers["Overlap"] += timer() - t0
        
        ##################
        #here goes refine_illumination
        t0 = timer()

        if refine_illumination:
            if True:
                if ii == 0: 
                    print("refining illum traditional,using lens mask")
                    plt.imshow(np.fft.ifftshift(lens_mask.get()))
                    plt.show()
                illumination = refine_illumination_function(
                        img+0, illumination_start+0, illumination_truth+0,frames+0, translations, split_cuda, overlap_cuda, GPU,lens_mask,ii)

                illumination /= xp.max(xp.abs(illumination)) #normalization kinda crucial
                illumination_start = illumination + 0.0 #make a copy
                
            else:
                #this part for the pairwise, need implementation
                if ii == 0:
                    print("refining illum pairwise, using lens mask")
                    lens_mask1 = abs(xp.fft.fft2(illumination_truth))>1e-3
                    print(lens_mask1.dtype,type(lens_mask1),lens_mask.dtype,type(lens_mask))
                    plt.imshow(lens_mask.get())
                    plt.show()
                    
                illumination = refine_illumination_pairwise(
    img * 0 , illumination_start, illumination_truth+0, frames, translations, split_cuda, overlap_cuda, lens_mask)
                illumination /= max(xp.abs(illumination).flatten())
                illumination_start = illumination + 0.0 #make a copy
                    
            #update the normalization after illum refinement 
            normalization = xp.zeros(img_truth.shape,dtype = xp.complex64)   
            overlap_cuda(normalization, 0, translations, illumination) 
   
            if sync == True:
                #update inormalization for sync
                inormalization_split = xp.zeros(frames_data.shape,dtype = xp.complex64)
                reg = reg0 * xp.max(xp.abs(normalization))
                split_cuda(1/(normalization+reg),inormalization_split,translations, 0)         
                
        timers["refine_illumination"] += timer() - t0
        
        
        # if GPU and ii<2:
        #     print('in loop, after split memory used, and total normalized:', mempool.used_bytes()/frames_data.nbytes,mempool.total_bytes()/frames_data.nbytes )
        #     print('----')
        
        
        ##############
        # compute residuals
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
                if (nmse0/nrm_truth < 1e-4) and (iii == True):
                    iii = False
                    print(f'nmse0 reach 1e-4 accuracy in {ii} iterations and {timer() - t000} time')
                    #break
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
    
    #plot all itermediate results
    plot_intermediate(images)
    if refine_illumination == True:
        plot_intermediate(illuminations)
    
    return img, frames, illumination, residuals



_ZQQZ_CACHE = {}


def _zqqz_kernel():
    """Lazily build (once) the raw zQQz Gramian kernel for the host-streaming sync path."""
    if "k" not in _ZQQZ_CACHE:
        _ZQQZ_CACHE["k"] = cp.RawKernel(Operators.zQQz_raw_kernel, "dotp", jitify=True,
                                        options=("--std=c++17",))
    return _ZQQZ_CACHE["k"]


def _gramian_host_chunked(F, frames_norm, translations, illumination, norm_reg_inv, plan,
                          split_cuda, nx, pair_chunk):
    """Assemble the sync Gramian H from a HOST-resident frame store F (numpy) by streaming
    locality-ordered PAIR-CHUNKS to the device: gather only the frames each chunk touches
    (peak ~ overlap-degree << nf), recompute inorm_split[U] via split_cuda (never O(nf) on
    device), run the raw zQQz kernel per chunk, scatter into the original val slots. Bit-
    identical to the single-shot Gramiam_calc_cuda (validated). Returns H."""
    kern = _zqqz_kernel()
    col = cp.asnumpy(plan["col"]).astype(np.int64)
    row = cp.asnumpy(plan["row"]).astype(np.int64)
    dx = plan["dx"]; dy = plan["dy"]; bw = int(plan["bw"]); nnz = col.size
    val = cp.zeros((nnz, 1), dtype=cp.complex64)
    fn = frames_norm.astype(cp.complex64)
    order = np.argsort(col, kind="stable")                       # frame-locality order
    for s in range(0, nnz, pair_chunk):
        pidx = order[s:min(nnz, s + pair_chunk)]
        c = col[pidx]; r = row[pidx]
        U = np.unique(np.concatenate([c, r]))
        pos = np.empty(int(U.max()) + 1, np.int64); pos[U] = np.arange(U.size)
        Ud = cp.asarray(U)
        F_sub = cp.asarray(F[U]).astype(cp.complex64)            # host -> device (<= 2*chunk frames)
        inorm_sub = cp.zeros((U.size, nx, nx), dtype=cp.complex64)
        split_cuda(norm_reg_inv, inorm_sub, translations[Ud], 0)
        cl = cp.asarray(pos[c]).astype(int); rl = cp.asarray(pos[r]).astype(int)
        pidxd = cp.asarray(pidx)
        vch = cp.zeros((pidx.size, 1), dtype=cp.complex64)
        kern((int(pidx.size),), (128,), (vch, F_sub, fn[Ud], illumination, inorm_sub,
             cl, rl, dx[pidxd], dy[pidxd], bw, int(pidx.size), nx, nx))
        val[pidxd] = vch
    return plan["val2H"](val.ravel())


def Alternating_projections_batched_c(
    sync, img, Gramiam, illumination, translations_x, translations_y,
    overlap_cuda, split_cuda, frames_data, refine_illumination, maxiter,
    normalization=None, img_truth=None, residuals_interval=1, sync_interval=1,
    num_iter=5, frame_batch=1024, frame_store="auto", pair_chunk=4096,
):
    """Frame-batched GPU drop-in for Alternating_projections_c.

    Same signature + ``frame_batch`` (B), ``frame_store`` and ``pair_chunk``; same return
    ``(img, frames, illumination, residuals)``. Processes the frames stack in batches of B
    instead of materialising it whole (PASS-A split->Project per batch -> F; SYNC Gramian +
    Eigensolver; PASS-B overlap(F*omega) per batch), dropping the full-stack ``frames_old``
    copy + FFT temp + framesl/framesr.

    TWO tiers, chosen by ``frame_store`` ("auto" | "device" | "host"):
      * DEVICE (fast): F device-resident + single-shot Gramiam_calc_cuda. peak ~ data + F +
        inorm + O(B) (~5x data vs Alternating_projections_c's ~10-12x) -> ~2x the frame cap
        at ~production speed (A100: 271 ms/iter @65536 fr nx=128, where the full stack OOMs).
      * HOST (fits any size): F on HOST, the Gramian assembled by streaming pair-chunks to
        the device (_gramian_host_chunked), inorm recomputed per chunk. device peak ~ O(B) +
        O(degree) -> unbounded frame count, but transfer-bound (~1 order slower).
    "auto" uses DEVICE when free GPU memory comfortably holds ~2 frame stacks, else HOST.
    Both are bit-exact vs Alternating_projections_c (validated ~1e-5, identical truth-NMSE).

    ``residuals[:,2]`` (||frames-frames_old||, ePIE step) is NOT computed -- that copy is what
    batching avoids -- left 0. In HOST mode the returned ``frames`` is a numpy array. GPU-only;
    refine_illumination not supported yet.
    """
    assert GPU, "batched solver is a GPU path"
    assert not refine_illumination, "refine_illumination not supported in the batched path yet"
    reg0 = 1e-8
    nframes, nx, ny = frames_data.shape
    B = frame_batch if (frame_batch and frame_batch > 0) else nframes
    illumination_start = illumination
    translations = (translations_x + 1j * translations_y).astype(cp.complex64)

    # ── tier select: device-resident F (fast) vs host-streamed F (any size) ──
    stack_bytes = nframes * nx * ny * 8                          # one complex64 frame stack
    if frame_store == "auto":
        free = int(cp.cuda.Device().mem_info[0])                # device tier ~ F + inorm (2 stacks)
        use_host = free < 2.6 * stack_bytes
    else:
        use_host = (frame_store == "host")

    # normalization + inorm image (stack-free)
    if normalization is None:
        normalization = cp.zeros(img.shape, dtype=cp.complex64)
        overlap_cuda(normalization, 0, translations, illumination_start + 0)
    reg = reg0 * float(xp.max(xp.abs(normalization)))
    norm_reg_inv = (1.0 / (normalization + reg)).astype(cp.complex64)
    inorm_split = None
    if sync:
        Operators.eig_reset()
        if not use_host:                                        # device tier keeps O(nf) inorm resident
            inorm_split = cp.zeros(frames_data.shape, dtype=cp.complex64)
            split_cuda(norm_reg_inv, inorm_split, translations, 0)

    # residual bookkeeping (matches Alternating_projections_c)
    nresiduals = int(np.ceil(maxiter / residuals_interval))
    residuals = xp.zeros((nresiduals, 4), dtype=xp.float32)
    frames_norm_sum = xp.linalg.norm(xp.sqrt(frames_data))
    frames_norm_r = frames_norm_sum / xp.sqrt(xp.prod(xp.array(frames_data.shape[-2:])))
    if img_truth is not None:
        nrm_truth = xp.linalg.norm(img_truth)

    F = (np.empty(frames_data.shape, dtype=np.complex64) if use_host
         else cp.zeros(frames_data.shape, dtype=cp.complex64))
    frames_norm = cp.zeros(nframes, dtype=cp.complex64)

    for ii in tqdm(range(maxiter)):
        cr = (ii % residuals_interval == 0)
        mse_acc = 0.0
        # PASS A: split + data-project per batch -> F (device or host) + per-frame norm
        for s in range(0, nframes, B):
            e = min(nframes, s + B)
            fb = cp.zeros((e - s, nx, ny), dtype=cp.complex64)
            split_cuda(img, fb, translations[s:e], illumination_start)
            fb, mse = Project_data(fb, frames_data[s:e], compute_residuals=cr)
            if cr and mse is not None:
                mse_acc += float(mse)
            F[s:e] = cp.asnumpy(fb) if use_host else fb
            if sync:
                frames_norm[s:e] = Precondition_calc(fb, bw=Gramiam["bw"])
        # SYNC: host-streamed chunked Gramian OR device single-shot -> omega
        omega = None
        if sync and (ii % sync_interval == 0):
            if use_host:
                H = _gramian_host_chunked(F, frames_norm, translations, illumination_start,
                                          norm_reg_inv, Gramiam, split_cuda, nx, pair_chunk)
            else:
                H = Operators.Gramiam_calc_cuda(F, Gramiam, illumination_start, inorm_split, frames_norm)
            omega = Operators.Eigensolver(H, num_iter).reshape(nframes, 1, 1)
        # PASS B: (apply omega) + overlap per batch -> object
        img0 = cp.zeros(img.shape, dtype=cp.complex64)
        for s in range(0, nframes, B):
            e = min(nframes, s + B)
            fb = cp.asarray(F[s:e]) if use_host else F[s:e]
            if omega is not None:
                fb = fb * omega[s:e]
            overlap_cuda(img0, fb, translations[s:e], illumination_start)
        img = img0 / (normalization + reg)
        # residuals (recon-side exact; step-size residuals[:,2] intentionally skipped)
        if cr:
            residuals[ii // residuals_interval, 1] = mse_acc
            if img_truth is not None:
                residuals[ii // residuals_interval, 0] = mse_calc(img_truth, img)

    residuals[:, 1] /= frames_norm_sum
    if img_truth is not None:
        residuals[:, 0] /= nrm_truth
        residuals[:, 3] = 1.0 / (residuals[:, 0] + 1e-30)
    return img, F, illumination_start, residuals


def plot_intermediate(images):
    n_images = len(images)
    print(n_images)
    cols = 5  # Set number of columns for the grid
    rows = (n_images + cols - 1) // cols  # Calculate number of rows needed

    fig, axs = plt.subplots(rows, cols, figsize=(15, rows * 3))

    for ax in axs.flat:
        ax.axis('off')  # Turn off axis for all subplots

    # Plot images
    for idx, image in enumerate(images):
        ax = axs.flat[idx]
        ax.imshow(abs(image),cmap="gray")
        ax.set_title(f'Image {idx * 100}')  # Title based on the iteration number

    plt.tight_layout()
    plt.show()
    

############################################
# Alternating projections with position retrieval
# (Section IV of arXiv:1209.4924; port of doraar0_shift.m + fit_shift.m)
############################################
def fit_drift_global(
    frames_data, illumination, translations_x, translations_y, nx, ny, Nx, Ny,
    model="linear", drift_max=5.0, ngrid=11, sweeps=3, ap_iters=10,
):
    """Global parametric drift fit (Section III.A / Eckert "rigid source projection").

    Recover a *structured* per-frame drift by minimizing the TOTAL Fourier-
    magnitude data misfit over a low-dimensional model of the shift -- a few
    scalars -- via coordinate-descent grid search. Each coordinate is swept over
    its whole plausible range (GLOBAL, no per-frame capture basin), so this
    catches drift far beyond the ~1/k_max basin of the local solver; the residual
    is then polished by the per-frame solver in the AP loop.

    The shift is modelled as xi = B @ c on the centered nominal scan coordinate
    s = (translations - mean):
      model="linear" : B = [sx, sy]              -> A.s  (magnification/rotation/
                       shear; 4 coeffs) = the real-space analog of Eckert's
                       affine rigid source projection of the scan grid.
      model="poly2"  : B = [sx, sy, sx^2, sx*sy, sy^2]  (10 coeffs) for curved drift.

    Returns per-frame (xi_x, xi_y). The constant (global translation) term is
    omitted -- it is the unobservable position gauge.
    """
    from Operators import Splitc, Overlapc, map_frames
    from position_retrieval import shift_probe_fourier

    mapid = map_frames(translations_x, translations_y, nx, ny, Nx, Ny)
    sx = translations_x - xp.mean(translations_x)
    sy = translations_y - xp.mean(translations_y)
    scale = max(float(xp.max(xp.abs(sx))), float(xp.max(xp.abs(sy))), 1.0)
    sx = sx / scale
    sy = sy / scale
    cols = [sx, sy, sx * sx, sx * sy, sy * sy] if model == "poly2" else [sx, sy]
    k = len(cols)
    B = xp.stack(cols, axis=1)  # (nframes, k)

    def xi_of(C):
        return B @ C[:k], B @ C[k:]

    def misfit(C):
        xi_x, xi_y = xi_of(C)
        ps = shift_probe_fourier(illumination, xi_x, xi_y)
        norm = Overlapc(xp.abs(ps) ** 2, Nx, Ny, mapid)
        norm = xp.where(xp.abs(norm) < 1e-6 * float(xp.max(xp.abs(norm))), 1.0, norm)
        im = xp.ones((Nx, Ny), dtype=ps.dtype)
        frames = Splitc(im, mapid) * ps
        mse = 0.0
        for _ in range(ap_iters):
            frames, mse = Project_data(frames, frames_data, compute_residuals=True)
            im = Overlapc(frames * xp.conj(ps), Nx, Ny, mapid) / norm
            frames = Splitc(im, mapid) * ps
        _, mse = Project_data(frames, frames_data, compute_residuals=True)
        return float(mse)

    C = xp.zeros(2 * k)
    half = float(drift_max)
    for _ in range(int(sweeps)):
        for i in range(2 * k):
            grid = float(C[i]) + xp.linspace(-half, half, ngrid)
            best_g, best_e = float(C[i]), float("inf")
            for g in grid:
                C[i] = g
                e = misfit(C)
                if e < best_e:
                    best_e, best_g = e, float(g)
            C[i] = best_g
        half = half * 2.0 / (ngrid - 1)  # shrink toward the grid spacing to refine
    return xi_of(C)


def Alternating_projections_position(
    img,
    illumination,
    frames_data,
    translations_x,
    translations_y,
    nx,
    ny,
    Nx,
    Ny,
    maxiter,
    position_start=100,
    position_every=1,
    max_step=0.5,
    reg=1e-10,
    method="diag",
    diag_method="taylor",
    backend=None,
    replan=False,
    drift_fit=False,
    drift_model="linear",
    drift_max=5.0,
    xi_x_init=None,
    xi_y_init=None,
    prelocalize=False,
    prelocalize_radius=4,
    prelocalize_rounds=6,
    prelocalize_every=10,
    img_truth=None,
    xi_x_truth=None,
    xi_y_truth=None,
    residuals_interval=1,
):
    """Alternating projections with position (scan-error) retrieval.

    Mirrors `Alternating_projections` but, on a cadence, refines a per-frame
    sub-pixel probe shift xi = (xi_x, xi_y) using `position_solve_diag`
    (Section IV).  The shift is carried through the probe Taylor model -- the
    integer frame map stays fixed and the illumination becomes a per-frame
    stack -- matching the MATLAB `fit_shift.m`/`doraar0_shift.m` approach.

    Parameters
    ----------
    img : (Nx, Ny) complex
        Initial image estimate.
    illumination : (nx, ny) complex
        Probe (single frame); derivatives are taken from it.
    frames_data : (nframes, nx, ny) real
        Measured intensities |F z|^2.
    translations_x, translations_y : (nframes,)
        Integer base scan positions (the sub-pixel part is retrieved).
    nx, ny, Nx, Ny : int
        Frame and image dimensions.
    maxiter : int
        Number of AP iterations.
    position_start : int
        Iteration at which to begin position updates (MATLAB: iter > 100).
    position_every : int
        Cadence of position updates (MATLAB Fig 7: every iteration).
    max_step : float
        Trust-region cap on the per-frame step, in pixels.
    method : {"diag", "coupled"}
        Position solver: "diag" = independent per-frame 2x2 solve;
        "coupled" = full Eq. (27) sparse solve with off-diagonal overlap
        coupling (more accurate, converges faster).
    diag_method : {"taylor", "exact"}
        Re-linearization for the diagonal solver: "taylor" = 2nd-order Taylor
        model (cheap, valid ~1/k_max); "exact" = exact band-limited shifted
        probe + derivatives (machine precision in-range, wider basin). Ignored
        by the coupled solver.
    prelocalize : bool
        One-shot multiplex coarse pre-localization (Section III.A) before the
        first position update: a convex, basin-free NNLS/matched-filter fit over
        a comb of candidate shifted probes on the consensus object localizes the
        integer per-frame shift, migrated into the map. Extends capture well
        beyond the local solver's ~1/k_max basin (pair with diag_method="exact").
    prelocalize_radius : int
        Comb half-width in pixels for prelocalize (candidates in +-radius).
    backend : {None, "auto", "python", "numba", "omp"}
        Overlap-kernel backend for the coupled solver (None -> module
        default; see position_retrieval.set_kernel_backend). Ignored by diag.
    replan : bool
        Map re-planning (opt-in): migrate the integer part of each per-frame
        shift into the integer frame map, keeping only the <0.5px residual in
        the Taylor model. Extends the representable shift range. Requires an
        apodized probe to be intensity-consistent; default off.
    drift_fit : bool
        Opt-in global parametric drift fit before the loop (see fit_drift_global):
        recovers STRUCTURED drift (magnification/rotation/shear/poly) by global
        search on the total data misfit -- no per-frame capture basin, catches
        drift far beyond ~1/k_max. The fit alone is the endpoint for purely
        structured drift; a *free* per-frame refine afterward over-fits it, so
        either set position_start >= maxiter (fit only) or keep per-frame updates
        for genuine random jitter only.
    drift_model : {"linear", "poly2"}
        Drift basis: "linear" = affine of the nominal scan (A.s; the real-space
        analog of Eckert's rigid source projection); "poly2" adds quadratic terms.
    drift_max : float
        Max drift (px) the global search spans per coefficient.
    xi_x_init, xi_y_init : (nframes,), optional
        Explicit per-frame shift warm start (integer part migrated into the map,
        residual into the Taylor model). Mutually exclusive with drift_fit.
    reg : float
        Tikhonov weight in the image least squares.
    img_truth : (Nx, Ny) complex, optional
        Ground-truth image, for the error metric only.
    xi_x_truth, xi_y_truth : (nframes,), optional
        Ground-truth shifts, for the position error metric only.
    residuals_interval : int
        How often to record residuals.

    Returns
    -------
    img, frames : reconstructed image and frames
    xi_x, xi_y : recovered per-frame sub-pixel shifts
    residuals : (nres, 4) array, columns:
        0) image MSE vs truth (if img_truth given)
        1) Fourier-magnitude residual
        2) frame step ||frames_new - frames_old||
        3) position error eps_xi = sum|xi-xi_truth|^2 / sum|xi_truth|^2
    """
    from Operators import Splitc, Overlapc, map_frames
    from position_retrieval import (
        probe_derivatives,
        taylor_shift_probe,
        position_solve_diag,
        position_solve_coupled,
        position_plan,
        multiplex_prelocalize,
    )

    nframes = frames_data.shape[0]
    mapid = map_frames(translations_x, translations_y, nx, ny, Nx, Ny)
    dp = probe_derivatives(illumination)

    plan = None
    if method == "coupled":
        plan = position_plan(translations_x, translations_y, nframes, nx, ny, Nx, Ny)

    # Accumulated sub-pixel shift estimate (starts at zero -> probe broadcast).
    xi_x = xp.zeros(nframes)
    xi_y = xp.zeros(nframes)
    prelocalize_count = 0  # multiplex coarse pre-localization rounds done

    # For map re-planning: keep the initial integer positions so the *total*
    # recovered shift = residual xi + the integers migrated into the map.
    translations_x0 = translations_x + 0.0
    translations_y0 = translations_y + 0.0

    # Warm start: a global parametric drift fit (structured drift, beyond the
    # local basin), or an explicit per-frame xi guess. The integer part is
    # migrated into the map so the Taylor/exact model holds only the residual;
    # the total-shift accounting (translations_*0 captured above) recovers it.
    if drift_fit:
        xi_x, xi_y = fit_drift_global(
            frames_data, illumination, translations_x, translations_y,
            nx, ny, Nx, Ny, model=drift_model, drift_max=drift_max,
        )
    elif xi_x_init is not None:
        xi_x = xi_x_init + 0.0
        xi_y = xi_y_init + 0.0
    if drift_fit or xi_x_init is not None:
        n_x = xp.round(xi_x)
        n_y = xp.round(xi_y)
        translations_y = translations_y - n_x
        translations_x = translations_x - n_y
        xi_x = xi_x - n_x
        xi_y = xi_y - n_y
        mapid = map_frames(translations_x, translations_y, nx, ny, Nx, Ny)
        if method == "coupled":
            plan = position_plan(translations_x, translations_y, nframes, nx, ny, Nx, Ny)

    def probe_and_norm(xi_x, xi_y):
        probe_stack = taylor_shift_probe(dp, xi_x, xi_y)["O"]  # (nframes, nx, ny)
        normalization = Overlapc(xp.abs(probe_stack) ** 2, Nx, Ny, mapid)
        return probe_stack, normalization

    probe_stack, normalization = probe_and_norm(xi_x, xi_y)

    # Initial frames from the starting image.
    frames = Splitc(img, mapid) * probe_stack

    nres = int(np.ceil(maxiter / residuals_interval))
    residuals = xp.zeros((nres, 4), dtype=xp.float64)

    if img_truth is not None:
        nrm_truth = xp.linalg.norm(img_truth)
    track_xi = xi_x_truth is not None and xi_y_truth is not None
    if track_xi:
        den_xi = xp.sum(xi_x_truth ** 2 + xi_y_truth ** 2)

    for ii in range(int(maxiter)):
        compute_residuals = (residuals_interval < np.inf) and (
            np.mod(ii, residuals_interval) == 0
        )

        # Fourier-magnitude projection.
        frames, mse_data = Project_data(
            frames, frames_data, compute_residuals=compute_residuals
        )

        frames_old = frames + 0.0

        # Position update on cadence, after warmup.
        if (
            ii >= position_start
            and position_every
            and np.mod(ii, position_every) == 0
        ):
            # Multiplex coarse pre-localization (Section III.A): convex,
            # basin-free integer localization from the *consensus* object,
            # catching large shifts (> the Taylor/exact capture basin). Run as a
            # coarse-to-fine BOOTSTRAP over a few rounds (every prelocalize_every
            # iters): each round localizes a fraction of frames, migrates their
            # integers into the map (probe<->map transpose+invert), and the
            # sharpened consensus lets the next round catch more -- breaking the
            # positions<->image chicken-and-egg that a single shot can't. Only
            # migrated frames have xi reset; the <0.5px residual is left to the
            # solver below, and the TOTAL-shift accounting picks up the migration.
            if (prelocalize and prelocalize_count < prelocalize_rounds
                    and np.mod(ii - position_start, prelocalize_every) == 0):
                cx, cy = multiplex_prelocalize(frames_data, img, dp, mapid,
                                               radius=prelocalize_radius)
                moved = (cx != 0) | (cy != 0)
                if float(xp.sum(moved)) > 0:
                    translations_y = translations_y - cx
                    translations_x = translations_x - cy
                    mapid = map_frames(translations_x, translations_y, nx, ny, Nx, Ny)
                    xi_x = xp.where(moved, 0.0, xi_x)
                    xi_y = xp.where(moved, 0.0, xi_y)
                    if method == "coupled":
                        plan = position_plan(translations_x, translations_y,
                                             nframes, nx, ny, Nx, Ny)
                    probe_stack, normalization = probe_and_norm(xi_x, xi_y)
                prelocalize_count += 1

            if method == "coupled":
                xi_x, xi_y = position_solve_coupled(
                    frames, dp, img, mapid, Nx, Ny, xi_x, xi_y, plan,
                    reg=reg, max_step=max_step, backend=backend,
                )
            else:
                xi_x, xi_y = position_solve_diag(
                    frames, dp, img, mapid, Nx, Ny, xi_x, xi_y,
                    reg=reg, max_step=max_step, method=diag_method,
                )

            # Map re-planning: migrate the integer part of the shift into the
            # integer frame map, keeping only the <0.5px residual in the Taylor
            # model (which is all it can represent). Verified equivalence (in
            # intensity, for an apodized probe): a probe xi_x shift of +n equals
            # a map shift of translations_y by -n (the relative shift inverts,
            # and x<->y are transposed by map_frames' 'xy' meshgrid).
            if replan:
                n_x = xp.round(xi_x)
                n_y = xp.round(xi_y)
                if float(xp.max(xp.abs(n_x))) or float(xp.max(xp.abs(n_y))):
                    translations_y = translations_y - n_x
                    translations_x = translations_x - n_y
                    xi_x = xi_x - n_x
                    xi_y = xi_y - n_y
                    mapid = map_frames(translations_x, translations_y, nx, ny, Nx, Ny)
                    if method == "coupled":
                        plan = position_plan(translations_x, translations_y,
                                             nframes, nx, ny, Nx, Ny)

            probe_stack, normalization = probe_and_norm(xi_x, xi_y)

        # Overlap (image) projection with the current per-frame probe.
        # Floor the normalization at uncovered pixels (a scan with a border,
        # rather than a periodic tiling, leaves normalization=0 there) so the
        # division stays finite; covered pixels are unchanged.
        nrm = xp.where(
            xp.abs(normalization) < 1e-6 * float(xp.max(xp.abs(normalization))),
            1.0, normalization,
        )
        img = Overlapc(frames * xp.conj(probe_stack), Nx, Ny, mapid) / nrm

        # Split back to frames and re-illuminate.
        frames = Splitc(img, mapid) * probe_stack

        if compute_residuals:
            k = ii // residuals_interval
            residuals[k, 1] = mse_data
            residuals[k, 2] = xp.linalg.norm(frames - frames_old)
            if img_truth is not None:
                residuals[k, 0] = mse_calc(img_truth, img)
            if track_xi:
                # total recovered shift = residual xi + integers in the map
                tot_x = xi_x + (translations_y0 - translations_y)
                tot_y = xi_y + (translations_x0 - translations_x)
                residuals[k, 3] = xp.sqrt(
                    xp.sum((tot_x - xi_x_truth) ** 2 + (tot_y - xi_y_truth) ** 2)
                    / den_xi
                )

    if img_truth is not None and nrm_truth != 0:
        residuals[:, 0] /= nrm_truth

    # return the TOTAL recovered shift (residual + migrated integers)
    xi_x_total = xi_x + (translations_y0 - translations_y)
    xi_y_total = xi_y + (translations_x0 - translations_x)
    return img, frames, xi_x_total, xi_y_total, residuals
