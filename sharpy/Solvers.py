import numpy as np
from timeit import default_timer as timer
from Operators import (
    make_probe,
    Illuminate_frames,
    Project_data,
    synchronize_frames_c,
    mse_calc,
    Precondition_calc
)
from Operators import Replicate_frame, synchronize_illum_c,refine_illumination_pairwise,refine_illumination_function
from wrap_ops import overlap_cuda,split_cuda
import matplotlib.pyplot as plt
from tqdm import tqdm
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
    