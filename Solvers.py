import numpy as np
from timeit import default_timer as timer
from Operators import Illuminate_frames, Project_data, synchronize_frames_c, mse_calc


from Operators import Replicate_frame
def Alternating_projections(img, illumination, Overlap, Split, frames_data, refine_illumination = False, maxiter = 100, normalization = None, img_truth = None):
    """
    Parameters
    ----------
    img : 2d matrix
        reconstructed image.
    illumination : 2d matrix
        ukkynubatuib.
    Overlap : TYPE
        DESCRIPTION.
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


    

    # we need the frames norm to normalize
    frames_norm = np.linalg.norm(np.sqrt(frames_data))
    # renormalize the norm for the ifft2 space
    frames_norm_r= frames_norm/np.sqrt(np.prod(frames_data.shape[-2:]))
    
    
    # get the frames from the inital image
    frames = Illuminate_frames(Split(img),illumination)

    residuals = np.zeros((maxiter,3))
    
    
    if type(img_truth) != type(None):
        nrm_truth = np.linalg.norm(img_truth)
    if type(normalization) == type(None):
        nframes = np.shape(frames_data)[0]            
        normalization=Overlap(Replicate_frame(np.abs(illumination)**2,nframes)) #check

        
    refine_illumination = 1
    eps_illum = None
    eps0 = 1e-2
    
    for ii in np.arange(maxiter):
        # data projection
        frames, mse_data = Project_data(frames,frames_data)
        residuals[ii,1] = mse_data/frames_norm
        
        frames_old =frames+0. # make a copy

        ##################
        # overlap projection
        img= Overlap(Illuminate_frames(frames,np.conj(illumination)))/normalization
        
        if ii>5 and refine_illumination:
            frames_split= Split(img)
            norm_frames = np.sum(np.abs(frames_split)**2,0)
            if type(eps_illum) == type(None):
                eps_illum = np.max(norm_frames)*eps0
            
            illumination = np.sum(frames*np.conj(Split(img)) + eps0 * illumination,0)/(norm_frames+eps_illum)
            normalization=Overlap(Replicate_frame(np.abs(illumination)**2,nframes)) #check
            
        
        frames = Illuminate_frames(Split(img),illumination)

        residuals[ii,2] = np.linalg.norm(frames-frames_old)/frames_norm_r
        

        if type(img_truth) != type(None):
            nmse0=mse_calc(img_truth,img)/nrm_truth
            residuals[ii,0] = nmse0
            
    
        
    # print('time sync:',time_sync)
    return img, frames, residuals



def Alternating_projections_c(opt, img,Gramiam,frames_data, illumination, normalization, Overlap, Split, maxiter,  img_truth = None):
    
    # we need the frames norm to normalize
    frames_norm = np.linalg.norm(np.sqrt(frames_data))
    # renormalize the norm for the ifft2 space
    frames_norm_r= frames_norm/np.sqrt(np.prod(frames_data.shape[-2:]))
    
    
    # get the frames from the inital image
    frames = Illuminate_frames(Split(img),illumination)
    inormalization_split = Split(1/normalization)
    time_sync = 0 

    
    residuals = np.zeros((maxiter,3))
    if type(img_truth) != type(None):
        nrm_truth = np.linalg.norm(img_truth)
        
    for ii in np.arange(maxiter):
        # data projection
        frames, mse_data = Project_data(frames,frames_data)
        residuals[ii,1] = mse_data/frames_norm
        
        frames_old =frames+0. # make a copy
        ####################
        # here goes the synchronization
        if opt==True:
            time0 = timer()
            omega=synchronize_frames_c(frames, illumination, inormalization_split, Gramiam)
            frames=frames*omega
            time_sync += timer()-time0

        ##################
        # overlap projection
        img= Overlap(Illuminate_frames(frames,np.conj(illumination)))/normalization
        
        frames = Illuminate_frames(Split(img),illumination)

        residuals[ii,2] = np.linalg.norm(frames-frames_old)/frames_norm_r
        

        if type(img_truth) != type(None):
            nmse0=mse_calc(img_truth,img)/nrm_truth
            residuals[ii,0] = nmse0
        
    print('time sync:',time_sync)
    return img, frames, residuals

