import numpy as np
xp = np

from timeit import default_timer as timer
from Operators import Illuminate_frames, Project_data, synchronize_frames_c, mse_calc


from Operators import Replicate_frame

eps0 = 1e-2
eps_illum = None

def refine_illumination_function(img, illumination, frames, Split, Overlap, lens_mask = None):
    '''
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

    '''

    # eps_illum = None
    global eps_illum
    frames_split= Split(img)
    norm_frames = xp.mean(xp.abs(frames_split)**2,0)
    if type(eps_illum) == type(None):
        eps_illum = xp.max(norm_frames)*eps0
       
    illumination = xp.sum(frames*xp.conj(Split(img)) + eps_illum * illumination,0)/(norm_frames+eps_illum)
    
    # apply mask to illumination
    if type(lens_mask) != type(None):
        illumination = xp.fft.fft2(illumination)
        illumination *= lens_mask
        illumination = xp.fft.ifft2(illumination)
        
    
    
    normalization=Overlap(Replicate_frame(xp.abs(illumination)**2,frames_split.shape[0])) #check
    return illumination, normalization


def Alternating_projections(img, illumination, Overlap, Split, frames_data, refine_illumination = False, maxiter = 100, normalization = None, img_truth = None):
    """
    Parameters
    ----------
    img : 2d matrix
        reconstructed image.
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


    

    # we need the frames norm to normalize
    frames_norm = xp.linalg.norm(xp.sqrt(frames_data))
    # renormalize the norm for the ifft2 space
    frames_norm_r= frames_norm/xp.sqrt(xp.prod(frames_data.shape[-2:]))
    
    
    # get the frames from the inital image
    frames = Illuminate_frames(Split(img),illumination)

    residuals = xp.zeros((maxiter,3))
    
    
    if type(img_truth) != type(None):
        nrm_truth = xp.linalg.norm(img_truth)
    if type(normalization) == type(None):
        nframes = xp.shape(frames_data)[0]            
        normalization=Overlap(Replicate_frame(xp.abs(illumination)**2,nframes)) #check

        
    # refine_illumination = 1
    #eps_illum = None
    #eps0 = 1e-2
    
    for ii in xp.arange(maxiter):
        # data projection
        frames, mse_data = Project_data(frames,frames_data)
        residuals[ii,1] = mse_data/frames_norm
        
        frames_old =frames+0. # make a copy

        ##################
        # overlap projection
        img= Overlap(Illuminate_frames(frames,xp.conj(illumination)))/normalization
        
        if ii>5 and refine_illumination:
            illumination, normalization = refine_illumination_function(img, illumination, frames, Split, Overlap, lens_mask = None)
            #frames_split= Split(img)
            #norm_frames = xp.sum(xp.abs(frames_split)**2,0)
            #if type(eps_illum) == type(None):
            #    eps_illum = xp.max(norm_frames)*eps0
            
            #illumination = xp.sum(frames*xp.conj(Split(img)) + eps0 * illumination,0)/(norm_frames+eps_illum)
            #normalization=Overlap(Replicate_frame(xp.abs(illumination)**2,nframes)) #check
            
        
        frames = Illuminate_frames(Split(img),illumination)

        residuals[ii,2] = xp.linalg.norm(frames-frames_old)/frames_norm_r
        

        if type(img_truth) != type(None):
            nmse0=mse_calc(img_truth,img)/nrm_truth
            residuals[ii,0] = nmse0
            
    
        
    # print('time sync:',time_sync)
    return img, frames, residuals



def Alternating_projections_c(opt, img,Gramiam,frames_data, illumination, normalization, Overlap, Split, maxiter,  img_truth = None):
    
    # we need the frames norm to normalize
    frames_norm = xp.linalg.norm(xp.sqrt(frames_data))
    # renormalize the norm for the ifft2 space
    frames_norm_r= frames_norm/xp.sqrt(xp.prod(frames_data.shape[-2:]))
    
    
    # get the frames from the inital image
    frames = Illuminate_frames(Split(img),illumination)
    inormalization_split = Split(1/normalization)
    time_sync = 0 

    
    residuals = xp.zeros((maxiter,3))
    if type(img_truth) != type(None):
        nrm_truth = xp.linalg.norm(img_truth)
        
    for ii in xp.arange(maxiter):
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
        img= Overlap(Illuminate_frames(frames,xp.conj(illumination)))/normalization
        
        frames = Illuminate_frames(Split(img),illumination)

        residuals[ii,2] = xp.linalg.norm(frames-frames_old)/frames_norm_r
        

        if type(img_truth) != type(None):
            nmse0=mse_calc(img_truth,img)/nrm_truth
            residuals[ii,0] = nmse0
        
    print('time sync:',time_sync)
    return img, frames, residuals

