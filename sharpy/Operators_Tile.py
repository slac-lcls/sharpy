"""
Ptycho operators

"""
#!/cds/home/y/yn754/anaconda3/envs/sharpy-env/bin/python
from timeit import default_timer as timer
from tqdm import tqdm
import numpy as np
import scipy as sp

import math
import numpy_groupies

import config
import pkg_resources
import bisect
from Operators import map_frames, Gramiam_plan, mapu2all,Precondition_calc,Project_data,synchronize_frames_c
from Operators import mse_calc
GPU = config.GPU

if GPU:
    import cupy as cp
    xp = cp
    import cupyx.scipy.sparse as sparse
    from fft_plan import fft2, ifft2
    import cupyx as cpx
    import cupy as cp
    mempool = cp.get_default_memory_pool()
    
    resource_package = __name__

    zQQz_raw_kernel = None
    if zQQz_raw_kernel == None:
        resource_path = '/'.join(('src','zQQz.cu'))
        file_name = pkg_resources.resource_filename(resource_package, resource_path)
        with open(file_name, 'r') as myfile:
            zQQz_raw_kernel = myfile.read()

else:
    xp = np
    import scipy.sparse as sparse
    from scipy.fftpack import fft2, ifft2


# import multiprocessing as mp

# timers: keep track of timing for different operators
from timeit import default_timer as timer

timers = {
    "Overlap": 0,
    "Split": 0,
    "Prox_data": 0,
    "Data_prox_tot": 0,
    "Propagate": 0,
    "mse_data": 0,
    "Gramiam": 0,
    "Gramiam_completion": 0,
    "Precondition": 0,
    "Eigensolver": 0,
    "Sync_setup": 0,
    "fd": 0,
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
    "overlap Tiles":0,
}
        
def get_times():
    return timers


def reset_times():
    for keys in timers:
        timers[keys] = 0


def normalize_times():
    tot = 0
    for keys in timers:
        tot += timers[keys]
    if tot != 0:
        for keys in timers:
            timers[keys] /= tot
    return tot


def make_tiles(Nx,Ny,NTx,NTy,translations_x,translations_y):    
    ix=xp.unique(translations_x).shape[0]
    iy=xp.unique(translations_y).shape[0]
    shift_Tx=xp.unique(translations_x)[::(ix//NTx)][0:NTx]
    shift_Tx=xp.append(shift_Tx,max(translations_x)+1) 
    shift_Ty=xp.unique(translations_y)[::(iy//NTy)][0:NTy]
    shift_Ty=xp.append(shift_Ty,max(translations_y)+1) 
    return shift_Tx.astype(int), shift_Ty.astype(int)

def pad_tiles(tiles,tiles_sizes,all_ones):
    #tiles input is a list, output is a 3d array
    Ntiles = len(tiles)
    #ntx_max = xp.maximum(tiles_sizes[:,1])
    ntx_max = xp.amax(tiles_sizes[:,1]).astype(int)
    #nty_max = xp.maximum(tiles_sizes[:,0])
    nty_max = xp.amax(tiles_sizes[:,1]).astype(int)
    
    padded_tiles = xp.zeros((Ntiles,int(nty_max),int(ntx_max)),dtype = xp.complex64)
    
    print(padded_tiles.shape)
    for j in range(Ntiles):
        if all_ones:
            print(type(tiles[j]*0 + 1))
            padded_tiles[j,:,:] = xp.pad(tiles[j]*0 + 1, ((0, int(nty_max - tiles_sizes[j,0])), (0, int(ntx_max - tiles_sizes[j,1]))), mode='constant',constant_values=0)
            #padded_tiles[j,:,:] = xp.pad(tiles[j]*0 + 1, ((0,2), (0, 0)), mode='constant',constant_values=0)
                
        else:
            print('Hello'*3, type(tiles[j]),tiles[j].shape)
            padded_tiles[j,:,:] = xp.pad(tiles[j]+0, ((0, int(nty_max - tiles_sizes[j,0])), (0,int(ntx_max - tiles_sizes[j,1]))), mode='constant', constant_values=0)
    return padded_tiles


def group_frames(translations_x,translations_y,shift_Tx,shift_Ty):
    #find the interval for which the frames lie in
    find_x=lambda x: bisect.bisect_right(shift_Tx, x)
    find_y=lambda y: bisect.bisect_right(shift_Ty, y)
    
    grouped_x=xp.array([find_x(i) for i in translations_x])
    grouped_y=xp.array([find_y(i) for i in translations_y])
    grouped=(grouped_x-1)+(grouped_y-1)*(np.shape(shift_Tx)[0]-1)  
    #groupid=(grouped_x-1)+(grouped_y-1)*(np.shape(shift_Tx)[0])  
    return grouped

def map_tiles(translations_x, translations_y, nx, ny, Nx, Ny,NTx,NTy,Tiles_plan):
    '''
    retunrs mapid for tiles
    '''
    mapid = map_frames(translations_x, translations_y, nx, ny, Nx, Ny)
    groupid = Tiles_plan['groupid']
    
    #mapid_tiles = []
    tiles_sizes = xp.zeros((NTx*NTy,2),dtype = int)
    
    
    for j in range(NTx * NTy):
        translations_xi = translations_x[groupid == j]
        translations_yi = translations_y[groupid == j]
        Nxi = (xp.max(translations_xi)-xp.min(translations_xi) + nx ).astype(int)
        Nyi = (xp.max(translations_yi)-xp.min(translations_yi) + ny ).astype(int)
        print('NXI',Nxi.dtype)
        #mapidi = mapid[groupid == j]
        
        #Nxi=shift_Tx[j%NTx+1]-shift_Tx[j%NTx]+nx
        #Nyi=shift_Ty[j//NTx+1]-shift_Ty[j//NTx]+ny
     
        #get the shift of tile
        #dxi=shift_Tx[j%NTx]
        #dyi=shift_Ty[j//NTx]
     
        #mapidi={'Nxi':Nxi,'Nyi':Nyi,'mapidi':mapidi}
    
        #mapid_tiles.append(mapidi)
        tiles_sizes[j,0] = Nyi
        tiles_sizes[j,1] = Nxi
    #mapid_tiles.append(mapidi)
    #mapid_tiles = xp.concatenate(mapid_tiles, axis=0)
    #return mapid_tiles, tiles_sizes
    return tiles_sizes

def mapid_tiles_concat_c(mapid_tiles):
    mapid_tiles_concat = []
    for mapidi in mapid_tiles:
        mapid_tiles_concat.append(mapidi.ravel())
    #return xp.asarray(mapid_tiles_concat)
    return xp.asarray(mapid_tiles_concat).ravel() #may not have same size


def Tile_plan_c(NTx,NTy,translations_x, translations_y, nx, ny, Nx, Ny):

    shift_Tx, shift_Ty = make_tiles(Nx,Ny,NTx,NTy,translations_x,translations_y)
    groupid = group_frames(translations_x,translations_y,shift_Tx,shift_Ty)
    

    Tiles_plan = {'shift_Tx':shift_Tx,'shift_Ty':shift_Ty,'groupid':groupid}
    
    return Tiles_plan

'''
def Gplan_sub0(Gplan,groupid):
    Gplan_tile = []
    for j in range(len(xp.unique(groupid))):
        grouped = xp.where(groupid==j)
        idxi=xp.isin(Gplan["col"], grouped) 
        idyi=xp.isin(Gplan["row"],grouped) 
        mask=idxi & idyi
        
        #extract the sub-plan
        plani = {"col": Gplan['col'][mask], "row": Gplan['row'][mask], "dd": Gplan['dd'][mask], "val": Gplan['val'][mask], "bw": Gplan['bw']}
        
        Gplan_tile.append(plani)
    return Gplan_tile
'''

#Gplan for sync within a tile
def Gplan_sub(groupid,translations_x,translations_y,nx,ny,Nx,Ny,bw):
    
    Gplan_tile = []
    for j in range(len(xp.unique(groupid))):
        mask = xp.where(groupid==j)[0]
        nframesintile = len(mask)
        Gplani = Gramiam_plan(translations_x[mask] + 0, translations_y[mask] + 0, nframesintile, nx, ny, Nx, Ny, bw)
        
        Gplan_tile.append(Gplani)
    
    return Gplan_tile

#gramiam plan to sync between tiles, maybe put into the original Gramiam_plan
#here ntx and nty are vectors that store the sizes of the tiles
def Gplan_tile_c0(translations_tx, translations_ty, ntiles, ntx, nty, Nx, Ny, btw=0):
    
    t0 = timer()
    # embed all geometric parameters into the gramiam function
    # calculates the difference of the coordinates between all tiles
    dtx = translations_tx.ravel(order="F").reshape(ntiles, 1)
    dty = translations_ty.ravel(order="F").reshape(ntiles, 1)
    dtx = xp.subtract(dtx, xp.transpose(dtx))
    dty = xp.subtract(dty, xp.transpose(dty))
    
    # calculates the difference in sizes between all tiles
    dsx = ntx.ravel(order="F").reshape(ntiles, 1)
    dsy = nty.ravel(order="F").reshape(ntiles, 1)
    dsx0 = xp.subtract(dsx, xp.transpose(dsx))
    dsx1 = xp.add(dsx, xp.transpose(dsx))
    dsy0 = xp.subtract(dsy, xp.transpose(dsy))
    dsy1 = xp.add(dsy, xp.transpose(dsy))

    # calculates the wrapping effect for a period boundary
    dtx = -(dtx + Nx * ((dtx < (-Nx / 2)).astype(int) - (dtx > (Nx / 2)).astype(int)))
    dty = -(dty + Ny * ((dty < (-Ny / 2)).astype(int) - (dty > (Ny / 2)).astype(int)))

    # find the tiles idex that overlaps
    # special care with tiles with different sizes
    row, col = xp.where((abs(dty) < (1/2 * xp.sign(dty) * dsx0 + 1/2 * dsx1 - 2 * btw)) \
                                * (abs(dtx) < 1/2 * xp.sign(dtx) * dsy0 + 1/2 * dsy1 - 2 * btw))
    print(row,col)
    # complete matrix using only values for the triu part
    val2H = mapu2all(row, col , ntiles) # why are col-row swapped? Maybe the .T
    
    #
    col, row = xp.where(xp.triu((abs(dty) < (1/2 * xp.sign(dty) * dsx0 + 1/2 * dsx1 - 2 * btw)) \
                                * (abs(dtx) < 1/2 * xp.sign(dtx) * dsy0 + 1/2 * dsy1 - 2 * btw)))

    # complex displacement (x, 1j y)
    # why are col-row swapped?
    dtx = dtx[row, col] 
    dty = dty[row, col]
   
 
    nnz = col.size
    val=xp.zeros((nnz,1),dtype=xp.complex64)
   
    # plan = {"col": xp.ascontiguousarray(col), "row": xp.ascontiguousarray(row), "dx": dx, "dy": dy,"val": val, "bw": bw}
    plan = {"col": col.astype(int), "row": row.astype(int), "dtx": dtx, "dty": dty,"val": val, "btw": btw,"val2H":val2H,"gram_calc":None}
        
    # we can pass the function instead of the plan
    if GPU: 
        nthreads = 128
        nblocks = nnz 
        def gram_calc(frames,frames_norm, illumination, normalization, value=val):
            
            ntx = frames.shape[1]
            nty = frames.shape[2]
            print(frames.shape,frames_norm.shape,illumination,normalization.shape,col,row,dtx,dty,btw,nnz,ntx,nty)
            print(frames_norm)
            cp.RawKernel(zQQz_raw_kernel,"dotp",jitify=True)\
            ((int(nblocks),),(int(nthreads),), \
            (value,frames,frames_norm, illumination, normalization,col.astype(int),row.astype(int),dtx,dty,btw,nnz, ntx, nty))
            
            return value
        
        plan["gram_calc"] = gram_calc
        
    return plan


def Gplan_tile_c(ntiles, nx, ny, Nx, Ny, btw=0):
    
    t0 = timer()
    
    # find the tiles idex that overlaps
    row, col = xp.where(xp.ones((ntiles,ntiles),dtype = bool))
  
    # complete matrix using only values for the triu part
    val2H = mapu2all(row, col , ntiles) # why are col-row swapped? Maybe the .T
    
    #
    col, row = xp.where(xp.triu(xp.ones((ntiles,ntiles),dtype = bool)))
   
    # complex displacement (x, 1j y)
    # why are col-row swapped?
    dtx = xp.zeros((col.size),dtype = int) 
    dty = dtx
   
    nnz = col.size
    val=xp.zeros((nnz,1),dtype=xp.complex64)
   
    # plan = {"col": xp.ascontiguousarray(col), "row": xp.ascontiguousarray(row), "dx": dx, "dy": dy,"val": val, "bw": bw}
    plan = {"col": col.astype(int), "row": row.astype(int), "dtx": dtx, "dty": dty,"val": val, "btw": btw,"val2H":val2H,"gram_calc":None}
        
    # we can pass the function instead of the plan
    if GPU: 
        nthreads = 128
        nblocks = nnz 
        def gram_calc(frames,frames_norm, illumination, normalization, value=val +0):
            
            ntx = frames.shape[1]
            nty = frames.shape[2]
            cp.RawKernel(zQQz_raw_kernel,"dotp",jitify=True)\
            ((int(nblocks),),(int(nthreads),), \
            (value,frames,frames_norm, illumination, normalization,col.astype(int),row.astype(int),dtx,dty,btw,nnz, ntx, nty))
            
            return value
        
        plan["gram_calc"] = gram_calc
        
    return plan

def Alternating_projections_tiles(
    sync,
    img,
    NTx,
    NTy,
    Gplan_tiles,
    Gplan_Tiles,
    Gplan,
    groupid,
    illumination,
    translations_x,
    translations_y,
    translations_tx,
    translations_ty,
    overlap_cuda,
    split_cuda,
    frames_data,
    refine_illumination,
    maxiter,
    normalization,
    img_truth,
    residuals_interval,
    num_iter,
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
 
    compute_residuals = True
    # we need the frames norm to normalize
    frames_norm_sum = xp.linalg.norm(xp.sqrt(frames_data))
    # renormalize the norm for the ifft2 space
    frames_norm_r = frames_norm_sum / xp.sqrt(xp.prod(xp.array(frames_data.shape[-2:])))
    translations = (translations_x + 1j * translations_y).astype(xp.complex64)
    translations_t = (translations_tx + 1j * translations_ty).astype(xp.complex64)
     

    # get the frames from the inital image
    if GPU:
        frames = xp.zeros(frames_data.shape,dtype = xp.complex64)
        split_cuda(img, frames, translations, illumination)
    else:
        frames = Illuminate_frames(Split(img), illumination)

    nresiduals = int(np.ceil(maxiter / residuals_interval))

    # residuals = xp.zeros((maxiter,3),dtype=xp.float32)
    residuals = xp.zeros((nresiduals, 4), dtype=xp.float32)

    if type(img_truth) != type(None):
        nrm_truth = xp.linalg.norm(img_truth)
    if type(normalization) == type(None):
        nframes = xp.shape(frames_data)[0]

        #normalization for within each tile
        normalization_tiles = xp.zeros((NTx * NTy,img_truth.shape[0],img_truth.shape[1]),dtype = xp.complex64)
        inormalization_split_tiles = [None] * (NTx * NTy) 
        
        reg = 1e-8
        
        import matplotlib.pyplot as plt
        for j in range(NTx * NTy):
          
            '''
            #the normalization is of sub-tile size. However, errornous in shift?
            normalization_tiles[j] = xp.zeros((int(tiles_sizes[j,0]),int(tiles_sizes[j,1])),dtype = xp.complex64)
            overlap_cuda(normalization_tiles[j], 0,translations[groupid == j] , illumination) 
            '''
            normalization_tilesi = xp.zeros(img_truth.shape,dtype = xp.complex64)
            translationsj = translations[groupid == j] + 0 
            overlap_cuda(normalization_tilesi, 0, translationsj , illumination) 
            
        
            normalization_tiles[j,:,:] = normalization_tilesi
            
            nframesintile = xp.sum(groupid == j)

            inormalization_split_tilesi = xp.zeros((int(nframesintile),frames_data.shape[1],frames_data.shape[2]),dtype = xp.complex64) 
            
            split_cuda(1/(normalization_tilesi+reg),inormalization_split_tilesi,translationsj , 0)
            
            inormalization_split_tiles[j] = inormalization_split_tilesi
            
            
            '''
            if j == 0:
                print(normalization_tiles[j].dtype, type(normalization_tiles[j]), normalization_tiles[j].shape)
                print(inormalization_split_tiles[j].dtype, type(inormalization_split_tiles[j]), inormalization_split_tiles[j].shape)
            '''
          
        #normalization for between tiles
        illumination_one = abs(xp.sign(normalization_tiles)).astype(xp.complex64) #get an all one Tiles
        
        normalization_overall = xp.sum(illumination_one,axis=0) 
        normalization_overall.astype(xp.complex64)   
        
        inormalization_split_overall = xp.zeros((NTx * NTy,img_truth.shape[0],img_truth.shape[1]),dtype = xp.complex64) #all ones?
        split_cuda(1/(normalization_overall + reg),inormalization_split_overall,translations_t * 0  , 0)
        #inormalization_split_overall = 1/(normalization_overall + reg) 
        #Need implementaion. values are same. One 2D and one 3D. Change the kernel.
        
        plt.imshow(abs(inormalization_split_overall[0].get()))
        plt.colorbar()
        plt.show()
    
    #frames_norm = Precondition_calc(frames, bw=Gramiam['bw'])
    frames_norm = Precondition_calc(frames, bw=Gplan_tiles[0]['bw']) #check!
  
    Tile = xp.zeros((NTx*NTy,img_truth.shape[0],img_truth.shape[1]),dtype = xp.complex64)
    
    for ii in tqdm(range(maxiter)):

    
        frames, mse_data = Project_data(
            frames, frames_data, compute_residuals=compute_residuals
        )
      
        if compute_residuals:
            residuals[ii // residuals_interval, 1] = mse_data
        

        t0 = timer()

        frames_old = frames + 0.0  # make a copy
        

        ####################
        # here goes the synchronization within tiles
        if sync==True:
            t0 = timer()
            for j in range(NTx * NTy):
                framesi = frames[groupid == j] + 0 
                translationsj = translations[groupid == j] + 0 
                frames_normj = frames_norm[groupid == j] + 0 
             
                omega=synchronize_frames_c(framesi, illumination +0, frames_normj, inormalization_split_tiles[j]+0, Gplan_tiles[j], Gplan_tiles[j]['bw'],num_iter)
                
                framesi = framesi * omega
                imagei = xp.zeros(img_truth.shape,dtype = xp.complex64)
                #overlap_cuda(imagei, framesi,translations[groupid == j] , illumination)
                overlap_cuda(imagei, framesi,translationsj , illumination)
                Tile[j,:,:] = imagei / (normalization_tiles[j,:,:]+reg) #normalize here or later?
                
            timers["Sync"] += timer() - t0
                        
            #padded_Tile = pad_tiles(Tile ,tiles_sizes,all_ones=False)
            
            '''
            if ii <= 3:
                for jj in range(NTx * NTy):
                    print(Tile[jj,:,:])
                    print(img_truth)
                    plt.imshow(abs(Tile[jj,:,:].get()))
                    plt.show()      
                    plt.imshow(abs(img_truth).get())
                    plt.show()
                    plt.imshow(abs(Tile[jj,:,:]-img_truth).get())
                    plt.show()
           '''        
        ##################
        #here goes synchronization between tiles
        Tiles_norm = Precondition_calc(Tile, Gplan_tiles[0]['bw'])
        '''
        print('checking here',padded_Tile.shape,type(padded_Tile),padded_Tile.dtype,Tiles_norm.shape,type(Tiles_norm),Tiles_norm.dtype,type(inormalization_split_overall),inormalization_split_overall.dtype)
        print(Tiles_norm)
        print(Gplan_Tiles)
        '''
            
        if GPU:
            if NTx* NTy == 1:
                omega_tiles = 1
            else:
                #inormalization_split_overall is be the same for all tiles. But synchorinize_frames_c only takes 3D input
                omega_tiles=synchronize_frames_c(Tile , 0, Tiles_norm, inormalization_split_overall, Gplan_Tiles,num_iter) 
                #print('!!!!!!!!!!',omega_tiles)
                  
        else:
            omega_tiles=synchronize_frames_c(padded_Tile, 1 + 0j, Tiles_norm, inormalization_split_overall, Gplan_Tiles,num_iter)
              
            #padded_Tile = padded_Tile * omega_tiles
            

        #print('HELLOOO',omega_tiles)
        img1 = xp.sum(Tile * omega_tiles,axis=0)
        img = xp.sum(Tile,axis=0)
        plt.imshow(abs(img.get()))
        #plt.show()
        img = img / (normalization_overall + reg)
        img1 = img1 / (normalization_overall + reg)
        plt.imshow(abs(img.get()))
        #plt.show()
        plt.imshow(abs(img1.get()))
        #plt.show()
        plt.imshow(abs((img1-img).get()))    
        #plt.show()
        

        
        frames = xp.zeros(frames_data.shape,dtype = xp.complex64)
        split_cuda(img,frames, translations,illumination)
        
            # if GPU and ii<2:
            #     print('in loop, after split memory used, and total normalized:', mempool.used_bytes()/frames_data.nbytes,mempool.total_bytes()/frames_data.nbytes )
            #     print('----')

        if compute_residuals:
            residuals[ii // residuals_interval, 2] = xp.linalg.norm(frames - frames_old)
                # residuals[ii//residuals_interval,2] = xp.inner((frames-frames_old).ravel(),(frames-frames_old).ravel())

       
        if type(img_truth) != type(None):
            t0 = timer()
            if compute_residuals:
                nmse0 = mse_calc(img_truth, img)
                residuals[ii // residuals_interval, 0] = nmse0
   
   

    residuals[:, 1] /= frames_norm_sum
    residuals[:, 2] /= frames_norm_r
    if type(img_truth) != type(None):
        residuals[:, 0] /= nrm_truth
        residuals[:,3] = 1/residuals[:,0] #|truth|/|img-truth| measures the SNR

    return img, frames, illumination, residuals