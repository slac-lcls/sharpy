"""
GPU kernel wrappers.

Thin Python wrappers around the raw CUDA kernels in src/*.cu. GPU-only --
this module imports cupy at top, so it must be imported only under
`config.GPU` (Operators.py / Solvers.py guard their imports accordingly).
See the Operators.py module docstring for the Gramian / braket / zQQz
naming glossary.

Key wrappers
------------
  split_cuda(image, frames, ...)     image -> frames   (src/split.cu)
  overlap_cuda(image, frames, ...)   frames -> image   (src/overlap.cu)
  Gramian_plan(...)                  precompute overlap geometry and a
                                     `gram_calc` closure that launches the
                                     zQQz Gramian kernel (src/zQQz.cu)
  refine_illumination_cuda(...)      pairwise illumination refinement
                                     (src/refine_illumination.cu)

The Gramian assembly itself (val -> sparse H) lives in
Operators.Gramiam_calc_cuda via plan["val2H"].
"""

import os
import cupy as cp
import cupyx.scipy.sparse as sparse

# hardcodeds
nthreads = 256

_src = os.path.join(os.path.dirname(__file__), 'src')

# ==================
# load cuda kernels
# ==================
with open(os.path.join(_src, 'split.cu'), 'r') as f:
    split_raw_kernel = f.read()

with open(os.path.join(_src, 'overlap.cu'), 'r') as f:
    overlap_raw_kernel = f.read()

with open(os.path.join(_src, 'zQQz.cu'), 'r') as f:
    gram_raw_kernel = f.read()

with open(os.path.join(_src, 'refine_illumination.cu'), 'r') as f:
    refine_illum_raw_kernel = f.read()
        
# ==================
# wrap cuda kernels
# ==================

#frames_shape parameter is needed because frames can be 0 when we overlap only the illumnation, 
#in this case frames_shape tracks the actual size of the stack of frames
#def overlap_cuda(image, frames, translations, illumination, frames_shape):

def overlap_cuda(image, frames, translations, illumination):
    '''
    img_y = image.shape[0]
    img_x = image.shape[1]
    

    frames_z = translations.size
    if illumination:
       frames_y = illumination.shape[0]
       frames_x = illumination.shape[1]
    else:
       frames_z = translations.size
       frames_y = frames.shape[1]
       frames_x = frames.shape[2] 
    '''
    nthreads = 256 
    
    if type(image)== type(None):
        image = xp.zeros((1)) #--?
    # we could verify that frames illumination and translations shapes are consistent
    
    if type(frames) == int:
        cp.RawKernel(overlap_raw_kernel, "Overlap") \
        ((int(translations.size),), (int(nthreads),), \
        (image, frames, cp.ascontiguousarray(translations), cp.ascontiguousarray(illumination), image.shape[0], image.shape[1], translations.size, illumination.shape[0], illumination.shape[1]))
    else:
        cp.RawKernel(overlap_raw_kernel, "Overlap") \
        ((int(translations.size),), (int(nthreads),), \
        (image, cp.ascontiguousarray(frames), cp.ascontiguousarray(translations), cp.ascontiguousarray(illumination), image.shape[0], image.shape[1], translations.size, illumination.shape[0], illumination.shape[1]))
        

    return image

def split_cuda(image, frames, translations, illumination):
    '''
    img_y = image.shape[0]
    img_x = image.shape[1]
    frames_z = frames.shape[0]
    frames_y = frames.shape[1]
    frames_x = frames.shape[2]
    '''
    nthreads = 256  # we could test changing nthreads for performancce
    tsize = 128  # not sure why this is different from nthreads

    frames_per_block = (nthreads/tsize)
    #nblocks = (frames.shape[0]+ frames_per_block-1)/frames_per_block
    nblocks = frames.shape[0]

    if type(image) == int and type(illumination) == int:
        cp.RawKernel(split_raw_kernel, "Split") \
        ((int(nblocks),), (int(nthreads),), \
        (image, frames, cp.ascontiguousarray(translations), illumination, int(image.shape[0]), int(image.shape[1]), int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2]), int(nthreads), int(tsize)))
    elif type(image) == int and type(illumination) != int:
        cp.RawKernel(split_raw_kernel, "Split") \
        ((int(nblocks),), (int(nthreads),), \
        (image, frames, cp.ascontiguousarray(translations), cp.ascontiguousarray(illumination), int(image.shape[0]), int(image.shape[1]), int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2]), int(nthreads), int(tsize)))
    elif type(image) != int and type(illumination) == int:
        cp.RawKernel(split_raw_kernel, "Split") \
        ((int(nblocks),), (int(nthreads),), \
        (cp.ascontiguousarray(image), frames, cp.ascontiguousarray(translations), illumination, int(image.shape[0]), int(image.shape[1]), int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2]), int(nthreads), int(tsize)))
    else:
        cp.RawKernel(split_raw_kernel, "Split") \
        ((int(nblocks),), (int(nthreads),), \
        (cp.ascontiguousarray(image), frames, cp.ascontiguousarray(translations), cp.ascontiguousarray(illumination), int(image.shape[0]), int(image.shape[1]), int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2]), int(nthreads), int(tsize)))
    return frames

def Gramian_plan(translations_x, translations_y, nframes, nx, ny, Nx, Ny, bw=0):
    # embed all geometric parameters into the gramiam function
    # calculates the difference of the coordinates between all frames

    dx = translations_x.ravel(order="F").reshape(nframes, 1)
    dy = translations_y.ravel(order="F").reshape(nframes, 1)
    dx = cp.subtract(dx, cp.transpose(dx))
    dy = cp.subtract(dy, cp.transpose(dy))

    # calculates the wrapping effect for a period boundary
    dx = -(dx + Nx * ((dx < (-Nx / 2)).astype(int) - (dx > (Nx / 2)).astype(int)))
    dy = -(dy + Ny * ((dy < (-Ny / 2)).astype(int) - (dy > (Ny / 2)).astype(int)))

    # find the frames idex that overlaps (only the lower tril)
    col, row = cp.where(cp.tril((abs(dy) < nx - 2 * bw) * (abs(dx) < ny - 2 * bw)).T)
    # maybe we need triu and remove the transpose ?

    # keep only the useful dx,dy
    dx = dx[row,col]
    dy = dy[row,col]

    nnz = col.size
    nblocks = nnz

    val=cp.empty((nnz,1),dtype=cp.complex64)

    plan = {"col": col.astype(int), "row": row.astype(int), "dx": dx, "dy": dy,"val": val, "bw": bw, "frame_width": nx, "frame_height": ny}

    # we cam pass the function instead of the plan
    def gram_calc(frames,frames_norm, illumination, normalization, value=val):
        cp.RawKernel(gram_raw_kernel,"dotp",jitify=True,options=("--std=c++17",))\
        ((int(nblocks),),(int(nthreads),), \
        (value,frames,frames_norm, illumination, normalization,col,row,dx,dy,bw,nnz, nx, ny))
        return value


    '''    
    gram_calc= lambda val,frames,frames_norm, illumination, normalization : 
         cp.RawKernel(zQQz_raw_kernel,"dotp",jitify=True,options=("--std=c++17",))\
        ((int(nblocks),),(int(nthreads),), \
        (value,frames,frames_norm, illumination, normalization,col,row,dx,dy,bw,nnz, nx, ny))
    '''

    return gram_calc


#    lambda Gramiam1
#    H=Gramiam(nframes,framesl,framesr,col,row,nx,ny,dx,dy)


def refine_illumination_cuda(frames,normalization, plan):
    col = plan['col']
    row = plan['row']
    dx = plan['dx']
    dy = plan['dy']
    bw = plan['bw']
    
    nnz = len(col)
    nblocks = nnz
    
    #initilize matrix
    A = cp.zeros((frames.shape[1]*frames.shape[2],frames.shape[1]*frames.shape[2]), dtype = cp.complex64) 
    print(A.shape)
    
    cp.RawKernel(refine_illum_raw_kernel,"Refine_illum")\
        ((int(nblocks),),(int(nthreads),), \
        (A,cp.ascontiguousarray(frames), cp.ascontiguousarray(normalization),col,row,dx,dy,bw,nnz, int(frames.shape[1]), int(frames.shape[2])))
   
    print('is there nan value in A',cp.any(cp.isnan(A)))
    A = sparse.coo_matrix(A)
    A += sparse.triu(A, k=1).conj().T
    A = A.tocsr()
 
    return A


