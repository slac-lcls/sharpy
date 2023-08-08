import pkg_resources
import cupy as cp

# hardcodeds
nthreads = 256

# ==================
# load cuda kernels
# ==================
split_raw_kernel = None
overlap_raw_kernel = None
gram_raw_kernel = None

resource_package = __name__

# split kernel (image, [frames,0], translations, [illumination,0])
# if split_raw_kernel is None:
resource_path = '/'.join(('src', 'split.cu'))
file_name = pkg_resources.resource_filename(resource_package, resource_path)
with open(file_name, 'r') as myfile:
     split_raw_kernel = myfile.read()

# overlap kenrel (image, frames, translations, illumination)
#if overlap_raw_kernel is None:
resource_path = '/'.join(('src', 'overlap.cu'))
file_name = pkg_resources.resource_filename(resource_package, resource_path)
with open(file_name, 'r') as myfile:
     overlap_raw_kernel = myfile.read()

# Gramian kernel
# if gram_raw_kernel == None:
resource_path = '/'.join(('src','zQQz.cu'))
file_name = pkg_resources.resource_filename(resource_package, resource_path)
with open(file_name, 'r') as myfile:
        gram_raw_kernel = myfile.read()
        

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
    cp.RawKernel(overlap_raw_kernel, "Overlap") \
        ((int(translations.size),), (int(nthreads),), \
        (image, frames, translations, illumination, image.shape[0], image.shape[1], translations.size, illumination.shape[0], illumination.shape[1]))

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

    cp.RawKernel(split_raw_kernel, "Split") \
        ((int(nblocks),), (int(nthreads),), \
        (image, frames, translations, illumination, int(image.shape[0]), int(image.shape[1]), int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2]), int(nthreads), int(tsize)))

    return frames

def Gramiam_calc_cuda(frames, illumination,normalization,frames_norm, gram_calc):
    col = plan['col']
    row = plan['row']
    dx = plan['dx']
    dy = plan['dy']
    bw = plan['bw']
    frame_width=plan['nx']
    frame_height=plan['ny']
    nnz = len(col)
    #value = xp.zeros(nnz,dtype = xp.complex64)
    value = plan['value'] #xp.zeros(nnz,dtype = xp.complex64)

    #frame_height = frames.shape[1]
    #frame_width = frames.shape[2]
    nthreads = 128
    nblocks = nnz
  
    t0 = timer()
    
    value = gram_calc(frames,frames_norm, illumination, normalization)

    

    #print('out',value)
    #print('value by Cuda',value)
    # Try cupy sparse
    timers['Gramiam'] = timer() - t0
    
    nframes = frames.shape[0]
    H = sparse.coo_matrix((value.ravel(), (col, row)), shape=(nframes, nframes))
    H += sparse.triu(H, k=1).conj().T
    H = H.tocsr()
    timers['Gramiam_completion']=timer() - t0
    return H



def Gramian_plan(translations_x, translations_y, nframes, nx, ny, Nx, Ny, bw=0):
    # embed all geometric parameters into the gramiam function
    # calculates the difference of the coordinates between all frames

    dx = translations_x.ravel(order="F").reshape(nframes, 1)
    dy = translations_y.ravel(order="F").reshape(nframes, 1)
    dx = xp.subtract(dx, xp.transpose(dx))
    dy = xp.subtract(dy, xp.transpose(dy))

    # calculates the wrapping effect for a period boundary
    dx = -(dx + Nx * ((dx < (-Nx / 2)).astype(int) - (dx > (Nx / 2)).astype(int)))
    dy = -(dy + Ny * ((dy < (-Ny / 2)).astype(int) - (dy > (Ny / 2)).astype(int)))

    # find the frames idex that overlaps (only the lower tril)
    col, row = xp.where(xp.tril((abs(dy) < nx - 2 * bw) * (abs(dx) < ny - 2 * bw)).T)
    # maybe we need triu and remove the transpose ? 
    
    # keep only the useful dx,dy
    dx = dx[row,col]
    dy = dy[row,col]
    
    nnz = col.size
       
    val=xp.empty((nnz,1),dtype=xp.complex64)

    plan = {"col": col.astype(int), "row": row.astype(int), "dx": dx, "dy": dy,"val": val, "bw": bw, "frame_width": nx, "frame_height": ny}

    # we cam pass the function instead of the plan
    def gram_calc(frames,frames_norm, illumination, normalization, value=val):
        cp.RawKernel(zQQz_raw_kernel,"dotp",jitify=True)\
        ((int(nblocks),),(int(nthreads),), \
        (value,frames,frames_norm, illumination, normalization,col,row,dx,dy,bw,nnz, nx, ny))
        return value


    '''    
    gram_calc= lambda val,frames,frames_norm, illumination, normalization : 
         cp.RawKernel(zQQz_raw_kernel,"dotp",jitify=True)\
        ((int(nblocks),),(int(nthreads),), \
        (value,frames,frames_norm, illumination, normalization,col,row,dx,dy,bw,nnz, nx, ny))
    '''

    return gram_calc


#    lambda Gramiam1
#    H=Gramiam(nframes,framesl,framesr,col,row,nx,ny,dx,dy)


