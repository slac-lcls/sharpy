"""
Ptycho operators

"""
#!/cds/home/y/yn754/anaconda3/envs/sharpy-env/bin/python

import numpy as np
import scipy as sp
import multiprocessing as mp

import math
import numpy_groupies

import config
import pkg_resources

GPU = config.GPU


if GPU:
    import cupy as cp
    xp = cp
    import cupyx.scipy.sparse as sparse
    from fft_plan import fft2, ifft2
    import cupyx as cpx
    import cupy as cp

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


def Propagate(frames):
    # simple propagation
    print('frames shape',frames.shape)
    print('frames type', type(frames))
    return fft2(frames)


def IPropagate(frames):
    # simple inverse propagation
    # return xp.fft.ifft2(frames)
    return ifft2(frames)


eps = xp.float32(1e-2)


if GPU:

    @cp.fuse(kernel_name="ProxD")
    def ProxD(x, y, eps):
        # return   x* cpx.rsqrt(((x*x.conj()).real+eps)/(y+eps))
        # return   x* cpx.rsqrt(((xp.real(x)*xp.real(x.real)+xp.imag(x)**2)+eps)/(y+eps))
        return x * cpx.rsqrt(((xp.real(x) ** 2 + xp.imag(x) ** 2) + eps) / (y + eps))

else:

    def ProxD(x, y, eps):
        # return   x* cpx.rsqrt(((x*x.conj()).real+eps)/(y+eps))
        # return   x* cpx.rsqrt(((xp.real(x)*xp.real(x.real)+xp.imag(x)**2)+eps)/(y+eps))
        return x * xp.sqrt((y + eps) / ((xp.real(x) ** 2 + xp.imag(x) ** 2) + eps))


def Project_data(frames, frames_data, compute_residuals=False):
    time00 = timer()
    time0 = time00
    # apply Fourier magnitude projections
    frames = Propagate(frames)
    timers["Propagate"] += timer() - time0

    time0 = timer()

    # compute mse
    if compute_residuals:
        mse = xp.linalg.norm(xp.abs(frames) - xp.sqrt(frames_data))
    else:
        mse = eps

    timers["mse_data"] += timer() - time0

    time0 = timer()
    # if False:
    #     fd = xp.float32(1)/(frames_data+eps)
    #     timers['fd']+=timer()-time0

    #     time0=timer()
    #     frames *= cpx.rsqrt(((frames*frames.conj()).real+eps)*fd)
    # else:
    #      frames *= xp.sqrt((frames_data+eps)/((frames*frames.conj()).real+eps))
    #      # frames *= cpx.rsqrt(((frames*frames.conj()).real+eps)/(frames_data+eps))
    # print('using proxD')
    frames = ProxD(frames, frames_data, eps)

    # frames *= xp.sqrt((frames_data+eps)/((frames*frames.conj()).real+eps))

    # frames *= xp.sqrt((frames_data+eps)/(xp.abs(frames)**2+eps))
    # frames *= xp.sqrt((frames_data+eps)/((frames.real*frames.real + frames.imag*frames.imag)+eps))
    # frames *= xp.sqrt((frames_data+eps)/((frames*frames.conj()).real+eps))

    timers["Prox_data"] += timer() - time0

    time0 = timer()
    frames = IPropagate(frames)
    timers["Propagate"] += timer() - time0
    timers["Data_prox_tot"] += timer() - time00

    return frames, mse


"""
if GPU:
    dotnorm2 = xp.ReductionKernel(
                'T x ,T x1, T y, T y1, Z zz', 'Z z',
                '(x-x1)*(y-y1)+zz*((y-y1)*(y-y1))',#'(x-y)* conj(x-y)+zz*(x*conj(x))',
                'a + b','z = a','0')
else:
    mse = xp.linalg.norm(xp.abs(frames)-xp.sqrt(frames_data))
"""


def Prox_data_r(frames, frames_data_r, compute_residuals=False):
    time00 = timer()
    time0 = time00
    # apply Fourier magnitude projections
    frames = Propagate(frames)
    timers["Propagate"] += timer() - time0

    time0 = timer()

    # compute mse
    if compute_residuals:
        mse = xp.linalg.norm(frames - cpx.rsqrt(frames_data_r))

    else:
        mse = eps

    timers["mse_data"] += timer() - time0

    time0 = timer()
    frames *= cpx.rsqrt(((frames * frames.conj()).real + eps) * (frames_data_r))

    # frames *= cpx.rsqrt(((frames.real**2+frames.imag**2).real+eps)*(frames_data_r))

    # frames *= xp.sqrt((frames_data+eps)/(xp.abs(frames)**2+eps))
    # frames *= xp.sqrt((frames_data+eps)/((frames.real*frames.real + frames.imag*frames.imag)+eps))
    # frames *= xp.sqrt((frames_data+eps)/((frames*frames.conj()).real+eps))

    timers["Prox_data"] += timer() - time0

    time0 = timer()
    frames = IPropagate(frames)
    timers["Propagate"] += timer() - time0
    timers["Data_prox_tot"] += timer() - time00

    return frames, mse


# def prox_data_plan(frames_data):
#     """
#     Parameters
#     ----------
#     frames_data : diffraction frames
#         3d matrix

#     Returns
#     -------
#     prox_data : function(frames, compute_residuals = False)
#         compute the proximal operator to frames and returns projected frames

#     """
#     if GPU:
#         if True:

#             def prox_data(frames, compute_residuals=False):
#                 frames, mse = Project_data(
#                     frames, frames_data, compute_residuals=compute_residuals
#                 )
#                 return frames, mse

#         else:
#             fdr = xp.float32(1) / (frames_data + eps)
#             #  print('type fdr:', type(fdr), 'dtype:', fdr.dtype)
#             def prox_data(frames, compute_residuals=False):
#                 frames, mse = Prox_data_r(
#                     frames, fdr, compute_residuals=compute_residuals
#                 )
#                 return frames, mse

#     else:

#         def prox_data(frames, compute_residuals=False):
#             frames, mse = Project_data(
#                 frames, frames_data, compute_residuals=compute_residuals
#             )
#             return frames, mse

#         # rox_data = lambda frames, compute : Project_data(frames, frames_data, compute_residuals = False)

#     return prox_data


def make_probe(nx, ny, r1=0.03, r2=0.06, fx=0.0, fy=0.0):
    """
    make an illumination (probe) in a (nx, ny) frame shape
    r1,r2 fractions of of the frame width
    fx,fy:  x-y quadradic fase (focus)

    """
    xi = xp.reshape(xp.arange(0, nx) - nx / 2, (nx, 1)) 

    xi = xp.fft.ifftshift(xi)

    rr = xp.sqrt(xi**2 + (xi.T) ** 2)
    r1 = r1 * nx  # define zone plate circles
    r2 = r2 * nx

    Fprobe = (rr >= r1) & (rr <= r2)

    phase = xp.exp(1j * fx * xp.pi * ((xi / nx) ** 2)) * xp.exp(
        1j * fy * xp.pi * ((xi.T / nx) ** 2)
    )

    Fprobe = Fprobe * phase

    probe = xp.fft.fftshift(xp.fft.ifft2(Fprobe))
    probe = probe / max(abs(probe).flatten())
    return probe


# close packing translations
def make_translations(Dx, Dy, nnx, nny, Nx, Ny):
    """
    make scan positions using spacing Dx,Dy, number of steps nnx, nny,
    image width Nx,Ny. The lattice is periodic with close-packing arrangement

    """
    #ix, iy = xp.meshgrid(
    #    xp.arange(0, Dx * nnx, Dx) + Nx / 2 - Dx * nnx / 2 + 1,
    #    xp.arange(0, Dy * nny, Dy) + Ny / 2 - Dy * nny / 2 + 1,
    #)
    
    ix, iy = xp.meshgrid(
        xp.arange(0, Dx * nnx, Dx) + Nx // 2 - Dx * nnx // 2 + 1,
        xp.arange(0, Dy * nny, Dy) + Ny // 2 - Dy * nny // 2 + 1,
    )
    
    xshift = math.floor(Dx / 2) * xp.mod(xp.arange(1, xp.size(ix, 1) + 1), 2)
    
    # adding shift in the x-direction to make close-packing lattice
    ix = xp.transpose(xp.add(xp.transpose(ix), xshift))
    ix = ix - xp.min(ix)
    iy = iy - xp.min(iy)

    ix = xp.reshape(ix, (nnx * nny, 1, 1))
    iy = xp.reshape(iy, (nnx * nny, 1, 1))
    ix = xp.asarray(ix)
    iy = xp.asarray(iy)
    return ix, iy


def map_frames(translations_x, translations_y, nx, ny, Nx, Ny):
    """
    return frame mapping: frames = image[mapid]
    """

    # map frames to image indices
    translations_x = xp.reshape(
        xp.transpose(translations_x), (xp.size(translations_x), 1, 1)
    )
    translations_y = xp.reshape(
        xp.transpose(translations_y), (xp.size(translations_y), 1, 1)
    )

    xframeidx, yframeidx = xp.meshgrid(xp.arange(nx), xp.arange(ny))
    # print('translations shapes:',xp.shape(translations_x),'frameidx',xp.shape(xframeidx))

    spv_x = xp.add(xframeidx, translations_x)
    spv_y = xp.add(yframeidx, translations_y)

    # enforce periodic boundaries
    mapidx = xp.mod(spv_x, Nx)
    mapidy = xp.mod(spv_y, Ny)

    mapid = xp.add(mapidx, mapidy * Nx)
    # mapid=xp.add(mapidx*Nx,mapidy)
    mapid = mapid.astype(np.uint32)

    return mapid


def Splitc(img, mapid):
    # Split an image into frames given mapping
    time0 = timer()
    frames_out = (img.ravel())[mapid]
    timers["Split"] += timer() - time0
    return frames_out


def Overlapc(frames, Nx, Ny, mapid):  # check
    # overlap frames onto an image using aggregate function
    time0 = timer()
    accum = xp.reshape(
        numpy_groupies.aggregate(mapid.ravel(), frames.ravel()), (Nx, Ny)
    )

    timers["Overlap"] += timer() - time0
    return accum


def Overlapd(frames, SS, shape):  # check
    # overlap frames onto an image using SPmV and reshape

    time0 = timer()
    output = SS * frames.ravel()
    output.shape = shape
    timers["Overlap"] += timer() - time0
    return output


#    accum = xp.reshape(numpy_groupies.aggregate(mapid.ravel(),frames.ravel()),(Nx,Ny))


def Split_Overlap_plan(translations_x, translations_y, nx, ny, Nx, Ny):
    mapid = map_frames(translations_x, translations_y, nx, ny, Nx, Ny)

    # for cupy we need a sparse matrix
    col = xp.arange(mapid.size)
    val = xp.ones((mapid.size), dtype=np.float32)
    SS = sparse.coo_matrix((val.ravel(), (mapid.ravel(), col.ravel())))
    SS = sparse.csr_matrix(SS)

    Split = lambda img: Splitc(img, mapid)
    Overlap = lambda frames: Overlapd(frames, SS, (Nx, Ny))
    return Split, Overlap


def Split_plan(translations_x, translations_y, nx, ny, Nx, Ny):
    mapid = map_frames(translations_x, translations_y, nx, ny, Nx, Ny)
    Split = lambda img: Splitc(img, mapid)
    return Split


def Overlap_plan(translations_x, translations_y, nx, ny, Nx, Ny):
    mapid = map_frames(translations_x, translations_y, nx, ny, Nx, Ny)
    Overlap = lambda frames: Overlapc(frames, Nx, Ny, mapid)
    return Overlap


def crop_center(img, cropx, cropy):
    # crop an image
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty : starty + cropy, startx : startx + cropx]


def cropmat(img, size):
    # crop an image to a given size
    left0 = math.floor((xp.size(img, 0) - size[0]) / 2)
    right0 = size[0] + math.floor((xp.size(img, 0) - size[0]) / 2)
    left1 = math.floor((xp.size(img, 1) - size[1]) / 2)
    right1 = size[1] + math.floor((xp.size(img, 1) - size[1]) / 2)
    crop_img = img[left0:right0, left1:right1]
    return crop_img


def Overlapc0(frames, Nx, Ny, mapid):

    # ret = xp.bincount(mapid, weights=frames.real)+1j*xp.bincount(mapid, weights=frames.imag)
    ret = xp.bincount(mapid.ravel(), weights=(frames.ravel()).real) + xp.bincount(
        mapid.ravel(), weights=(frames.ravel()).imag
    )
    ret.shape = (Nx, Ny)
    return ret


# broadcast
def Illuminate_frames(frames, Illumination):
    # frames =frames*xp.reshape(Illumination,(1,xp.shape(Illumination)[0],xp.shape(Illumination)[1]))
    Illuminated = frames * xp.reshape(
        Illumination, (1, xp.shape(Illumination)[0], xp.shape(Illumination)[1])
    )
    return Illuminated


def Replicate_frame(frame, nframes):
    # replicate a frame along the first dimension
    Replicated = xp.repeat(frame[xp.newaxis, :, :], nframes, axis=0)
    return Replicated


def Sum_frames(frames):
    Summed = xp.add(frames, axis=0)
    return Summed


def Stack_frames(frames, omega):
    # multiply frames by a vector in the first dimension
    omega = omega.reshape([len(omega), 1, 1])
    # stv=xp.multiply(frames,omega)
    stv = frames * omega
    return stv


def ket(ystackr, dx, dy, bw=0):
    # extracts the portion of the left frame that overlaps
    # dxi=dx[ii,jj].astype(int)
    # dyi=dy[ii,jj].astype(int)
    nx, ny = ystackr.shape
    #dxi = dx.astype(int)
    #dyi = dy.astype(int)
    dxi = dx
    dyi = dy
    ket = ystackr[
        max([0, dyi]) + bw : min([nx, nx + dyi]) - bw,
        max([0, dxi]) + bw : min([nx, nx + dxi]) - bw,
    ]
    # ket=ystackr[max([0,dxi])+bw:min([nx,nx+dxi])-bw,
    #             max([0,dyi])+bw:min([nx,nx+dyi])-bw]

    return ket


def bra(ystackl, dx, dy, bw=0):
    # calculates the portion of the right frame that overlaps
    bra = ket(ystackl, dx, dy, bw)
    return bra


#def braket(ystackl, ystackr, dd, bw):
def braket(ystackl, ystackr, dx,dy, bw):
    # calculates inner products between the overlapping portion
    #    dxi=dx[ii,jj]
    #    dyi=dy[ii,jj]
    #dxi = dd.real
    #dyi = dd.imag

    # bracket=xp.sum(xp.multiply(bra(ystackl[jj],nx,ny,-dxi,-dyi),ket(ystackr[ii],nx,ny,dxi,dyi)))
    # bracket=xp.vdot(bra(ystackl[jj],nx,ny,-dxi,-dyi),ket(ystackr[ii],nx,ny,dxi,dyi))
    #bket = xp.vdot(bra(ystackl, -dxi, -dyi, bw), ket(ystackr, dxi, dyi, bw))
    bket = xp.vdot(bra(ystackl, -dx, -dy, bw), ket(ystackr, dx, dy, bw))

    return bket

def braket_i(ii,framesl,framesr,col,row,dx,dy,bw):
#def braket_i(ystackl,ystackr,dx,dy,bw):
    #val = braket(framesl[col[ii]], framesr[row[ii]], dx[ii],dy[ii], bw).get()
    val = xp.vdot(bra(framesl[col[ii]], -dx[ii], -dy[ii], bw), ket(framesr[row[ii]], dx[ii], dy[ii], bw))
   #    val[ii] = braket(framesl[col[ii]], framesr[row[ii]], dd[ii], bw)
    return val
#braket_i = cp.fuse(kernel_name='braket_i')(braket_i)
    

def Gramiam_calc(framesl, framesr, plan):
    # computes all the inner products between overlaping frames
    col = plan["col"]
    row = plan["row"]
    #dd = plan["dd"]
    dx = plan["dx"]
    dy = plan["dy"]
    bw = plan["bw"]
    val = plan["val"]
    #print(type(col.dtype),type(dd.dtype))
    
    nframes = framesl.shape[0]
    nnz = len(col)
    # val=xp.empty((nnz,1),dtype=framesl.dtype)
    # val = shared_array(shape=(nnz),dtype=xp.complex128)
    
    #col=np.array([np.argwhere(col[i]==np.unique(col)) for i in range(np.size(col))]).ravel()
    #row=np.array([np.argwhere(row[i]==np.unique(row)) for i in range(np.size(row))]).ravel()
    
   
    # def proc1(ii):
    #    return braket(framesl[col[ii]],framesr[row[ii]],dd[ii],bw)

            
    #@cp.fuse(kernel_name='braket_i')
    
    time0 = timer()
    print(nnz)
    for ii in range(nnz):
        #braket_i(ii)
        val[ii] = braket_i(ii,framesl,framesr,col,row,dx,dy,bw)
    #    if ii == 0:
    #        t0 = timer()
    #        braket_i(ii)
    #        print('each dot takes', timer() - t0)
    
    '''
    #def parallel_braket_i(nnz,framesl,framesr,col,row,dx,dy,bw):
        # Create a pool of workers
    #    pool = mp.Pool()
        # Parallelize the loop
    #    results = pool.map(braket_i, range(nnz))
        # Close the pool
    #    pool.close()
    #    pool.join()
    '''
    #parallel_braket_i(nnz,framesl,framesr,col,row,dx,dy,bw)
    
    timers["Gramiam"] += timer() - time0
    time0 = timer()
    
    
    # H=sp.sparse.csr_matrix((val.ravel(), (col, row)), shape=(nframes,nframes))
    if GPU == False:
        H = sp.sparse.coo_matrix((val.ravel(), (col, row)), shape=(nframes, nframes))
        H = H + (sp.sparse.triu(H, 1)).getH()
        H = H.tocsr()
        timers["Gramiam_completion"] += timer() - time0
    else:
        H = sparse.coo_matrix((val.ravel(), (col, row)), shape=(nframes, nframes))
        H += sparse.triu(H, k=1).conj().T
        H = H.tocsr()
        timers["Gramiam_completion"] += timer() - time0
        
    return H

resource_package = __name__

zQQz_raw_kernel = None
if zQQz_raw_kernel == None:
    resource_path = '/'.join(('src','zQQz.cu'))
    file_name = pkg_resources.resource_filename(resource_package, resource_path)
    with open(file_name, 'r') as myfile:
        zQQz_raw_kernel = myfile.read()
        
def Gramiam_calc_cuda(frames, plan,illumination,normalization):
    col = plan['col']
    row = plan['row']
    dx = plan['dx']
    dy = plan['dy']
#    ptr = xp.concatenate(([0],xp.nonzero(xp.diff(col))[0]+1,[len(col)]))
    nnz = len(col)
    frame_height = frames.shape[1]
    frame_width = frames.shape[2]
    framesl = Illuminate_frames(frames, xp.conj(illumination))
    framesr = framesl * normalization
    
    nthreads = 128
    nblocks = nnz
    
#    #include <thrust/device_vector.h>
#    #include <thrust/complex.h>
#    thrust::device_vector<thrust::complex<float>> value(nframes);
    t0 = timer()
    value = xp.zeros(nnz,dtype = xp.complex64)
    cp.RawKernel(zQQz_raw_kernel,"dotp")\
        ((int(nblocks),),(int(nthreads),), \
         (value,framesl,framesr,col,row,dx,dy, nnz, frame_height, frame_width))
    # Try cupy sparse
    timers['Gramiam'] = timer() - t0
    
    nframes = framesl.shape[0]
    H = sparse.coo_matrix((value.ravel(), (col, row)), shape=(nframes, nframes))
    H += sparse.triu(H, k=1).conj().T
    H = H.tocsr()
    timers['Gramiam_completion']=timer() - t0
    return H

def Gramiam_plan(translations_x, translations_y, nframes, nx, ny, Nx, Ny, bw=0):
    # embed all geometric parameters into the gramiam function
    # calculates the difference of the coordinates between all frames
    print('!!!!!',type(translations_x.dtype))
    dx = translations_x.ravel(order="F").reshape(nframes, 1)
    dy = translations_y.ravel(order="F").reshape(nframes, 1)
    dx = xp.subtract(dx, xp.transpose(dx))
    dy = xp.subtract(dy, xp.transpose(dy))
    print('!!!!!',type(dx.dtype))
    # calculates the wrapping effect for a period boundary
    #dx = -(dx + Nx * ((dx < (-Nx / 2)).astype(float) - (dx > (Nx / 2)).astype(float)))
    dx = -(dx + Nx * ((dx < (-Nx / 2)).astype(int) - (dx > (Nx / 2)).astype(int)))
    #dy = -(dy + Ny * ((dy < (-Ny / 2)).astype(float) - (dy > (Ny / 2)).astype(float)))
    dy = -(dx + Nx * ((dx < (-Nx / 2)).astype(int) - (dx > (Nx / 2)).astype(int)))
    
    # find the frames idex that overlaps
    # col,row=xp.where(xp.tril(xp.logical_and(abs(dy)< nx,abs(dx) < ny)).T)
    col, row = xp.where(xp.tril((abs(dy) < nx - 2 * bw) * (abs(dx) < ny - 2 * bw)).T)
    #row, col = xp.where(xp.tril((abs(dy) < nx - 2 * bw) * (abs(dx) < ny - 2 * bw)))
    
    # complex displacement (x, 1j y)
    # why are col-row swapped? Maybe the .T
    #dd = dx[row, col].astype(np.int16) + 1j * dy[row, col].astype(np.int16)
    #print('dd type',type(dd.dtype))
    dx = dx[row,col]
    dy = dy[row,col]
    
    # col,row,dd=frames_overlap(translations_x,translations_y,nframes,nx,ny,Nx,Ny, bw)

    nnz = col.size
    val=xp.empty((nnz,1),dtype=xp.complex64)
    #val = shared_array(shape=(nnz, 1), dtype=xp.complex128)

    #plan = {"col": col, "row": row, "dd": dd, "val": val, "bw": bw}
    plan = {"col": col, "row": row, "dx": dx, "dy": dy,"val": val, "bw": bw}
    # Gramiam = lambda framesl,framesr: Gramiam_calc(framesl,framesr,plan)
    return plan


#    lambda Gramiam1
#    H=Gramiam(nframes,framesl,framesr,col,row,nx,ny,dx,dy)


def Precondition(H, frames, bw=0):
    time0 = timer()
    fw, fh = frames.shape[1:]
    frames_norm = xp.linalg.norm(frames[:, bw : fw - bw, bw : fh - bw], axis=(1, 2))
    print(type(frames_norm))
    print(frames_norm.shape)
    if GPU == False:
        D = sp.sparse.diags(1 / frames_norm)
    else: 
        D = sparse.diags(1 / frames_norm , format='csr')
    H1 = D @ H @ D
    # H1=(H1+H1.getH())/2
    timers["Precondition"] += timer() - time0 
    return H1, D

if GPU:
    from cupyx.scipy.sparse.linalg import eigsh
else:
    from scipy.sparse.linalg import eigsh


def Eigensolver(H):
    time0 = timer()

    nframes = xp.shape(H)[0]
    # print('nframes',nframes)
    v0 = xp.ones((nframes, 1))
    if GPU: 
        eigenvalues, eigenvectors = eigsh(H, k=1, which="LM", tol=1e-9)
    else:
        eigenvalues, eigenvectors = eigsh(H, k=1, which="LM", v0=v0, tol=1e-9)
    # eigenvalues, eigenvectors = eigsh(H, k=1,which='LM',v0=v0, tol=0)
    # eigenvalues, eigenvectors = eigsh(H, k=2,which='LM',v0=v0)
    # if dont specify starting point v0, converges to another eigenvector
    omega = eigenvectors[:, 0]
    timers["Eigensolver"] += timer() - time0

    omega = omega / xp.abs(omega)

    # subtract the average phase
    so = xp.conj(xp.sum(omega))
    so /= abs(so)
    omega *= so
    ########

    omega = xp.reshape(omega, (nframes, 1, 1))
    return omega


# def synchronize_frames_c(frames, illumination, normalization,translations_x,translations_y,nframes,nx,ny,Nx,Ny):
def synchronize_frames_c(frames, illumination, normalization, plan, bw=0):
    # col,row,dx,dy=frames_overlap(translations_x,translations_y,nframes,nx,ny,Nx,Ny)
    # Gramiam = Gramiam_plan(translations_x,translations_y,nframes,nx,ny,Nx,Ny)

    time0 = timer()
    timers["Sync_setup"] += timer() - time0
    if GPU:
        H = Gramiam_calc_cuda(frames, plan,illumination,normalization)
    else:
        framesl = Illuminate_frames(frames, xp.conj(illumination))
        framesr = framesl * normalization
        H = Gramiam_calc(framesl, framesr, plan)

    if "Preconditioner" in plan:
        time0 = timer()
        # print('hello')
        D = plan["Preconditioner"]
        H1 = D @ H @ D
        timers["Precondition"] = timer() - time0
    else:
        H1, D = Precondition(H, frames, bw)

    # compute the largest eigenvalue of H1
    omega = Eigensolver(H1)
    return omega


# def synchronize_frames_plan(inormalization_split,Gramiam):
#    omega=lambda frames synchronize_frames_c(frames, illumination, inormalization_split, Gramiam)
#    Gramiam = lambda framesl,framesr: Gramiam_calc(framesl,framesr,nframes,col,row,nx,ny,dx,dy)
#    return Gramiam


def mse_calc(img0, img1):
    # calculate the MSE between two images after global phase correction
    nnz = xp.size(img0)
    # compute the best phase
    phase = xp.dot(xp.reshape(xp.conj(img1), (1, nnz)), xp.reshape(img0, (nnz, 1)))[
        0, 0
    ]
    phase = phase / xp.abs(phase)
    # compute norm after correcting the phase
    mse = xp.linalg.norm(img0 - img1 * phase)
    #compute the best phase and scalar
    #phase = xp.conj(phase) / xp.linalg.norm(img0)**2
    #mse = xp.linalg.norm(img0 - img1 / phase)
    return mse

def common_scale(img0,img1):
    scale = xp.dot(img0.ravel(),img1.ravel()) / xp.dot(img0.ravel(),img0.ravel())   
    return scale

import ctypes
from multiprocessing import sharedctypes


def shared_array(shape=(1,), dtype=np.float32):
    np_type_to_ctype = {
        np.float32: ctypes.c_float,
        np.float64: ctypes.c_double,
        np.bool: ctypes.c_bool,
        np.uint8: ctypes.c_ubyte,
        np.uint64: ctypes.c_ulonglong,
        np.complex128: ctypes.c_double,
        np.complex64: ctypes.c_float,
    }

    numel = np.int(np.prod(shape))
    iscomplex = dtype == np.complex128 or dtype == xp.complex64
    # numel *=
    arr_ctypes = sharedctypes.RawArray(np_type_to_ctype[dtype], numel * (1 + iscomplex))
    np_arr = np.frombuffer(arr_ctypes, dtype=dtype, count=numel)
    np_arr.shape = shape

    return np_arr


def circular_aperture(radius,img):
    aperture = xp.zeros_like(img)
    center = (img.shape[0]//2,img.shape[1]//2)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if xp.sqrt((i - center[0])**2 + (j - center[1])**2) <= radius:
                aperture[i,j] = 1              
    return aperture

