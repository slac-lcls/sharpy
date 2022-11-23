"""
Ptycho operators

"""
import numpy as np
import scipy as sp
import math
import numpy_groupies

#import multiprocessing as mp

# timers: keep track of timing for different operators
from timeit import default_timer as timer
timers={'Overlap':0, 'Split':0, 'Project_data':0, 'Gramiam':0, 'Gramiam_completion':0, 'Precondition':0, 'Eigensolver':0, 'Sync_setup':0}

def get_times(): 
    return timers
def reset_times():
    for keys in timers: timers[keys]=0


def make_probe(nx,ny, r1= 0.03, r2=0.06, fx = 0., fy = 0.):
    """
    make an illumination (probe) in a (nx, ny) frame shape
    r1,r2 fractions of of the frame width 
    fx,fy:  x-y quadradic fase (focus)

    """
    xi=np.reshape(np.arange(0,nx)-nx/2,(nx,1))
    
    xi = np.fft.ifftshift(xi)
    
    rr=np.sqrt(xi**2+(xi.T)**2)
    r1= r1*nx #define zone plate circles
    r2= r2*nx

    Fprobe=((rr>=r1) & (rr<=r2))
    
    
    phase = np.exp(1j*fx*np.pi*((xi/nx)**2)) * np.exp(1j*fy*np.pi*((xi.T/nx)**2))
    
    Fprobe = Fprobe * phase
    
    probe=np.fft.fftshift(np.fft.ifft2(Fprobe))
    probe=probe/max(abs(probe).flatten())
    return probe
    

# close packing translations    
def make_translations(Dx,Dy,nnx,nny,Nx,Ny):
    """ 
     make scan positions using spacing Dx,Dy, number of steps nnx, nny, 
     image width Nx,Ny. The lattice is periodic with close-packing arrangement
     
    """
    
    ix,iy=np.meshgrid(np.arange(0,Dx*nnx,Dx)+Nx/2-Dx*nnx/2+1,
                      np.arange(0,Dy*nny,Dy)+Ny/2-Dy*nny/2+1)
    xshift=math.floor(Dx/2)*np.mod(np.arange(1,np.size(ix,1)+1),2)
    # adding shift in the x-direction to make close-packing lattice 
    ix=np.transpose(np.add(np.transpose(ix),xshift))
    
    ix=ix-np.min(ix)
    iy=iy-np.min(iy)
    
    
    ix=np.reshape(ix,(nnx*nny,1,1))
    iy=np.reshape(iy,(nnx*nny,1,1))
    
    
    return ix,iy
    

def map_frames(translations_x,translations_y,nx,ny,Nx,Ny):
    """
    return frame mapping: frames = image[mapid]
    """

    # map frames to image indices 
    translations_x=np.reshape(np.transpose(translations_x),(np.size(translations_x),1,1))
    translations_y=np.reshape(np.transpose(translations_y),(np.size(translations_y),1,1))

    
    xframeidx,yframeidx=np.meshgrid(np.arange(nx),np.arange(ny))
    # print('translations shapes:',np.shape(translations_x),'frameidx',np.shape(xframeidx))
    
    spv_x=np.add(xframeidx,translations_x) 
    spv_y=np.add(yframeidx,translations_y) 
    
    mapidx=np.mod(spv_x,Nx)
    mapidy=np.mod(spv_y,Ny)
    mapid=np.add(mapidx,mapidy*Nx) 
    #mapid=np.add(mapidx*Nx,mapidy) 
    mapid=mapid.astype(int)
    return mapid
    

def Splitc(img,mapid):
    time0=timer()
    # Split an image into frames given mapping
    frames_out = (img.ravel())[mapid]
    timers['Split']+=timer()-time0
    return frames_out

def Overlapc(frames,Nx,Ny, mapid): #check
    # overlap frames onto an image
    time0=timer()
    accum = np.reshape(numpy_groupies.aggregate(mapid.ravel(),frames.ravel()),(Nx,Ny))
    timers['Overlap']+=timer()-time0
    return accum

def Split_plan(translations_x,translations_y,nx,ny,Nx,Ny):
    mapid=map_frames(translations_x,translations_y,nx,ny,Nx,Ny)  
    Split = lambda img: Splitc(img,mapid)
    return Split

def Overlap_plan(translations_x,translations_y,nx,ny,Nx,Ny):
    mapid=map_frames(translations_x,translations_y,nx,ny,Nx,Ny)  
    Overlap = lambda frames: Overlapc(frames,Nx,Ny,mapid)
    return Overlap



def crop_center(img, cropx, cropy):
    # crop an image
       y, x = img.shape
       startx = x // 2 - (cropx // 2)
       starty = y // 2 - (cropy // 2)
       return img[starty:starty + cropy, startx:startx + cropx]


def cropmat(img,size):
    # crop an image to a given size
    left0=math.floor((np.size(img,0)-size[0])/2)
    right0=(size[0]+math.floor((np.size(img,0)-size[0])/2))
    left1=math.floor((np.size(img,1)-size[1])/2)
    right1=(size[1]+math.floor((np.size(img,1)-size[1])/2))
    crop_img= img[left0:right0,left1:right1]
    return crop_img



def Overlapc0(frames,Nx,Ny,mapid):
    
    #ret = np.bincount(mapid, weights=frames.real)+1j*np.bincount(mapid, weights=frames.imag)
    ret = np.bincount(mapid.ravel(), weights=(frames.ravel()).real)+np.bincount(mapid.ravel(), weights=(frames.ravel()).imag)
    ret.shape=(Nx,Ny)
    return ret

# broadcast
def Illuminate_frames(frames,Illumination):
    Illuminated=frames*np.reshape(Illumination,(1,np.shape(Illumination)[0],np.shape(Illumination)[1]))
    return Illuminated
  
def Replicate_frame(frame,nframes):
    # replicate a frame along the first dimension
    Replicated= np.repeat(frame[np.newaxis,:, :], nframes, axis=0)
    return Replicated

def Sum_frames(frames):
    Summed=np.add(frames,axis=0)
    return Summed

def Stack_frames(frames,omega):
    # multiply frames by a vector in the first dimension
    omega=omega.reshape([len(omega),1,1])
    #stv=np.multiply(frames,omega)
    stv=frames*omega
    return stv


def ket(ystackr,dx,dy,bw=0): 
    #extracts the portion of the left frame that overlaps
    #dxi=dx[ii,jj].astype(int)
    #dyi=dy[ii,jj].astype(int)
    nx,ny = ystackr.shape
    dxi=dx.astype(int)
    dyi=dy.astype(int)
    ket=ystackr[max([0,dyi])+bw:min([nx,nx+dyi])-bw,
                max([0,dxi])+bw:min([nx,nx+dxi])-bw]
    # ket=ystackr[max([0,dxi])+bw:min([nx,nx+dxi])-bw,
    #             max([0,dyi])+bw:min([nx,nx+dyi])-bw]

    return ket

def bra(ystackl,dx,dy,bw=0):
    #calculates the portion of the right frame that overlaps
    bra=ket(ystackl,dx,dy,bw)
    return bra

def braket(ystackl,ystackr,dd,bw):
    #calculates inner products between the overlapping portion
#    dxi=dx[ii,jj]
#    dyi=dy[ii,jj]
    dxi=dd.real
    dyi=dd.imag
    
    #bracket=np.sum(np.multiply(bra(ystackl[jj],nx,ny,-dxi,-dyi),ket(ystackr[ii],nx,ny,dxi,dyi)))
    #bracket=np.vdot(bra(ystackl[jj],nx,ny,-dxi,-dyi),ket(ystackr[ii],nx,ny,dxi,dyi))
    bket=np.vdot(bra(ystackl,-dxi,-dyi,bw),ket(ystackr,dxi,dyi,bw))
    
    
    return bket



#from multiprocessing import Process

def Gramiam_calc(framesl,framesr,plan):
    # computes all the inner products between overlaping frames
    col=plan['col']
    row=plan['row']
    dd=plan['dd']
    bw=plan['bw']
    val=plan['val']
    
    nframes=framesl.shape[0]
    nnz=len(col)
    #val=np.empty((nnz,1),dtype=framesl.dtype)
    #val = shared_array(shape=(nnz),dtype=np.complex128)
 
    def braket_i(ii):
        val[ii] = braket(framesl[col[ii]],framesr[row[ii]],dd[ii],bw)
    #def proc1(ii):
    #    return braket(framesl[col[ii]],framesr[row[ii]],dd[ii],bw)
    
    time0=timer()        
    for ii in range(nnz):
        braket_i(ii)
    timers['Gramiam']+=timer()-time0
    time0=timer()
        
    #H=sp.sparse.csr_matrix((val.ravel(), (col, row)), shape=(nframes,nframes))
    H=sp.sparse.coo_matrix((val.ravel(), (col, row)), shape=(nframes,nframes))    
    H=H+(sp.sparse.triu(H,1)).getH()
    H=H.tocsr()
    timers['Gramiam_completion']+=timer()-time0
    
    return H


def Gramiam_plan(translations_x,translations_y,nframes,nx,ny,Nx,Ny,bw =0):
    # embed all geometric parameters into the gramiam function
    #calculates the difference of the coordinates between all frames
    dx=translations_x.ravel(order='F').reshape(nframes,1)
    dy=translations_y.ravel(order='F').reshape(nframes,1)
    dx=np.subtract(dx,np.transpose(dx))
    dy=np.subtract(dy,np.transpose(dy))
    
    #calculates the wrapping effect for a period boundary
    dx=-(dx+Nx*((dx < (-Nx/2)).astype(float)-(dx > (Nx/2)).astype(float)))
    dy=-(dy+Ny*((dy < (-Ny/2)).astype(float)-(dy > (Ny/2)).astype(float)))    
 
    #find the frames idex that overlaps
    #col,row=np.where(np.tril(np.logical_and(abs(dy)< nx,abs(dx) < ny)).T)
    col,row=np.where(np.tril((abs(dy)< nx-2*bw)*(abs(dx) < ny-2*bw)).T)

    # complex displacement (x, 1j y)
    # why are col-row swapped?
    dd = dx[row,col]+1j*dy[row,col]

    #col,row,dd=frames_overlap(translations_x,translations_y,nframes,nx,ny,Nx,Ny, bw)
 
    nnz=col.size
#    val=np.empty((nnz,1),dtype=np.complex128)
    val = shared_array(shape=(nnz,1),dtype=np.complex128)
    
    plan={'col':col,'row':row,'dd':dd, 'val':val,'bw':bw}
    #Gramiam = lambda framesl,framesr: Gramiam_calc(framesl,framesr,plan)
    return  plan
    
    

#    lambda Gramiam1
#    H=Gramiam(nframes,framesl,framesr,col,row,nx,ny,dx,dy)


def Precondition(H,frames, bw = 0):
    time0=timer()
    fw,fh=frames.shape[1:]
    frames_norm=np.linalg.norm(frames[:,bw:fw-bw ,bw:fh-bw],axis=(1,2))
    D=sp.sparse.diags(1/frames_norm)
    H1=D @ H @ D
    #H1=(H1+H1.getH())/2
    timers['Precondition']+=timer()-time0
    return H1, D

    
from scipy.sparse.linalg import eigsh
def Eigensolver(H):
    time0=timer()
    
    nframes=np.shape(H)[0]
    #print('nframes',nframes)
    v0=np.ones((nframes,1))
    eigenvalues, eigenvectors = eigsh(H, k=1,which='LM',v0=v0, tol=1e-9)
    #eigenvalues, eigenvectors = eigsh(H, k=1,which='LM',v0=v0, tol=0)
    #eigenvalues, eigenvectors = eigsh(H, k=2,which='LM',v0=v0)
    #if dont specify starting point v0, converges to another eigenvector
    omega=eigenvectors[:,0]
    timers['Eigensolver']+=timer()-time0

    omega=omega/np.abs(omega)
    
    # subtract the average phase
    so=np.conj(np.sum(omega))
    so/=abs(so)    
    omega*=so
    ########
    
    omega=np.reshape(omega,(nframes,1,1))
    return omega


#def synchronize_frames_c(frames, illumination, normalization,translations_x,translations_y,nframes,nx,ny,Nx,Ny):
def synchronize_frames_c(frames, illumination, normalization,plan, bw=0):
    #col,row,dx,dy=frames_overlap(translations_x,translations_y,nframes,nx,ny,Nx,Ny)
    # Gramiam = Gramiam_plan(translations_x,translations_y,nframes,nx,ny,Nx,Ny)

    time0=timer()   
    framesl=Illuminate_frames(frames,np.conj(illumination))
    framesr=framesl*normalization    
    timers['Sync_setup']+=timer()-time0

    H=Gramiam_calc(framesl,framesr,plan)
    
    if 'Preconditioner' in plan: 
        time0=timer()
        #print('hello')
        D = plan['Preconditioner']
        H1 = D @ H @ D
        timers['Precondition']=timer()-time0
    else:
        H1, D = Precondition(H,frames,bw)

    
    #compute the largest eigenvalue of H1    
    omega=Eigensolver(H1)
    return omega

#def synchronize_frames_plan(inormalization_split,Gramiam):
#    omega=lambda frames synchronize_frames_c(frames, illumination, inormalization_split, Gramiam)
#    Gramiam = lambda framesl,framesr: Gramiam_calc(framesl,framesr,nframes,col,row,nx,ny,dx,dy)
#    return Gramiam

def mse_calc(img0,img1):
    # calculate the MSE between two images after global phase correction
    nnz=np.size(img0)
    # compute the best phase
    phase=np.dot(np.reshape(np.conj(img1),(1,nnz)),np.reshape(img0,(nnz,1)))[0,0]
    phase=phase/np.abs(phase)
    # compute norm after correcting the phase
    mse=np.linalg.norm(img0-img1*phase)
    return mse

def Propagate(frames):
    # simple propagation
    return np.fft.fft2(frames)

def IPropagate(frames):
    # simple inverse propagation
    return np.fft.ifft2(frames)

eps = 1e-8
def Project_data(frames,frames_data):
    time0=timer()
    # apply Fourier magnitude projections
    frames = Propagate(frames)
    mse = np.linalg.norm(np.abs(frames)-np.sqrt(frames_data))

    frames *= np.sqrt((frames_data+eps)/(np.abs(frames)**2+eps))
    frames = IPropagate(frames)
    timers['Project_data']+=timer()-time0
    return frames, mse
    



import ctypes
from multiprocessing import sharedctypes
def shared_array(shape=(1,), dtype=np.float32):  
    np_type_to_ctype = {np.float32: ctypes.c_float,
                        np.float64: ctypes.c_double,
                        np.bool: ctypes.c_bool,
                        np.uint8: ctypes.c_ubyte,
                        np.uint64: ctypes.c_ulonglong,
                        np.complex128: ctypes.c_double,
                        np.complex64: ctypes.c_float}

    numel = np.int(np.prod(shape))
    iscomplex=(dtype == np.complex128 or dtype == np.complex64)
    #numel *= 
    arr_ctypes = sharedctypes.RawArray(np_type_to_ctype[dtype], numel*(1+iscomplex))
    np_arr = np.frombuffer(arr_ctypes, dtype=dtype, count=numel)
    np_arr.shape = shape

    return np_arr     
        