from cupyx.scipy import fftpack

global plan1D
global plan2D
plan1D = None
plan2D = None
# print("Plan1D",plan1D)
owrite = True
Plan = True


def fft_reset(x):
    global plan1D
    global plan2D
    plan1D = None
    plan2D = None


def fft(x):
    global plan1D
    # print("Plan1D",plan1D)
    if plan1D == None and Plan:
        plan1D = fftpack.get_fft_plan(x, axes=(-1))

    #    print('plan1d', plan1D)
    #    print('plan1d', plan1D)
    #    print('plan1d', plan1D)
    try:
        return fftpack.fft(x, overwrite_x=owrite, plan=plan1D)
    except:
        plan1D = fftpack.get_fft_plan(x, axes=(-1))
        return fftpack.fft(x, overwrite_x=owrite, plan=plan1D)

    # return x


def ifft(x):
    global plan1D
    if plan1D == None and Plan:
        plan1D = fftpack.get_fft_plan(x, axes=(-1))

    return fftpack.ifft(x, overwrite_x=owrite, plan=plan1D)

    # return
    # return x


def fft2(x):
    global plan2D
    if plan2D == None and Plan:
        plan2D = fftpack.get_fft_plan(x, axes=(-2, -1))
        if True:
            print('plan2D is None')
            print(type(x.dtype))
            print(plan2D)
    else:
        #fft_reset(x)
        #plan2D = fftpack.get_fft_plan(x, axes=(-2, -1))
        if True:
            print('Using existing plan2D')
            print(type(x.dtype))
            print(plan2D)
    # return
    return fftpack.fft2(x, overwrite_x=owrite, plan=plan2D)
    # return x


def ifft2(x):
    global plan2D
    if plan2D == None and Plan:
        plan2D = fftpack.get_fft_plan(x, axes=(-2, -1))

    #   return
    return fftpack.ifft2(x, overwrite_x=owrite, plan=plan2D)
    # return x
