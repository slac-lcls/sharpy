from cupyx.scipy import fftpack

# Cached cuFFT plans. A cupy FFT plan is specific to the input SHAPE (incl. the
# batch dimension) and DTYPE, so we cache the plan together with the (shape,
# dtype) key it was built for and re-plan whenever that key changes. Reusing a
# plan for a different batch size / dtype silently produces WRONG results (it
# does not raise) -- this previously caused reconstructions to stagnate when the
# number of frames changed within one process (e.g. running several scan sizes
# in one notebook kernel).
global plan1D
global plan2D
plan1D = None
plan2D = None
_key1D = None
_key2D = None
# print("Plan1D",plan1D)
owrite = True
Plan = True


def fft_reset(x=None):
    global plan1D, plan2D, _key1D, _key2D
    plan1D = None
    plan2D = None
    _key1D = None
    _key2D = None


def _plankey(x):
    return (x.shape, x.dtype)


def fft(x):
    global plan1D, _key1D
    if not Plan:
        return fftpack.fft(x, overwrite_x=owrite)
    k = _plankey(x)
    if plan1D is None or _key1D != k:
        plan1D = fftpack.get_fft_plan(x, axes=(-1))
        _key1D = k
    return fftpack.fft(x, overwrite_x=owrite, plan=plan1D)


def ifft(x):
    global plan1D, _key1D
    if not Plan:
        return fftpack.ifft(x, overwrite_x=owrite)
    k = _plankey(x)
    if plan1D is None or _key1D != k:
        plan1D = fftpack.get_fft_plan(x, axes=(-1))
        _key1D = k
    return fftpack.ifft(x, overwrite_x=owrite, plan=plan1D)


def fft2(x):
    global plan2D, _key2D
    if not Plan:
        return fftpack.fft2(x, overwrite_x=owrite)
    k = _plankey(x)
    if plan2D is None or _key2D != k:
        plan2D = fftpack.get_fft_plan(x, axes=(-2, -1))
        _key2D = k
    return fftpack.fft2(x, overwrite_x=owrite, plan=plan2D)


def ifft2(x):
    global plan2D, _key2D
    if not Plan:
        return fftpack.ifft2(x, overwrite_x=owrite)
    k = _plankey(x)
    if plan2D is None or _key2D != k:
        plan2D = fftpack.get_fft_plan(x, axes=(-2, -1))
        _key2D = k
    return fftpack.ifft2(x, overwrite_x=owrite, plan=plan2D)
