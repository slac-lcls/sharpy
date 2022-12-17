import cupy as cp
import numpy as np

def test_init():
    img = np.array((2,3))
    img_cp = cp.asarray(img)
    assert(np.allclose(img, img_cp))
