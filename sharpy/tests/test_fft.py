import pytest
from sharpy.fft_plan import fft2 as fft2_cp
from numpy.fft import fft2 as fft2_np
import numpy as np
import cupy as cp

@pytest.fixture
def image():
    '''Returns a 2D image as numpy and cupy'''
    img_np = np.random.rand(2,3)
    img_cp = cp.array(img_np)
    return img_np, img_cp

def test_fft2_equivalence(image):
    assert np.allclose(fft2_np(image[0]), cp.asnumpy(fft2_cp(image[1])))


