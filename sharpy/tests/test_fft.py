import pytest
from numpy.fft import fft2 as fft2_np
from numpy.fft import ifft2 as ifft2_np
import numpy as np

@pytest.fixture
def image():
    '''Returns a 2D image as numpy'''
    img_np = np.random.rand(2,3)
    return img_np

def test_fft2_equivalence(image):
    assert np.allclose(image, ifft2_np(fft2_np(image)))


