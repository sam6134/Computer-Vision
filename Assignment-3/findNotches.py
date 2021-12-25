from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def doDFT(img: np.array) -> np.array:
    """
        Caculates discrete fourier transform of image array
    """
    dft = np.fft.fft2(img, axes=(0,1))
    dftShift = np.fft.fftshift(dft)
    return dftShift

notchImg = Image.open("Assignment-3/notch.JPEG").convert('L')
notchImg = np.array(notchImg)
dftNotchImg = doDFT(notchImg)
plt.imshow(np.log(np.abs(dftNotchImg)), cmap='gray')
plt.show()