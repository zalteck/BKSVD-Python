import numpy as np

def rgb2od(rgb):
    """Converts RGB to optical density."""
    # Add a small epsilon to avoid taking the log of zero
    eps = 1e-6
    return -np.log((rgb.astype(float) + 1) / 256 + eps)

def od2rgb(od):
    """Converts optical density to RGB."""
    return np.clip(256 * np.exp(-od) - 1,0,255).astype('uint8')

def directDeconvolve(I, RM):
    """Directly deconvolves the image.

    Args:
        I: The image to deconvolve.
        RM: The reference matrix.

    Returns:
        The deconvolved image.
    """
    Y = rgb2od(I)
    m, n, c = Y.shape
    YT = Y.reshape((m * n, c)).T
    CT = np.linalg.lstsq(RM, YT, rcond=None)[0]
    return CT

