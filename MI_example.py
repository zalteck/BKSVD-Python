import numpy as np
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt
from bksvd import MI_MB_BKSVD4SD, Normaliza
import glob

# Params
ns = 2

# Initial reference matrix
mat_data = scipy.io.loadmat('data/MLandini.mat')
RM = mat_data['RM']
D0 = RM[:, :ns]

# Reference Images
ref_images = glob.glob('data/hist1*.jpg')
Mref, Cref_Rmax = MI_MB_BKSVD4SD(ref_images, D0, ns)

# Images to normalize
images_to_normalize = glob.glob('data/hist2*.jpg')
M, C_Rmax = MI_MB_BKSVD4SD(images_to_normalize, D0, ns)

# Deconvolution and normalization
norm_fac = Cref_Rmax / C_Rmax
neg2cero = False

# Load images to be normalized
loaded_images = [Image.open(p) for p in images_to_normalize[:5]]

fig, axes = plt.subplots(2, len(loaded_images), figsize=(15, 10))

for i, img in enumerate(loaded_images):
    # Direct deconvolution is implicitly handled within Normaliza in the MATLAB code
    # but here we need to calculate C first.
    # For simplicity, we will use the M from the multi-image processing.
    # A more accurate approach would be to deconvolve each image with the learned M.
    from utils import directDeconvolve
    C = directDeconvolve(np.array(img), M)

    Irec_norm, _ = Normaliza(Mref, C, M, neg2cero, norm_fac)
    Irec_norm = np.clip(Irec_norm, 0, 255).astype(np.uint8)

    axes[0, i].imshow(img)
    axes[0, i].set_title(f'Original Image')
    axes[0, i].axis('off')

    axes[1, i].imshow(Irec_norm)
    axes[1, i].set_title(f'Normalized Image')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()
