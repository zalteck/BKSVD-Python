import numpy as np
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt
from bksvd import MI_MB_BKSVD4SD, Normaliza
import glob
from utils import directDeconvolve

# Params
ns = 2

# Initial reference matrix
mat_data = scipy.io.loadmat('data/MLandini.mat')
RM = mat_data['RM']
D0 = RM[:, :ns]


#Assume known reference (Estimated in the single example)
if 'Mref' not in locals():
    Mref = np.array([[0.47422288, 0.32263002],
           [0.79314221, 0.82746922],
           [0.38214931, 0.45956997]])
    Cref_Rmax = np.array([3.56385862, 1.97801672])


# Images to normalize
images_to_normalize = glob.glob('data/patched_image/*')
M, C_Rmax = MI_MB_BKSVD4SD(images_to_normalize, D0, ns)

# Deconvolution and normalization
norm_fac = Cref_Rmax / C_Rmax
neg2cero = False


# Load images to be normalized
loaded_images = [plt.imread(p) for p in images_to_normalize[:5]]

fig, axes = plt.subplots(2, len(loaded_images), figsize=(15, 10))

for i, p in enumerate(images_to_normalize[:5]):

    img = plt.imread(p)
    # --- 1. Convert to uint8 ---
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    # --- 2. Remove alpha channel if present (RGBA → RGB) ---
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    
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
