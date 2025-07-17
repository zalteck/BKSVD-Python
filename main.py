import numpy as np
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt
from bksvd import MB_BKSVD4SD, Normaliza

# Params
ns = 2
max_iter = 100
batch_size = 1000
n_batches = 10

# Initial reference matrix
mat_data = scipy.io.loadmat('data/MLandini.mat')
RM = mat_data['RM']
D0 = RM[:, :ns]

# Reference Image
I_ref = Image.open('data/Reference.jpg')
Mref, Cref = MB_BKSVD4SD(I_ref, D0, ns)
Cref_Rmax = np.percentile(Cref, 99, axis=1)

# Images to process
Images = ['data/hist1.jpg', 'data/hist2.jpg', 'data/hist3.jpg']
neg2cero = False

fig, axes = plt.subplots(2, len(Images), figsize=(15, 10))

for i, img_path in enumerate(Images):
    I = Image.open(img_path)
    m, n, c = np.array(I).shape
    M, C = MB_BKSVD4SD(I, D0, ns)
    C_Rmax = np.percentile(C, 99, axis=1)
    norm_fac = Cref_Rmax / C_Rmax
    Irec_norm, _ = Normaliza(Mref, C, M, neg2cero, norm_fac)
    Irec_norm = np.clip(Irec_norm, 0, 255).astype(np.uint8)

    axes[0, i].imshow(I)
    axes[0, i].set_title(f'Original Image: {img_path}')
    axes[0, i].axis('off')

    axes[1, i].imshow(Irec_norm)
    axes[1, i].set_title(f'Normalized Image: {img_path}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()

