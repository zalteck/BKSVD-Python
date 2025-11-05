import numpy as np
import utils
import matplotlib.pyplot as plt
# from PIL import Image

def MB_BKSVD4SD(I, D0, K, maxIter=100, batch_size=1000, n_batches=10):
    Y = utils.rgb2od(np.array(I))
    m, n, c = Y.shape
    Y_full = Y.reshape((m * n, c)).T

    tmp = np.mean(Y_full, axis=0)
    marcar = tmp > 0.2
    Y_filtered = Y_full[:, marcar]

    if Y_filtered.shape[1] < batch_size:
        print('batch_size reduced, not enough pixels')
        batch_size = Y_filtered.shape[1]
        n_batches = 1

    D = D0[:, :K]
    Devol = 1
    termD = 5.e-03
    current_batch = 0

    while current_batch < n_batches and Devol > termD:
        D_batch = D.copy()
        current_batch += 1
        icol = np.random.permutation(Y_filtered.shape[1])[:batch_size]
        Y_batch = Y_filtered[:, icol]
        P, Q = Y_batch.shape
        
        try:
            X0 = np.linalg.lstsq(D, Y_batch, rcond=None)[0]
        except np.linalg.LinAlgError:
            print("Singular matrix, cannot compute initial X0. Skipping batch.")
            continue

        X0[X0 < np.finfo(float).eps] = np.finfo(float).eps
        X = X0.copy()

        # Inits
        gamma = np.ones((K, Q))
        S_xq = np.zeros((K, K, Q))
        term = 1.e-04
        iter_count = 0
        minIter = 2
        convH, convE = 1, 1 # Initialize convergence variables

        while (iter_count <= minIter or (convH > term or convE > term)) and iter_count <= maxIter:
            iter_count += 1

            pX = [np.zeros((K, 1)) for _ in range(Q)]
            p_gamma = [np.zeros((K, 1)) for _ in range(Q)]
            pS_xq = [np.zeros((K, K)) for _ in range(Q)]

            for q in range(Q):
                lambda_q = (2 * K - 2) / np.sum(gamma[:, q])
                if lambda_q == 0: lambda_q = np.finfo(float).eps

                for s in range(K):
                    gamma_s1 = -1 / (2 * lambda_q) + np.sqrt(1 / (4 * lambda_q**2) + (X0[s, q]**2 + S_xq[s, s, q]) / lambda_q)
                    if gamma_s1 == 0: gamma_s1 = np.finfo(float).eps
                    p_gamma[q][s] = gamma_s1

                try:
                    inv_matrix = np.linalg.inv(np.diag(p_gamma[q].flatten()))
                    S_inv = np.linalg.inv(D.T @ D + inv_matrix)
                    pS_xq[q] = S_inv
                    pX[q] = S_inv @ D.T @ Y_batch[:, q]
                except np.linalg.LinAlgError:
                    # Handle cases where matrix is singular
                    pS_xq[q] = np.zeros((K,K))
                    pX[q] = np.zeros((K,1))


            X = np.array(pX).squeeze().T
            X[np.isnan(X)] = 0
            gamma = np.array(p_gamma).squeeze().T
            S_xq = np.array(pS_xq).transpose(1, 2, 0)

            if maxIter > 1:
                Dnew = D.copy()
                Sq = np.sum(S_xq, axis=2)
                used_all = np.where(np.sum(np.abs(X), axis=1) != 0)[0]

                for k in used_all:
                    # Create a list of indices excluding k
                    other_indices = [i for i in range(K) if i != k]
                    
                    # Safely compute ak
                    if len(other_indices) > 0:
                        ak = D[:, other_indices] @ Sq[other_indices, k]
                    else:
                        ak = np.zeros((P,))

                    bk = (Y_batch - D @ X + np.outer(D[:, k], X[k, :])) @ X[k, :].T
                    ck = np.sum(S_xq[k, k, :])
                    ek = np.sum(X[k, :]**2) + ck
                    
                    if ek == 0: continue # Avoid division by zero

                    tk = (1 / np.sqrt(ek)) * (bk - ak)
                    
                    norm_tk = np.linalg.norm(tk)
                    if norm_tk > 0:
                        Dnew[:, k] = tk / norm_tk
                
                D = Dnew

        Devol = np.linalg.norm(D_batch - D)
        print(f'- BKSVD - batch: {current_batch} - iter: {iter_count} of {maxIter}')

    X_final = utils.directDeconvolve(np.array(I), D)
    return D, X_final

def Normaliza(Mref, C, M, neg2cero, norm_fac):
    if neg2cero:
        C[C < np.finfo(float).eps] = np.finfo(float).eps

    ns, p = C.shape
    c = 3
    m = int(np.sqrt(p))
    n = m

    CT_norm = C * norm_fac[:, np.newaxis]
    Yrec = Mref[:, :ns] @ CT_norm
    Y2d = Yrec.T.reshape(m, n, c)
    Inorm = utils.od2rgb(Y2d)

    StainNorm = []
    for i in range(ns):
        stain = Mref[:, i, np.newaxis] @ CT_norm[i, np.newaxis, :]
        stain = utils.od2rgb(stain).T.reshape(m, n, c)
        StainNorm.append(stain)

    return Inorm, StainNorm

def MI_MB_BKSVD4SD(Images, D0, K):
    n_images = len(Images)
    images_per_batch = min(20, n_images)

    i_imgs = np.random.permutation(n_images)[:images_per_batch]
    
    # loaded_images = [np.array(plt.imread(Images[i])) for i in i_imgs]
    loaded_images = []
    for i in i_imgs:
        img = plt.imread(Images[i])
        # --- 1. Convert to uint8 ---
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        # --- 2. Remove alpha channel if present (RGBA → RGB) ---
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        loaded_images.append(img)
    
    I = np.concatenate(loaded_images, axis=0)

    D, CT = MB_BKSVD4SD(I, D0, K)
    C_Rmax = np.percentile(CT, 99, axis=1)

    return D, C_Rmax
