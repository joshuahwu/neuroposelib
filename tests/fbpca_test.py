import fbpca
import numpy as np
import time


t = time.time()
features = np.random.Generator.random((int(2e7), 1000), dtype=np.float32)
latent_mem = np.zeros(features.shape)
n_pcs = 60
t1 = time.time()

(_, _, V) = fbpca.pca(features, k=n_pcs)
t2 = time.time()

pca_feats = np.matmul(features, V.T)
t3 = time.time()
latent_mem += 1

print("Create features", t1 - t)
print("fbpca", t2 - t1)
print("matmul", t3 - t2)
