import numpy as np
import math
import sys
from sklearn.cluster import KMeans
from libKMCUDA import kmeans_cuda
import sklearn.cluster.k_means_
from sklearn.utils.extmath import row_norms, squared_norm
from numpy.random import RandomState
import time


def k_means_vector_gpu_fp32(weight_vector, n_clusters, verbosity=0, seed=int(time.time()), gpu_id=7):

    if n_clusters == 1:
        mean_sample = np.mean(weight_vector, axis=1)
        weight_vector = np.tile(mean_sample, (weight_vector.shape[0], 1))
        return weight_vector
    elif weight_vector.shape[1] == 1:
        return weight_vector
    elif weight_vector.shape[0] == n_clusters:
        return weight_vector
    else:
        init_centers = sklearn.cluster.k_means_._k_init(X=weight_vector, n_clusters=n_clusters, x_squared_norms=row_norms(weight_vector, squared=True), random_state=RandomState(seed))
        centers, labels = kmeans_cuda(samples = weight_vector, clusters = n_clusters, init=init_centers, yinyang_t=0, seed=seed, device=gpu_id, verbosity=verbosity)
        weight_vector_compress = np.zeros((weight_vector.shape[0], weight_vector.shape[1]), dtype=np.float32)
        for v in range(weight_vector.shape[0]):
            weight_vector_compress[v, :] = centers[labels[v], :]
        return weight_vector_compress
