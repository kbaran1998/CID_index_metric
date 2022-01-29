# Adjusted from https://github.com/llvll/imgcluster
import numpy as np
from sklearn.cluster import SpectralClustering, AffinityPropagation
from skimage.metrics import structural_similarity


""" Returns the normalized similarity value (from 0.0 to 1.0) for the provided pair of images.
    * SSIM: Structural Similarity Index
"""


def get_image_similarity(img1, img2):
    return structural_similarity(img1, img2, channel_axis=2)


"""Fetches all images from the provided directory and calculates the similarity value per image pair."""


def build_similarity_matrix(images):
    num_images = len(images)
    sm = np.zeros(shape=(num_images, num_images),
                  np.dtype=np.float64)
    np.fill_diagonal(sm, 1.0)

    # Traversing the upper triangle only - transposed matrix will be used
    # later for filling the empty cells.
    k = 0
    for i in range(sm.shape[0]):
        for j in range(sm.shape[1]):
            j = j + k
            if i != j and j < sm.shape[1]:
                sm[i][j] = get_image_similarity(images[i], images[j])
        k += 1

    # Adding the transposed matrix and subtracting the diagonal to obtain
    # the symmetric similarity matrix
    sm = sm + sm.T - np.diag(sm.diagonal())
    return sm


""" Returns a dictionary with the computed performance metrics of the provided cluster.
    Several functions from sklearn.metrics are used to calculate the following:
    * Silhouette Coefficient
      Values near 1.0 indicate that the sample is far away from the neighboring clusters.
      A value of 0.0 indicates that the sample is on or very close to the decision boundary
      between two neighboring clusters and negative values indicate that those samples might
      have been assigned to the wrong cluster.
"""


def get_silhouette_coefficient(X, labels):
    metrics_dict = dict()
    np.fill_diagonal(X, 0.0)
    return metrics.silhouette_score(X, labels, metric='precomputed')


def get_best_spectral_coefficient(matrix, cluster_range=15):
    best_sc = None
    best_sc_silhouette_coefficient = None
    c_range = int(cluster_range/2)
    for n_clusters in range(2, c_range):
        sc = SpectralClustering(n_clusters=n_clusters,
                                affinity='precomputed').fit(matrix)
        sc_silhouette_coefficient = get_silhouette_coefficient(
            matrix, sc.labels_)
        if best_sc is None or best_sc_silhouette_coefficient < sc_silhouette_coefficient:
            best_sc_silhouette_coefficient = sc_silhouette_coefficient
            best_sc = sc
    return best_sc, best_sc_silhouette_coefficient


""" Executes two algorithms for similarity-based clustering:
    * Spectral Clustering
    * Affinity Propagation
    ... and selects the best results according to the clustering performance metrics.
"""


def perform_clustering(images: np.array):
    matrix = build_similarity_matrix(images)

    sc, sc_silhouette_coefficient = get_best_spectral_coefficient(
        matrix, cluster_range=len(images)-2)

    af = AffinityPropagation(affinity='precomputed').fit(matrix)
    af_silhouette_coefficient = get_silhouette_coefficient(matrix, af.labels_)

    if sc_silhouette_coefficient >= af_silhouette_coefficient:
        return sc.labels_
    return af.labels_
