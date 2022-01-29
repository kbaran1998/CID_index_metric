import numpy as np
from clustering import perform_clustering


def calculate_diversity(generated_remaining_imgs_arr: np.array):
    clusters = perform_clustering(generated_remaining_imgs_arr)
    num_clusters = len(set(clusters))

    diversity = 0
    for cluster_index in range(num_clusters):
        cluster_count = 0
        for cluster_num in clusters:
            if cluster_num == cluster_index:
                cluster_count += 1
        p_i = cluster_count / len(generated_remaining_imgs_arr)
        diversity += p_i * np.log([p_i])[0]
    diversity = -1 * diversity
    return diversity
