import numpy as np
from creativity import calculate_creativity
from inheritance import calculate_inheritance
from diversity import calculate_diversity


def calculate_cid_index(generated_imgs_arr: np.array, real_imgs_arr: np.array):
    generated_remaining_imgs_arr, creativity = calculate_creativity(
        generated_imgs_arr, real_imgs_arr)
    inheritance = calculate_inheritance(
        generated_remaining_imgs_arr, real_imgs_arr)
    diversity = calculate_diversity(generated_remaining_imgs_arr)

    cid = creativity * inheritance * diversity

    return cid, (creativity, inheritance, diversity)
