import numpy as np
from skimage.metrics import structural_similarity


def calculate_creativity(generated_imgs_arr: np.array, real_imgs_arr: np.array):
    generated_imgs_arr_remaining = []
    for generated_img in generated_imgs_arr:
        found_duplicate = False
        for real_img in real_imgs_arr:
            if structural_similarity(generated_img, real_img, channel_axis=2) >= 0.8:
                found_duplicate = True
                break
    if not found_duplicate:
        generated_imgs_arr_remaining.append(generated_img)
    creativity = len(generated_imgs_arr_remaining) / len(generated_imgs_arr)
    return generated_imgs_arr_remaining, creativity
