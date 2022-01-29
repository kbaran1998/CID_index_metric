import numpy as np
from skimage.feature import graycomatrix, graycoprops


def rgb2gray(rgb: np.array):
    res = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]) * 255
    return res.astype(np.uint8)


def contrast(images_gray_scale_uint8: np.array):
    avg_contrast = 0
    for grey_scaled_img in images_gray_scale_uint8:
        glcm = graycomatrix(grey_scaled_img, [5], [0])
        contrast_calc = graycoprops(glcm, 'contrast')[0, 0]
        avg_contrast += contrast_calc
    avg_contrast = avg_contrast / len(images_gray_scale_uint8)
    return avg_contrast


def calculate_inheritance(generated_remaining_imgs_arr: np.array, real_imgs_arr: np.array):
    generated_remaining_imgs_grey = [
        rgb2gray(img) for img in generated_remaining_imgs_arr]
    real_imgs_grey = [rgb2gray(img) for img in real_imgs_arr]
    g_rem_contrast = contrast(generated_remaining_imgs_grey)
    r_contrast = contrast(real_imgs_grey)
    inheritance = 1 - (abs(r_contrast - g_rem_contrast) /
                       max(r_contrast, g_rem_contrast))
    return inheritance
