# image_processing.py

import cv2
import numpy as np
from scipy import stats

_img_cache = {
    'im_name': None,
    'raw_image_np': None
}

def mean_window(data, axis):
    res = np.sum(data, axis=axis)
    return res

def roll(a, b_shape, dx=1, dy=1):
    shape = a.shape[:-2] + \
        ((a.shape[-2] - b_shape[-2]) // dy + 1,) + \
        ((a.shape[-1] - b_shape[-1]) // dx + 1,) + \
        b_shape
    strides = a.strides[:-2] + \
        (a.strides[-2] * dy,) + \
        (a.strides[-1] * dx,) + \
        a.strides[-2:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def sliding_window(data, win_shape, fcn, dx=1, dy=1):
    n = data.ndim
    result = fcn(roll(data, win_shape, dx, dy), axis=(n, n+1))
    return result

def tile_array(a, b0, b1):
    r, c = a.shape
    rs, cs = a.strides
    x = np.lib.stride_tricks.as_strided(a, (r, b0, c, b1), (rs, 0, cs, 0))
    return x.reshape(r*b0, c*b1)

def edge_density(img_np: np.array,
                 win_size: int,
                 win_step: int = 10,
                 canny_1: float = 41,
                 canny_2: float = 207) -> np.array:
    dxy = win_step
    mid = cv2.Canny(img_np, canny_1, canny_2)

    result = sliding_window(mid, (win_size, win_size),
                            mean_window, dx=dxy, dy=dxy) // ((win_size*win_size))
    result = tile_array(result, dxy, dxy)
    h_pad = img_np.shape[0] - result.shape[0]
    w_pad = img_np.shape[1] - result.shape[1]
    result = np.pad(result, ((
        h_pad//2, h_pad//2+img_np.shape[0] % 2), (w_pad//2, w_pad//2+img_np.shape[1] % 2)), 'edge')
    return result
