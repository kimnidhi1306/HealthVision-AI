# data_handling.py

import pandas as pd
import numpy as np
from pathlib import Path

IMAGE_PATH = Path("/content/Data/MCF-7 cell populations Dataset/images")
IMAGE_PATH_LIST = list(IMAGE_PATH.glob("*.png"))
IMAGE_PATH_LIST = sorted(IMAGE_PATH_LIST)

print(f'Total Images = {len(IMAGE_PATH_LIST)}')

MASK_PATH = Path("/content/Data/MCF-7 cell populations Dataset/masks")
MASK_PATH_LIST = list(MASK_PATH.glob("*.png"))
MASK_PATH_LIST = sorted(MASK_PATH_LIST)

print(f'Total Masks = {len(MASK_PATH_LIST)}')

def im_load(name):
    f = open(name, "rb")
    chunk = bytearray(f.read())
    chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
    img = cv2.imdecode(chunk_arr, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    f.close()

    return img

def im_save(target_path: str, image_np: np.ndarray):
    filename, file_extension = os.path.splitext(target_path)
    is_success, im_buf_arr = cv2.imencode(file_extension, image_np)
    return im_buf_arr.tofile(target_path)