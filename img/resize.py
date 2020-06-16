""" Utils for image processing. """
import cv2
import numpy as np
import math
import torch

def resize(x, size):
    x = np.array(x)
    x = x[:, :, :3]
    h, w = x.shape[0], x.shape[1]
    scale = max(size[0] / h,size[1] / w)
    x = cv2.resize(x, (int(math.ceil(scale * w)),
                        int(math.ceil(scale * h))))
    crop = [math.ceil((x.shape[1] - size[1]) / 2),
            math.floor((x.shape[1] - size[1]) / 2),
            math.ceil((x.shape[0] - size[0]) / 2),
            math.floor((x.shape[0] - size[0]) / 2)]
    return x[crop[2]: x.shape[0] - crop[3], crop[0]: x.shape[1] - crop[1], :].astype(np.uint8)