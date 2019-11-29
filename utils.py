import numpy as np
import torch 
from torch.autograd import Variable 
import pickle
import os

def image_preprocess(x):
    """
    reshape image RGB to BGR
    HWC to CHW
    :param x:
    :return:
    """
    image = np.array(x, dtype=np.float32)
    image -= np.mean(image)
    image = image[:, :, ::-1] - np.zeros_like(image)
    image = image.transpose((2, 0, 1))

    return image

def save_img_list(img_dir, pkl_dir):
    image_list = os.listdir(img_dir)
    with open(pkl_dir, 'wb') as handle:
        pickle.dump(image_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_img_list(pkl_dir):
    with open(pkl_dir, 'rb') as handle:
        pkl = pickle.load(handle)
        return pkl