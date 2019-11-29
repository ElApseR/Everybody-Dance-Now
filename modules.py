import os
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader
import imageio

class DanceImageDataset(Dataset):
    """
    generate dance image dataset
    """
    def __init__(self, image_dir, stick_dir, list_pkl_file, transform=None):
        """
        SEE https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

        Args:
            image_dir (string): Directory to frame image
            stick_dir (string): Directory to stick pose image
            list_pkl_file (string): Frame file name of image and stick
        """
        self.image_dir = image_dir
        self.stick_dir = stick_dir
        self.list_pkl_file = list_pkl_file
        self.transform = transform

    def __len__(self):
        return len(self.list_pkl_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, self.list_pkl_file[idx])
        stick_name = os.path.join(self.stick_dir, self.list_pkl_file[idx])
        image = imageio.imread(img_name)
        stick = imageio.imread(stick_name)

        sample = {'image': image, 'stick': stick}

        if self.transform:
            sample = self.transform(sample)

        return sample