import os
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader
import imageio
import pickle

class DanceImageDataset(Dataset):
    """
    generate dance image dataset
    TODO: interpolation
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
        self.transform = transform
        with open(list_pkl_file, 'rb') as f:
            self.filename_list=pickle.load(f)  

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, self.filename_list[idx])
        stick_name = os.path.join(self.stick_dir, self.filename_list[idx])
        image = imageio.imread(img_name)
        stick = imageio.imread(stick_name)

        sample = {'image': image, 'stick': stick}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, stick = sample['image'], sample['stick']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img_transform = transform.resize(image, (new_h, new_w))
        stk_transform = transform.resize(stick, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively

        return {'image': img_transform, 'stick': stk_transform}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, stick = sample['image'], sample['stick']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        stick = stick.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'stick': torch.from_numpy(stick)}