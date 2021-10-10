import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class RailData(Dataset):
    def __init__(self, images_path, masks_path, size=None, augmentation=None, preprocessing=None):

        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)
        self.size = size

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, index):
        """ Reading image """

        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        if self.size:
            image = cv2.resize(image, self.size)
            mask = cv2.resize(mask, self.size)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)

            image, mask = sample['image'], sample['mask'],

        image = image / 255.0  ## (512, 512, 3)
        image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        #             mask = cv2.resize(mask, self.size)
        mask = mask / 255.0  ## (512, 512)
        mask = np.expand_dims(mask, axis=0)  ## (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        #         image = np.transpose(image, (2, 0, 1))
        #         mask = np.expand_dims(mask, axis=0)

        return image, mask

    def __len__(self):
        return self.n_samples

