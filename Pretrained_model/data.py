import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class RailData(Dataset):
    def __init__(self, images_path, masks_path, size):  # size=(512,512)

        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)
        self.size = size

    #         self.size = size

    def __getitem__(self, index):
        """ Reading image """

        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        preprocess = transforms.Compose([
            transforms.Resize(self.size, 2),
            transforms.ToTensor()
        ])

        X = Image.fromarray(image).convert('RGB')
        X = preprocess(X)

        trfresize = transforms.Resize(self.size, 2)
        trftensor = transforms.ToTensor()
        Y = Image.fromarray(mask).convert('L')
        Y = trftensor(trfresize(Y))
        Y = Y.type(torch.float)

        return X, Y

    def __len__(self):
        return self.n_samples