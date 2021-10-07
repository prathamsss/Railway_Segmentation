import os
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, Rotate, Rotate, RandomBrightness, RandomBrightnessContrast
import cv2
from albumentations.augmentations import transforms as T
import argparse


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(path):
    train_x = sorted(glob(os.path.join(path, "Train", "Images", "*.jpg")))
    train_y = sorted(glob(os.path.join(path, "Train", "Masks", "*.png")))

    valid_x = sorted(glob(os.path.join(path, "Valid", "Images", "*.jpg")))
    valid_y = sorted(glob(os.path.join(path, "Valid", "Masks", "*.png")))

    return train_x, train_y, valid_x, valid_y


def augment_data(images, masks, save_path, augment=True):
    size = (1200, 900)

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        name_img = x.split('/')[-1]
        name_mask = y.split('/')[-1]

        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.imread(y)

        if augment == True:
            aug = HorizontalFlip(p=0.8)
            augmented = aug(image=x, mask=y)
            x1 = augmented['image']
            y1 = augmented['mask']

            aug = Rotate(limit=15, p=0.8)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            aug = RandomBrightness(p=0.5)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = T.RandomSnow(brightness_coeff=1.8, always_apply=True, p=0.5)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = T.RandomRain(p=0.5, always_apply=True)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']

            aug = T.RandomSunFlare(p=0.4, always_apply=False)
            augmented = aug(image=x, mask=y)
            x6 = augmented['image']
            y6 = augmented['mask']

            X = [x, x1, x2, x3, x4, x5, x6]
            Y = [y, y1, y2, y3, y4, y5, y6]




        else:
            X = [x]
            Y = [y]
        index = 0
        for i, m in zip(X, Y):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            tmp_img_name = f"{name_img.split('.')[0]}_{index}.jpg"
            tmp_mask_name = f"{name_mask.split('.')[0]}_{index}.png"

            img_path = os.path.join(save_path, "Images", tmp_img_name)
            mask_path = os.path.join(save_path, "Masks", tmp_mask_name)
            index = +1
            print(img_path)

            cv2.imwrite(img_path, i)
            cv2.imwrite(mask_path, m)




parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="Path cotaining Dataset with train and test folders",required=True)
parser.add_argument("--dataset_type", help="Augementation for trai for valid set: Value could be 'Train' OR 'Valid'",required=True )
parser.add_argument("--augment", help="  'True' OR 'False' value - to do augmentation OR just to change reshape, i.e, False",required=True)

args = parser.parse_args()


path = args.dir
print(path)
train_x, train_y, valid_x, valid_y = load_data(args.dir)
print(len(train_x),len(train_y),len(valid_x), len(valid_y))

create_dir(os.path.join(args.dir,"data_aug/Train/Images"))
create_dir(os.path.join(args.dir,"data_aug/Train/Masks"))

create_dir(os.path.join(args.dir,"data_aug/Valid/Masks"))
create_dir(os.path.join(args.dir,"data_aug/Valid/Masks"))

augment_data(train_x, train_y, save_path=os.path.join(args.dir,"/data_aug",args.dataset_type), augment=args.augment)

