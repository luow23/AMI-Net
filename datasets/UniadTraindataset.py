import os
# import tarfile
from PIL import Image
# import urllib.request
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import imgaug.augmenters as iaa
import glob
from datasets.perlin import rand_perlin_2d_np
import numpy as np
import cv2
import random
CLASS_NAMES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
    'toothbrush', 'transistor', 'wood', 'zipper'
]
class UniTrainData(Dataset):
    def __init__(self, opt, resize=256):
        self.images = []
        root = opt.data_root
        self.transform_x = transforms.Compose([
            transforms.Resize(resize, Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std= [0.229, 0.224, 0.225])])
        for v in CLASS_NAMES:
            path_list = os.path.join(root, v, 'train')
            imgs = GetFiles(path_list, ["JPG", "jpg", "bmp", "png"])
            imgs = [img for img in imgs]
            self.images.extend(imgs)

    def __getitem__(self, item):
        img_path = self.images[item]
        x = Image.open(img_path).convert('RGB').resize((256, 256))
        img = self.transform_x(x)

        return img
    def __len__(self):
        return len(self.images)

def GetFiles(file_dir, file_type, IsCurrent=False):
    file_list = []
    for parent, dirnames, filenames in os.walk(file_dir):
        for filename in filenames:
            for type in file_type:
                if filename.endswith(('.%s' % type)):
                    file_list.append(os.path.join(parent, filename))

        if IsCurrent == True:
            break
    return file_list