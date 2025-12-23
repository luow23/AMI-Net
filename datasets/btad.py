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

# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
CLASS_NAMES = [
'01', '02', '03'
]
random.seed(1)
class BTADDataset(Dataset):
    def __init__(self,
                 dataset_path='D:\IMSN-LW\dataset\BTech_Dataset_transformed',
                 class_name='01',
                 is_train=True,
                 resize=256,
                 ):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize

        self.x, self.y, self.mask = self.load_dataset_folder()
        self.len = len(self.x)
        self.name = []
        # set transforms
        self.transform_x = transforms.Compose([
            transforms.Resize((resize,resize), Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        self.transform_mask = transforms.Compose(
            [transforms.Resize((resize,resize), Image.NEAREST),
             transforms.ToTensor()])


        for i in range(self.len):
            names = self.x[i].split("\\")
            name = names[-2]+"!"+names[-1]
            self.name.append(name)

    def __getitem__(self, idx):

        x, y, mask, name = self.x[idx], self.y[idx], self.mask[idx], self.name[idx]
        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, self.resize, self.resize])
        else:
            mask = Image.open(mask).convert('L')
            mask = self.transform_mask(mask)
        # if self.train_stage == 1:
        #     return x, y, mask
        # elif self.train_stage == 2:
        return x, y, mask, name


    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted(
                [os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.bmp') or f.endswith('.png')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'ok':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                pic_name = '.bmp' if self.class_name=='03' else '.png'
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + pic_name) for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)


class FewshotBTADDataset(Dataset):
    def __init__(self,
                 dataset_path='D:\IMSN-LW\dataset\BTech_Dataset_transformed',
                 class_name='01',
                 is_train=True,
                 resize=256,
                 k=4,
                 ):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.shot = k
        self.x, self.y, self.mask = self.load_dataset_folder()
        self.len = len(self.x)
        self.name = []
        # set transforms
        self.transform_x = transforms.Compose([
            transforms.Resize((resize,resize), Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std= [0.229, 0.224, 0.225])])

        self.transform_mask = transforms.Compose(
            [transforms.Resize((resize, resize), Image.NEAREST),
             transforms.ToTensor()])


        for i in range(self.len):
            names = self.x[i].split("\\")
            name = names[-2]+"!"+names[-1]
            self.name.append(name)

    def __getitem__(self, idx):

        x, y, mask, name = self.x[idx], self.y[idx], self.mask[idx], self.name[idx]
        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, self.resize, self.resize])
        else:
            mask = Image.open(mask).convert('L')
            mask = self.transform_mask(mask)
        # if self.train_stage == 1:
        #     return x, y, mask
        # elif self.train_stage == 2:
        return x, y, mask, name


    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted(
                [os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.bmp') or f.endswith('.png')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'ok':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                pic_name = '.bmp' if self.class_name=='03' else '.png'
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + pic_name) for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return random.sample(list(x), self.shot), random.sample(list(y), self.shot), random.sample(list(mask), self.shot)

def tensor_to_np(tensor_img):
    np_img = np.array(tensor_img)
    np_img = np.transpose(np_img, (1, 2, 0))
    if np_img.shape[2] == 3:
        np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    return np_img
def denormalize(img):
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    x = (((img.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x
# if __name__ == '__main__':
    # mvtec = MVTecDataset()
    # x, y, mask, aug_x, aug_mask,_  = mvtec[0]
    # # print(x)
    # # print(y.shape)
    # # print(mask.shape)
    # # print(aug_x)
    # # print(aug_mask)
    # x = tensor_to_np(x)
    # cv2.imwrite('luowei1.jpg', x*255)
    # np_img = tensor_to_np(aug_x)
    # cv2.imwrite('luowei.jpg', np_img*255)
    # aug_mask = tensor_to_np(aug_mask)
    # cv2.imwrite('luowei2.jpg', aug_mask*255)
