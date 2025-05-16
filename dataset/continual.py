import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import json
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


import logging

logger = logging.getLogger()

class ImageDataset(Dataset):
    def __init__(self,
                 data_root,
                 meta_file="", # meta file path
                 resize=240,
                 mode="train",
                 aug=False,
                 with_fg_mask=False,
                 test_class='None'
                 ):

        self.data_root = data_root
        self.resize = resize
        self.mode = mode
        self.test_class = test_class
        self.with_fg_mask = with_fg_mask
        self.aug = aug

        if isinstance(meta_file, str):
            meta_info = json.load(open(meta_file, 'r'))
        else:
            meta_info = meta_file

        self.data_list = []
        if self.mode == "train":
            meta_info = meta_info[mode]
            for cls_name, data_list in meta_info.items():
                self.data_list.extend(data_list)
                # for data in data_list:
                #     if data["anomaly"] == 0:
                #         self.data_list.append(data)
            self.class_names = list(meta_info.keys())
        else:
            meta_info = meta_info[mode][test_class]
            self.data_list.extend(meta_info)
            self.class_names = [test_class]

        
        self.resize_img_transform = transforms.Resize((self.resize, self.resize), interpolation=Image.BICUBIC)
        self.resize_mask_transform = transforms.Resize((self.resize, self.resize), interpolation=Image.NEAREST)
        self.aug_transform = A.Compose([
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0, rotate_limit=0, p=0.2),
            # A.Rotate(limit=30, p=0.5),
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        data = self.data_list[idx]
        img_path, mask_path, cls_name, anomaly = data["img_path"], data["mask_path"], data["cls_name"], data["anomaly"]
        
        img_path = os.path.join(self.data_root, img_path)
        mask_path = os.path.join(self.data_root, mask_path)
        if self.with_fg_mask:
            fg_mask_path = img_path.replace(self.data_root, "sam2_fg_mask")
            fg_mask_path = fg_mask_path[:-3] + "png"

        image = Image.open(img_path).convert('RGB')
        image = ImageOps.exif_transpose(image)
        image = self.resize_img_transform(image)

        if anomaly == 0:
            mask = Image.fromarray(np.zeros((self.resize, self.resize)), mode='L')
        else:
            mask = np.array(Image.open(mask_path).convert('L')) > 0
            mask = Image.fromarray(mask.astype(np.uint8) * 255, mode='L')
            mask = self.resize_mask_transform(mask)

        if self.with_fg_mask:
            fg_mask = np.array(Image.open(fg_mask_path).convert('L')) > 0
            fg_mask = Image.fromarray(fg_mask.astype(np.uint8) * 255, mode='L')
            fg_mask = self.resize_mask_transform(fg_mask)
            
        else:
            fg_mask = torch.zeros(1, self.resize, self.resize)

        if self.mode == "train" and self.aug:
            image = np.array(image).astype(np.float32)
            mask = np.array(mask)
            augmented = self.aug_transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            if self.with_fg_mask:
                fg_mask = np.array(fg_mask)
                fg_mask = self.aug_transform(mask=fg_mask)['mask']

        else:
            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)
            if self.with_fg_mask:
                fg_mask = transforms.ToTensor()(fg_mask)

        return {"image": image, "mask": mask, "fg_mask": fg_mask, "cls_name": cls_name, "anomaly": anomaly}

    def __len__(self):
        return len(self.data_list)



if __name__ == '__main__':

    ds = ImageDataset(is_train=True)
