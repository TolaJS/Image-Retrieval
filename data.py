"""
author:         Tola Shobande
name:           data.py
date:           30/09/2024
description:    
"""

import os
import re
import args
import torch
import pandas as pd
from PIL import Image
from utils.utils import set_seed
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision import datasets, transforms as T

set_seed(args.seed)


class ModifiedImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        class_name = self.classes[label]
        path = self.samples[index][0]

        return img, label, class_name, path


class CSVImageDataset(Dataset):
    def __init__(self, csv_root, root_dir, transform=None):
        self.dataset = pd.read_csv(csv_root)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.dataset.iloc[idx, 0])
        img = Image.open(img_path).convert('RGB')
        label = self.dataset.iloc[idx, 1]

        match = re.search(r'./Test/(.+?)\.[^.]+$', img_path)
        class_name = match.group(1) if match else "Unknown"

        if self.transform:
            img = self.transform(img)
        return img, label, class_name, img_path


class ProjectDataset:
    def __init__(self, mode, root_dir, csv_root=None):

        self.mode = mode
        self.full_dataset = ModifiedImageFolder(root=root_dir, transform=self.data_transforms())

        if self.mode == 'train':
            self.dataset, _ = self.split_dataset(args.seed)
        elif self.mode == 'val':
            _, self.dataset = self.split_dataset(args.seed)
        elif self.mode == 'test':
            self.dataset = CSVImageDataset(csv_root, root_dir, transform=self.data_transforms())

    def data_transforms(self):
        if self.mode == 'train':
            transforms = [
                T.Resize((224, 224)),  # Might need to change this for different models
                T.RandomRotation(45),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(),
                T.RandomErasing(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
            return T.Compose(transforms)
        elif self.mode in ['val', 'test']:
            transforms = [
                T.Resize((224, 224)),  # Again, might need to change this
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
            return T.Compose(transforms)
        else:
            raise ValueError('Invalid mode')

    def split_dataset(self, seed=0):
        total_size = len(self.full_dataset)
        train_size = int(0.9 * total_size)  # 90% for training
        val_size = total_size - train_size  # 10% for validation

        generator = torch.Generator().manual_seed(seed)

        train_dataset, val_dataset = random_split(self.full_dataset, [train_size, val_size], generator=generator)
        return train_dataset, val_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
