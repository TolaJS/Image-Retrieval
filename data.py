"""
author:         Tola Shobande
name:           data.py
date:           30/09/2024
description:    
"""

from torchvision import datasets, transforms as T


class ModifiedImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)

        class_name = self.classes[label]

        path = self.samples[index][0]

        return img, label, class_name, path


class ProjectDataset:
    def __init__(self, mode, root_dir):

        self.mode = mode
        self.dataset = ModifiedImageFolder(root=root_dir, transform=self.data_transforms())

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
        elif self.mode == 'val':
            transforms = [
                T.Resize((224, 224)),  # Again, might need to change this
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
            return T.Compose(transforms)
        else:
            raise ValueError('Invalid mode')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
