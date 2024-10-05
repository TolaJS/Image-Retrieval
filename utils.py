"""
author:         Tola Shobande
name:           utils.py
date:           30/09/2024
description:
"""
from torchvision import models
import torch

def get_model_info(model:torch.nn.Module):
    if isinstance(model, models.ResNet):
        num_features = model.fc.in_features
        model.fc = torch.nn.Identity()
        
    if isinstance(model, models.GoogLeNet):
        num_features = model.fc.in_features
        model.fc = torch.nn.Identity()
        
    elif isinstance(model, models.EfficientNet):
        num_features = model.classifier[1].in_features
        model.classifier = torch.nn.Identity()
        
    else:
        raise ValueError("Unsupported model type: Model is meant for torchvison's GoogLeNet, Inception3, ResNet or EfficientNet")
    
    return model, num_features

    