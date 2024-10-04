"""
author:         Tola Shobande
name:           model.py
date:           30/09/2024
description:
"""

import torch
import torch.nn as nn
from torchvision import models
  

class EmbeddingNet(nn.Module):
    def __init__(self, model:nn.Module, embedding_dim=128, freeze_weights=True):
        super(EmbeddingNet, self).__init__()
        self.model = model
        
        if freeze_weights:
            for param in self.model.parameters():
                param.requires_grad = False

        if isinstance(self.model, models.ResNet):
            num_features = self.model.fc.in_features
            self.model.fc = nn.Identity
            
        if isinstance(self.model, models.GoogLeNet):
            num_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
            
        elif isinstance(self.model, models.EfficientNet):
            num_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Identity()
            
        else:
            raise ValueError("Unsupported model type: Model is meant for torchvison's GoogLeNet, Inception3, ResNet or EfficientNet")
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(num_features, embedding_dim)
    
    def forward(self,x):
        x = self.model(x)
        if isinstance(x, tuple):
            x = x[0]
            
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x   
