"""
author:         Tola Shobande
name:           model.py
date:           30/09/2024
description:
"""

import torch
import torch.nn as nn
from torchvision import models


class NeuralNet(nn.Model):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    
    def __init__(self,freeze_weights: bool=True):
        super(NeuralNet, self).__init__()
        self.model = models.resnet50(weights = "IMAGENET1K_V2")
        
        if freeze_weights:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model = nn.Sequential(*list(self.model.children())[:-1])
    
    def forward(self, x):
        return self.model(x)
    

class EmbeddingNet(nn.Model):
    def __init__(self, backbone, embedding_dim=128):
        super(EmbeddingNet, self).__init__()
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        
        
        dummy_input = torch.randn(1, 3, 224, 224)  # Adjust size if needed
        with torch.no_grad():
            output = self.backbone(dummy_input)
        num_features = output.shape[1]
        
