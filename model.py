"""
author:         Tola Shobande
name:           model.py
date:           30/09/2024
description:
"""

import torch
import torch.nn as nn
from torchvision import models
from utils.utils import modify_fc_layer


class EmbeddingNet(nn.Module):
    def __init__(self, model: nn.Module, embedding_dim=None, freeze_weights=True):
        super(EmbeddingNet, self).__init__()
        self.model = model

        if freeze_weights:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model = modify_fc_layer(self.model, embedding_dim=embedding_dim)
    
    def forward(self, x):
        x = self.model(x)
        return x   
