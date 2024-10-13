"""
author:         Tola Shobande
name:           model.py
date:           30/09/2024
description:
"""

import torch
import torch.nn as nn
from torchvision import models
from utils.utils import get_model_and_features


class EmbeddingNet(nn.Module):
    def __init__(self, model: nn.Module, embedding_dim=None, freeze_weights=True):
        super(EmbeddingNet, self).__init__()
        self.model = model

        if freeze_weights:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model, out_features = get_model_and_features(self.model)
        if embedding_dim is not None:   # To experiment with different dimensional feature vectors
            self.model.fc = nn.Linear(out_features, embedding_dim)
    
    def forward(self, x):
        x = self.model(x)
        return x   
