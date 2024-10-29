"""
author:         Tola Shobande
name:           utils.py
date:           30/09/2024
description:
"""

import args
import torch
import random
import numpy as np
from torchvision import models


def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # Numpy module
    torch.manual_seed(seed_value)  # Sets the seed for generating random numbers for PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    # print(f"Random seed set to: {seed_value}")


def get_model_and_features(model: torch.nn.Module):
    if isinstance(model, (models.ResNet, models.GoogLeNet, models.Inception3)):
        out_features = model.fc.in_features
        model.fc = torch.nn.Identity()

    elif isinstance(model, models.EfficientNet):
        out_features = model.classifier[1].in_features
        model.classifier = torch.nn.Identity()

    else:
        raise ValueError(
            f"Unsupported model type: Model is meant for torchvison's "
            f"GoogLeNet, Inception3, ResNet or EfficientNet")

    return model, out_features


def format_log_message(epoch, train_loss, val_loss):
    """Log formatting function."""
    return (f'| Epoch:{epoch + 1}/{args.epochs} | Train_Loss:{train_loss:8.3f} | '
            f'Val_Loss:{val_loss:8.3f}')


def print_summary(logger, name, train_size, val_size, test_size):
    """Print summary model and data statistics."""
    logger.info('*' * 17 + 'Summary' + '*' * 17)
    logger.info(f'# Model: {name}')
    logger.info(f'# Train size: {train_size}')
    logger.info(f'# Validation size: {val_size}')
    logger.info(f'# Test size: {test_size}')
    logger.info(f'# Epochs: {args.epochs}')
    logger.info('*' * 41 + '\n')
