"""
author:         Tola Shobande
name:           main.py
date:           30/09/2024
description:
"""

import os
import args
import torch
import logging
import torchvision
import utils.utils
from uuid import uuid4
from utils.utils import *
from model import EmbeddingNet
from torchvision import models
from data import ProjectDataset
from timm.utils import AverageMeter
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Log')


def train(train_laoder, model, criterion, optimizer, scaler, device, epoch, scheduler):
    model.train()
    losses = AverageMeter()
    
    for i, (images, labels) in enumerate(train_laoder):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update
        
        losses.update(loss.item(), images.size(0))
        
        log = format_log_message(mode='Train', i=i, epoch=epoch, loss=losses.avg)
        logger.info(log)
    
    scheduler.step()
    return losses.avg


def eval(val_loader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            with torch.amp.autocast:
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            losses.update(loss.item(), images.size())
    return losses.avg


def main():
    train_data = ProjectDataset(mode='train',root_dir='')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    train_len = len(train_loader.dataset)
    
    val_data = ProjectDataset(mode='val', root_dir='', seed=args.seed)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    val_len = len(val_loader.dataset)

    test_data = ProjectDataset(mode='test', root_dir='', seed=args.seed)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    test_len = len(test_loader.dataset)
    
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    pretrained_model = models.resnet50(weights="IMAGENET1K_V2")
    model = EmbeddingNet(model=pretrained_model)
    model.to(device)
    model_name = model.model.__class__.__name_
    
    if not os.path.exists("./checkpoint/"):
        os.mkdir("./checkpoint/")


if __name__ == '__main__':
    main()
