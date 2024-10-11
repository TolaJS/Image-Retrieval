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
from loss import TripletLoss
from datetime import datetime
from model import EmbeddingNet
from torchvision import models
from data import ProjectDataset
from timm.utils import AverageMeter
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Log')


def train(train_laoder, model, criterion, optimizer, scaler, device, scheduler):
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
        scaler.update()

        losses.update(loss.item(), images.size(0))

    scheduler.step()
    return losses.avg


def evaluate(val_loader, model, criterion, device):
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
    train_data = ProjectDataset(mode='train', root_dir='')  #todo dataset path
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    train_len = len(train_loader.dataset)

    val_data = ProjectDataset(mode='val', root_dir='')
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    val_len = len(val_loader.dataset)

    test_data = ProjectDataset(mode='test', root_dir='')
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

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = TripletLoss
    scaler = torch.cuda.amp.GradScaler()

    log_dir = (f"logs/runs/"
               f"{model_name}/"
               f"lr_{args.lr}_bs_{args.batch_size}/"
               f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_"
               f"{uuid4().hex[:6]}")
    writer = SummaryWriter(log_dir)  # TENSORBOARD
    handler = logging.FileHandler(f'{log_dir}/log.txt')
    logger.addHandler(handler)

    if args.eval:
        if args.model_path:
            model.load_state_dict(torch.load(args.model_path))
            model.to(device)
        else:
            raise ValueError("Model path not set in args.py")
        val_loss = evaluate(val_loader, model, criterion, device)
        logger.info("=> Testing Results")
        logger.info("Validation Loss: {:.2f}".format(val_loss))
    else:

        for epoch in range(args.epochs):
            train_loss = train(train_loader, model, criterion, optimizer, scaler, device, epoch, scheduler)
            val_loss = evaluate(val_loader, model, criterion, device)

            writer.add_scalar('Train/Loss', train_loss, epoch)
            writer.add_scalar('Validation/Loss', val_loss, epoch)
            log = format_log_message(epoch, train_loss, val_loss)
            logger.info(log)

            filename = f"Model-{model_name}-lr_{args.lr}_bs_{args.batch_size}-E{epoch + 1}.pth"
            save_path = os.path.join(args.save_path, filename)
            torch.save(model.state_dict(), save_path)
        logger.info("=> Training Finished")
        handler.close()


if __name__ == '__main__':
    main()
