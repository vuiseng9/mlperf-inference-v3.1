import argparse
import logging as log
import sys

import time
import numpy as np

import os
import torch
import torchvision
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
    InterpolationMode
)
from torch.utils.data import DataLoader
from copy import copy


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_imagenet_valset_loader(dataset_dir, batch_size,
    n_worker=16,
    image_size=224, 
    crop_pct=0.875,
    mode=InterpolationMode.BICUBIC,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)):

    size = int(image_size / crop_pct)

    preprocess_list = [
        Resize(size, interpolation=mode),
        CenterCrop(image_size),
        ToTensor(),
        Normalize(mean=mean, std=std)]

    val_transforms = Compose(preprocess_list)

    val_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'val'), val_transforms)

    val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=n_worker, 
        sampler=val_sampler, drop_last=False)

    return val_loader

def load_model(model_path):
    model = torch.jit.load(model_path)
    model = torch.jit.freeze(model.eval())
    return model

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

if __name__ == "__main__":

    # model_path = "./quantized_model_rn50.pt"
    model_path = "./quantized_resnet50.pt"
    datadir = "/data1/datasets/imgnet-train1k-val1k-dev/"
    datadir = "./ILSVRC2012_img_val"
    batch_size=1

    val_loader = create_imagenet_valset_loader(datadir, batch_size=batch_size)
    model = load_model(model_path)

    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for i, (input_, target) in enumerate(val_loader):
            # print(i, target)
            probs = model(input_)
            acc1, acc5 = accuracy(probs.detach(), target, topk=(1, 5))
            top1.update(acc1.item(), input_.size(0))
            top5.update(acc5.item(), input_.size(0))
            print(f"batch {i:3} | total images: {top1.count:5}, top1.avg: {top1.avg:5.2f}, top5.avg: {top5.avg:5.2f}")
            if top1.count >= 10:
                break
