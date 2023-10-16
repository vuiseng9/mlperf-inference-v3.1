import os
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
    InterpolationMode
)

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

    val_dataset = ImageFolder(os.path.join(dataset_dir, 'val'), val_transforms)

    val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=n_worker, 
        sampler=val_sampler, drop_last=False)

    return val_loader