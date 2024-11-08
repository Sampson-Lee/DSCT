# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .caer import build as build_caer
from .emotic import build as build_emotic

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):

    if args.dataset_file == 'caer':
        return build_caer(image_set, args)
    if args.dataset_file == 'emotic':
        return build_emotic(image_set, args)

    raise ValueError(f'dataset {args.dataset_file} not supported')
