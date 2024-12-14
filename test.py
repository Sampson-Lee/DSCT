# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import cv2
import sys
import argparse
from pathlib import Path
from typing import Iterable
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import numpy as np

import torch
import torchvision
import util.misc as utils

from models import build_model
from datasets.caer import make_face_transforms
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from IPython import embed

from sklearn.metrics import accuracy_score, confusion_matrix, average_precision_score
import json
from tqdm import tqdm

def face_matching(refer, bboxes):
    area_refer = (refer[2] - refer[0] + 1) * (refer[3] - refer[1] + 1)
    area_bboxes = (bboxes[:, 2] - bboxes[:, 0] + 1) * (bboxes[:, 3] - bboxes[:, 1] + 1)
    
    xx1 = np.maximum(refer[0], bboxes[:, 0])
    yy1 = np.maximum(refer[1], bboxes[:, 1])
    xx2 = np.minimum(refer[2], bboxes[:, 2])
    yy2 = np.minimum(refer[3], bboxes[:, 3])

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / (area_refer + area_bboxes - inter)

    return ovr.argsort()[-1] # choose the largest IoU


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # print(scores)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h,
                          img_w, img_h
                          ], dtype=torch.float32)
    return b

def get_images(in_path):
    img_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            else:
                print('invalid image')
    return img_files


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Segmentation Only. If set, only the mask head will be trained")
    parser.add_argument('--pretrained_weights', type=str, default='/home/xinpeng/Deformable-DETR-main/r50_deformable_detr-checkpoint.pth', 
                        help="Path to the pretrained model.")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=10, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--interpolate_factor', default=0.5, type=float)
    parser.add_argument('--noise', default=0.2, type=float)
    parser.add_argument('--model', default='deformable_transformer', type=str)
    parser.add_argument('--detr', default='deformable_detr', type=str)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_em', default=5, type=float,
                        help="Emotion coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--em_loss_coef', default=5, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='caer')
    parser.add_argument('--data_path', type=str, default='/home/xinpeng/CAER_S/')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='/home/xinpeng/CAER_S/log_ddetr/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    parser.add_argument('--iou_thresh', default=0.01, type=float)
    parser.add_argument('--score_thresh', default=0.2, type=float)
    parser.add_argument('--json_path', type=str)
    parser.add_argument('--binary_flag', default=0, type=int)
    parser.add_argument('--face_num', default=0, type=int)
    parser.add_argument('--modality', default='', type=str, nargs="+")

    return parser

caer_list = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotic_list = ['Affection','Anger','Annoyance','Anticipation','Aversion','Confidence','Disapproval','Disconnection','Disquietment','Doubt/Confusion','Embarrassment','Engagement','Esteem','Excitement','Fatigue','Fear','Happiness','Pain','Peace','Pleasure','Sadness','Sensitivity','Suffering','Surprise','Sympathy','Yearning']

@torch.no_grad()
def infer(dataset, model, postprocessors, device, args):
    if args.dataset_file == 'emotic': emo_list = emotic_list
    if args.dataset_file == 'caer': emo_list = caer_list
    model.eval()
    duration = 0
    gt_list = []; pred_list = []
    gt_list_b = []; pred_list_b = []
    for orig_image, target in dataset:
        
        # evaluate the image with specific face number when face_num is not 0
        if 0<args.face_num<5:
            print('face number is', args.face_num)
            if len(target) != args.face_num: continue
        elif args.face_num>=5:
            print('face number is', args.face_num)
            if len(target) < 5: continue
        else: pass

        img_name = os.path.join(args.data_path, dataset.coco.imgs[target[0]["image_id"]]['file_name'])
        orig_image = Image.open(img_name)

        w, h = orig_image.size
        transform = make_face_transforms("val")
        dummy_target = {
            "size": torch.as_tensor([int(h), int(w)]),
            "orig_size": torch.as_tensor([int(h), int(w)])}
        image, targets = transform(orig_image, dummy_target)
        image = image.unsqueeze(0)
        image = image.to(device)

        start_t = time.perf_counter()
        outputs = model(image)
        end_t = time.perf_counter()
        infer_time = end_t - start_t
        duration += infer_time

        if args.binary_flag:
            probas = outputs['pred_logits'][0,:,:-1].sigmoid().cpu() # (num_query, emo_class)
        else:
            probas = outputs['pred_logits'][0,:,:-1].softmax(-1).cpu() # (num_query, emo_class)

        outputs["pred_boxes"] = outputs["pred_boxes"].cpu()
        bboxes = rescale_bboxes(outputs['pred_boxes'][0].cpu(), orig_image.size) # (num_query, 4)

        dets = torch.cat([bboxes, probas.mean(dim=-1).unsqueeze(-1)], dim=1)
        keep = nms(dets.numpy(), 0.01)
        probas = probas[keep]; bboxes = bboxes[keep]
        
        for ann_idx, ann in enumerate(target):
            img_bbox = ann["bbox"]
            img_bbox = [img_bbox[0], img_bbox[1], img_bbox[0]+img_bbox[2], img_bbox[1]+img_bbox[3]]

            if args.binary_flag:
                img_class_b = bin(ann["category_id"])[2:]
                img_class_b = np.array([0,]*(len(emo_list) - len(img_class_b)) + [int(x) for x in img_class_b])
                img_class = np.where(img_class_b==1)[0]
            else:
                img_class = ann["category_id"]
                img_class_b = np.zeros(len(emo_list))
                img_class_b[img_class] = 1
                img_class = [img_class]

            gt_list.append(img_class[0])
            gt_list_b.append(np.array(img_class_b))

            idx_sele = face_matching(img_bbox, bboxes)
            prob_sele = probas[idx_sele].data.numpy()

            prob_related, cls_related = prob_sele.max(), prob_sele.argmax()
            pred_list.append(cls_related)
            prob_related, cls_related = [prob_related, ], [cls_related, ]
            pred_list_b.append(prob_sele)

    avg_duration = duration / len(dataset.coco.imgs)
    print("Totally {} image; Avg. Time: {:.3f}s".format(len(dataset.coco.imgs), avg_duration))
    acc = accuracy_score(gt_list, pred_list)
    print('acc', acc)
    mAP = average_precision_score(np.array(gt_list_b), np.array(pred_list_b))
    print('mAP', mAP)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('DSCT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # if args.output_dir:
        # Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    model, _, postprocessors = build_model(args)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device)

    dataset = torchvision.datasets.CocoDetection(args.data_path, args.json_path)

    infer(dataset, model, postprocessors, device, args)