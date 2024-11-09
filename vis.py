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
import seaborn as sns

import torch
import torchvision
import util.misc as utils

from models import build_model
from datasets.caer import make_face_transforms
import matplotlib.pyplot as plt
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from IPython import embed
from sklearn.metrics import accuracy_score, confusion_matrix, average_precision_score
import json
import numpy as np
from tqdm import tqdm
from matplotlib.colors import ListedColormap
import torch.nn.functional as F
import random
from sklearn.manifold import TSNE

color_list = [(223,111,26),(107,173,55),(222,119,177),(0,153,204),(81,81,81),(64,64,241),
            (204,203,0),(78,78,125),(0,142,142),(1,101,251),(204,102,153),(2,184,111)] #(B,G,R)

def get_face_points(model):
    face_points = model.transformer.decoder.layers[-1].face_points.cpu().data.numpy()
    bs_n, sets_n, lvls_n, coords_n = face_points.shape
    # face_points = face_points.reshape(sets_n, lvls_n, coords_n)
    face_points = face_points[:,:,0,:].reshape(sets_n, coords_n)

    face_samples = model.transformer.decoder.layers[-1].face_samples.cpu().data.numpy()
    bs_n, sets_n, heads_n, lvls_n, pts_n, coords_n = face_samples.shape
    face_samples = face_samples.reshape(sets_n, heads_n*lvls_n*pts_n, coords_n)
    # face_samples = face_samples[:,:,0,0,:,:].reshape(sets_n, pts_n, coords_n) # used for showing some locations
    face_samples = np.clip(face_samples, 0, 1)
    
    return face_samples, face_points

def get_trajectory(model, image, target, device, keep, args):
    face_point_tracks = {}
    for i in range(len(keep)): face_point_tracks[i] = []
    for n in range(49, 0, -1):
        s = "%04d" % n
        checkpoint = torch.load(args.resume[:-8]+s+'.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        model.to(device)
        outputs = model(image)
        face_samples, face_points = get_face_points(model, keep)
        for ann_idx, ann in enumerate(keep):
            face_point_tracks[ann_idx].append(face_points[ann_idx])
    return face_point_tracks

def get_context_points(model):
    context_points = model.transformer.decoder.layers[-1].context_points.cpu().data.numpy()
    bs_n, sets_n, lvls_n, coords_n = context_points.shape
    context_points = context_points[0,:,0,:]

    context_samples = model.transformer.decoder.layers[-1].context_samples.cpu().data.numpy()
    bs_n, sets_n, heads_n, lvls_n, pts_n, coords_n = context_samples.shape
    context_samples = context_samples[0,:,:,:,:,:].reshape(sets_n, heads_n*lvls_n*pts_n, coords_n)

    context_points = np.clip(context_points, 0, 1)
    context_samples = np.clip(context_samples, 0, 1)

    relevance_values = model.transformer.decoder.layers[-1].relevance_values.cpu().data.numpy()[0,:,:]
    context_points_sp = model.transformer.decoder.layers[-1].context_points_sp.cpu().data.numpy()[0,:,:,:]
    context_points_se = model.transformer.decoder.layers[-1].context_points_se.cpu().data.numpy()[0,:,:,:]

    return context_samples, context_points, relevance_values, context_points_sp, context_points_se

def plot_trajectory(plot_image, face_point_tracks):

    overlay = plot_image.copy()
    h, w = plot_image.shape[:2]
    r = int(h // 200)

    for face_idx, key in enumerate(face_point_tracks.keys()):
        face_point_track = np.stack(face_point_tracks[key])
        face_point_track = face_point_track * (w, h)
        for idx, face_point in enumerate(face_point_track):
            cv2.circle(overlay, (int(face_point[0]), int(face_point[1])), r, color_list[face_idx], -1)    
            # color = (181+int((35-181)*idx/len(face_point_tracks)), 120+int((36-120)*idx/len(face_point_tracks)), 40+int((200-40)*idx/len(face_point_tracks))) # (255, 0, 0)->(0, 0, 255) B G R
            # if idx != face_point_tracks.shape[0]-1: 
            #     cv2.line(overlay, (int(face_point[0]), int(face_point[1])), (int(face_point_tracks[idx+1][0]), int(face_point_tracks[idx+1][1])), color, int(0.5*r))
        cv2.circle(overlay, (int(face_point_track[0][0]), int(face_point_track[0][1])), r, (0,0,255), -1)

    return overlay

def plot_points(plot_image, face_samples, face_points, color=None, context_samples=None, context_points=None, relevance_values=None):
    overlay = plot_image.copy()
    h, w = plot_image.shape[:2]
    r = int(h // 200)
    
    if context_points is not None:
        # context_samples = context_samples * (w, h); context_points = context_points * (w, h)
        for idx in range(len(context_points)):
            point = context_points[idx]
            # relev = relevance_values[idx]
            # cv2.circle(overlay, (int(point[0]*w), int(point[1]*h)), int(r/2)+int(3/2*r*relev), (0, 0, int(255*0.5)), -1) # (B, G, R) 0 191 255
            # cv2.circle(overlay, (int(point[0]), int(point[1])), int(r)+int(r*relev), (0, 0, 0), 1) # (B, G, R) 25 25 112
            cv2.circle(overlay, (int(point[0]*w), int(point[1]*h)), r, color, -1) # (B, G, R) 25 25 112
            cv2.circle(overlay, (int(point[0]*w), int(point[1]*h)), r, (0, 0, 0), 1) # (B, G, R) 25 25 112
            
            # for context_sample in context_samples[idx]:
            #     cv2.circle(overlay, (int(context_sample[0]), int(context_sample[1])), int((int(r)+int(r*relev))/2), tuple(int(x*0.5) for x in color), -1)
            #     cv2.circle(overlay, (int(context_sample[0]), int(context_sample[1])), int(r)+int(r*relev), (0, 0, 0), 1)

    # for face_sample in face_samples[0]:
    #     # embed()
    #     cv2.circle(overlay, (int(face_sample[0]*w), int(face_sample[1]*h)), r, (192,192,192), -1)
        # cv2.circle(overlay, (int(face_sample[0]), int(face_sample[1])), r, (0, 0, 0), 1)
    # for face_point in face_points:
    #     cv2.circle(overlay, (int(face_point[0]*w), int(face_point[1]*h)), 2*r, color, -1) # (B, G, R)
        # cv2.circle(overlay, (int(face_point[0]*w), int(face_point[1]*h)), 2*r, color, -1) # (B, G, R)
        # cv2.circle(overlay, (int(face_point[0]*w), int(face_point[1]*h)), 2*r, color, -1) # (B, G, R)
    # cv2.circle(overlay, (int(face_points[0]), int(face_points[1])), 2*r, (0, 0, 255), 5) # (B, G, R)
    # cv2.drawMarker(overlay, (int(face_points[0]), int(face_points[1])), markerSize=10*r, color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=5) # (B, G, R)

    # cv2.addWeighted(overlay, 0.4, plot_image, 1 - 0.4, 0, plot_image)
    return overlay

font=cv2.FONT_HERSHEY_TRIPLEX; font_scale=0.1; font_thickness=1
def plot_result(plot_image, bboxes, probas=None, classes=None, emo_list=None, color=[0,0,255]):
    """Plot Results
    bboxes is a list, size is [subjects]
    probs, classes is a list, size is [subjects, classes]
    """
    h, w = plot_image.shape[:2]
    r = int(w // 200)
    overlay = plot_image.copy()
    for i, value in enumerate(bboxes):
        
        bbox = bboxes[i]
        bbox = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
            ]).astype(np.int32)
        bbox = bbox.reshape((4, 2))
        # color = color_list[i]

        if probas is not None:
            pos = [bbox[3][0], bbox[3][1]]
            for j in range(len(probas[i])):
                prob = probas[i][j]
                class_ = int(classes[i][j])
                # label_str = emo_list[class_] + ': {:.2}'.format(prob)
                label_str = emo_list[class_] 

                # overlay = plot_image.copy()
                text_size, _ = cv2.getTextSize(label_str, font, w/2000, font_thickness)
                text_w, text_h = text_size
                x, y = pos
                # cv2.rectangle(overlay, (x, y), (x + text_w, y + 2*text_h), (255, 255, 255), -1)
                
                cv2.putText(overlay, label_str, (x, y + int(1.5*text_h)), font, w/2000, tuple(color), font_thickness)
                pos[1] += 2*text_h

        cv2.polylines(overlay, [bbox], True, tuple(color), r)

    cv2.addWeighted(overlay, 0.9, plot_image, 1 - 0.9, 0, plot_image)
    return plot_image

def plot_cfm(gt_list, pred_list):
    cfm = confusion_matrix(gt_list, pred_list)

    group_counts = ["{0:0.0f}".format(value) for value in cfm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in (cfm / np.sum(cfm, axis=1, keepdims=True)).flatten()]

    labels = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(len(emo_list), len(emo_list))

    ax = sns.heatmap(cfm, annot=labels, fmt='', cmap='Blues')
    ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels')
    ax.xaxis.set_ticklabels(emo_list); ax.yaxis.set_ticklabels(emo_list)

    save_cfm_path = os.path.join(*(img_name.split('/')[:-2])).replace('Val', 'log').replace('Data', 'log')
    plt.savefig('/'+save_cfm_path + '/cmf.jpg')

def plot_fea(images_, feature_maps_, save_name):
    feature_maps_ = F.interpolate(feature_maps_[-1].tensors, size=(224,224), mode='bilinear', align_corners=False)
    images_ = F.interpolate(images_, size=(224,224), mode='bilinear', align_corners=False)

    feature_map = feature_maps_[0].cpu().numpy().mean(0)
    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
    # feature_map[feature_map<0.3] = 0.0
    feature_map[feature_map<0.1] = 0.0

    mean = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std = torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    image_denorm = (((images_*std.to(device)) + mean.to(device))*255).int()
    image_denorm = image_denorm[0].cpu().numpy()

    plt.imshow(image_denorm.transpose(1, 2, 0))
    cmap_colors = plt.cm.viridis.colors
    cmap_colors[0] = (0, 0, 0, 0)
    cmap = ListedColormap(cmap_colors)
    plt.imshow(feature_map, cmap=cmap, alpha=0.5)
    plt.axis('off')
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.close()

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

def get_tsne(model, path):
    queries = model.hs_all.cpu().numpy()[-1, 0] # (level, n_bs, n_sbj_ctx, n_dim)
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(queries)
    plt.figure(figsize=(8, 8))
    plt.scatter(X_tsne[:300, 0], X_tsne[:300, 1], c='blue', marker='o')
    plt.scatter(X_tsne[300:, 0], X_tsne[300:, 1], c='red', marker='o')

    # context_queries = model.query_embed.weight.cpu().numpy()
    # subject_queries = model.query_embed_fetr.weight.cpu().numpy()
    # X = np.concatenate((context_queries, subject_queries))
    # tsne = TSNE(n_components=2, random_state=0)
    # X_tsne = tsne.fit_transform(X[:,256:])
    # plt.figure(figsize=(8, 8))
    # plt.scatter(X_tsne[:300, 0], X_tsne[:300, 1], c='blue', marker='o')
    # plt.scatter(X_tsne[300:, 0], X_tsne[300:, 1], c='red', marker='o')

    # tsne = TSNE(n_components=3, random_state=0)
    # X_tsne = tsne.fit_transform(X)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X_tsne[:300, 0], X_tsne[:300, 1], X_tsne[:300, 2], c='blue', marker='o')
    # ax.scatter(X_tsne[300:, 0], X_tsne[300:, 1], X_tsne[300:, 2], c='red', marker='o')

    plt.savefig(path)

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

    return parser

caer_list = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
sfew_list = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"]
heco_list = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Peace", "Excitement"]
emotic_list = ['Affection','Anger','Annoyance','Anticipation','Aversion','Confidence','Disapproval','Disconnection','Disquietment','Doubt/Confusion','Embarrassment','Engagement','Esteem','Excitement','Fatigue','Fear','Happiness','Pain','Peace','Pleasure','Sadness','Sensitivity','Suffering','Surprise','Sympathy','Yearning']

feature_maps = None
@torch.no_grad()
def infer(dataset, model, postprocessors, device, args):
    if args.dataset_file == 'heco': emo_list = heco_list
    if args.dataset_file == 'emotic': emo_list = emotic_list
    if args.dataset_file == 'sfew2': emo_list = sfew_list
    if args.dataset_file == 'caer': emo_list = caer_list
    model.eval()
    duration = 0
    gt_list = []; pred_list = []
    gt_list_b = []; pred_list_b = []
    face_point_tracks = None
    
    def hook_fn(module, input, output):
        global feature_maps
        feature_maps, _ = output
    target_layer = model.backbone
    hook = target_layer.register_forward_hook(hook_fn)

    for orig_image, target in dataset:
        if 0<args.face_num<5:
            if len(target) != args.face_num: continue
        elif args.face_num>=5:
            if len(target) < 5: continue
        else: pass

        img_name = os.path.join(args.data_path, dataset.coco.imgs[target[0]["image_id"]]['file_name'])
        # img_name = '/data1/xinpeng/EMOTIC/images/mscoco/images/COCO_train2014_000000094388.jpg'
        # img_name = '/data1/xinpeng/EMOTIC/images/emodb_small/images/9d05mol0ji5qlmawoc.jpg'
        # img_name = '/data1/xinpeng/EMOTIC/images/emodb_small/images/b98lhbcesn2m5in4wn.jpg'
        # img_name = '/data1/xinpeng/EMOTIC/images/emodb_small/images/dyxwbrrg5gvylxzhec.jpg'
        # img_name = '/data1/xinpeng/EMOTIC/images/ade20k/images/sun_aaflptztmjujrsgx.jpg'
        if img_name != '/data1/xinpeng/EMOTIC/images/emodb_small/images/b98lhbcesn2m5in4wn.jpg': continue

        orig_image = Image.open(img_name)

        plot_image = np.array(orig_image)
        plot_image = cv2.cvtColor(plot_image, cv2.COLOR_BGR2RGB)
        plot_image = cv2.resize(plot_image, (1000, 800))

        print("processing...{}".format(img_name))
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

        if args.binary_flag:
            probas = outputs['pred_logits'].sigmoid()[0].cpu() # (num_query, emo_class)
        else:
            probas = outputs['pred_logits'][0,:,:-1].softmax(-1).cpu() # (num_query, emo_class)

        outputs["pred_boxes"] = outputs["pred_boxes"].cpu()
        bboxes = rescale_bboxes(outputs['pred_boxes'][0].cpu(), orig_image.size) # (num_query, 4)
        bboxes_plot = rescale_bboxes(outputs['pred_boxes'][0].cpu(), (plot_image.shape[1], plot_image.shape[0])) # (num_query, 4)
        # dets = torch.cat([bboxes, torch.topk(probas, 8)[0].mean(-1).unsqueeze(-1)], dim=1)
        # dets = torch.cat([bboxes, probas.max(dim=-1)[0].unsqueeze(-1)], dim=1)
        dets = torch.cat([bboxes, probas.mean(dim=-1).unsqueeze(-1)], dim=1)

        keep = nms(dets.numpy(), 0.01)
        # keep = list(range(dets.shape[0]))
        # keep = torch.argsort(probas.mean(dim=-1), descending=True)
        # keep = [0,1,2,3]

        # probas = probas[keep]; bboxes = bboxes[keep]; bboxes_plot = bboxes_plot[keep]
        # prob_, cls_ = probas.max(dim=-1)
        # sorted_idx = torch.argsort(prob_, descending=True)
        # face_samples, face_points = get_face_points(model)
        # context_samples, context_points, relevance_values, context_points_sp, context_points_se = get_context_points(model)

        # keep = list(range(face_points.shape[0]))
        # # # plot_image = plot_points(plot_image, context_samples, context_points, (255, 255, 255)/keep_value)
        # for keep_idx, keep_value in enumerate(keep):
        #     if keep_idx==3:
        #         # plot_image = plot_points(plot_image, [face_samples[keep_value]], [face_points[keep_value]], color_list[keep_idx], context_samples=context_samples, context_points=context_points)
        #         # plot_image = plot_points(plot_image, [face_samples[keep_value]], [face_points[keep_value]], color_list[keep_idx], context_samples=context_samples, context_points=face_samples[keep_value])
        #         # plot_image = plot_points(plot_image, [face_samples[keep_value]], [face_points[keep_value]], color_list[keep_idx], context_samples=context_samples, context_points=context_points_sp[:100, keep_value])
        #         plot_image = plot_points(plot_image, [face_samples[keep_value]], [face_points[keep_value]], color_list[keep_idx], context_samples=context_samples, context_points=context_points_se[:50, keep_value])
        #     else:
        #         pass
        #         plot_image = plot_points(plot_image, [face_samples[keep_value]], [face_points[keep_value]], color_list[keep_idx], context_samples=context_samples, context_points=context_points_se[:50, keep_value])

        # for keep_idx, keep_value in enumerate(keep):
        #     plot_image = plot_points(plot_image, [face_samples[keep_value]], [face_points[keep_value]], color_list[keep_idx])
        #     break
            # plot_image = plot_points(plot_image, [face_samples[keep_value]], [face_points[keep_value]], color_list[keep_value])
            # plot_image = plot_result(plot_image, [bboxes_plot[keep_value].data.numpy()], color_list[keep_value]) # (B, G, R)
        
        # keep = keep[:len(target)] if len(keep)>len(target) else keep + [keep[0]] * (len(target) - len(keep))
        for ann_idx, ann in enumerate(target):
            img_bbox = ann["bbox"]
            img_bbox = [img_bbox[0], img_bbox[1], img_bbox[0]+img_bbox[2], img_bbox[1]+img_bbox[3]]
            img_bbox_plot = [img_bbox[0]*plot_image.shape[1]/orig_image.size[0], 
                             img_bbox[1]*plot_image.shape[0]/orig_image.size[1], 
                             img_bbox[2]*plot_image.shape[1]/orig_image.size[0], 
                             img_bbox[3]*plot_image.shape[0]/orig_image.size[1]]

            if args.binary_flag:
                img_class_b = bin(ann["category_id"])[2:]
                img_class_b = np.array([0,]*(len(emo_list) - len(img_class_b)) + [int(x) for x in img_class_b])
                img_class = np.where(img_class_b==1)[0]
            else:
                img_class = ann["category_id"]
                img_class_b = np.zeros(len(emo_list))
                img_class_b[img_class] = 1
                img_class = [img_class]

        #     gt_list.append(img_class[0])
        #     gt_list_b.append(np.array(img_class_b))
            plot_image = plot_result(plot_image, [img_bbox_plot], probas=[[1.0]*len(img_class)], classes=[img_class,], emo_list=emo_list, color=color_list[ann_idx]) # (B, G, R)

        #     idx_sele = face_matching(img_bbox, bboxes)
        #     prob_sele = probas[idx_sele].data.numpy()

        #     prob_related, cls_related = prob_sele.max(), prob_sele.argmax()
        #     pred_list.append(cls_related)
        #     prob_related, cls_related = [prob_related, ], [cls_related, ]
        #     pred_list_b.append(prob_sele)
        #     if args.binary_flag:
        #         # print(prob_sele)
        #         # cls_related = np.where(prob_sele[:-1]>0.2)[0]
        #         cls_related = np.argsort(prob_sele[:-1])[::-1][:6]
        #         prob_related = prob_sele[cls_related]

        #     plot_image = plot_points(plot_image, face_samples[idx_sele], face_points[idx_sele], context_samples_list[idx_sele], context_points[idx_sele], relevance_values[idx_sele])
            # plot_image = plot_points(plot_image, face_samples[idx_sele], face_points[idx_sele], color_list[ann_idx])
            # plot_image = plot_result(plot_image, [bboxes_plot[idx_sele].data.numpy()], probas=[prob_related], classes=[cls_related], emo_list=emo_list, color=color_list[ann_idx]) # (B, G, R)
            
            # if(cls_related != img_class): print(img_name.replace('Val', 'log').replace('Data', 'log'))
        # if face_point_tracks is None: face_point_tracks = get_trajectory(model, image, target, device, keep, args)
        # plot_image = plot_trajectory(plot_image, face_point_tracks)
        # plot_fea(image, feature_maps.copy(), img_save_path)

        path = 'log_deformable_detr_gt'
        img_save_path = img_name.replace('Val', path).replace('Data', path).replace('images', path).replace('test', path)
        if not os.path.exists(os.path.dirname(img_save_path)):
            os.makedirs(os.path.dirname(img_save_path))
        print(img_save_path)
        cv2.imwrite(img_save_path, plot_image)
        # get_tsne(model, img_save_path)
        embed()
        
        end_t = time.perf_counter()
        infer_time = end_t - start_t
        duration += infer_time
        print('infer_time: {}'.format(infer_time))
        
    hook.remove()
    avg_duration = duration / len(dataset.coco.imgs)
    print("Totally {} image; Avg. Time: {:.3f}s".format(len(dataset.coco.imgs), avg_duration))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    model, _, postprocessors = build_model(args)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        embed()
        model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device)

    dataset = torchvision.datasets.CocoDetection(args.data_path, args.json_path)

    infer(dataset, model, postprocessors, device, args)
