# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Plotting utilities to visualize training logs.
"""
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np

from pathlib import Path, PurePath
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from IPython import embed
from itertools import product

import json
from IPython import embed
from collections import Counter
import torchvision

def plot_logs(logs, fields=('loss_bbox', 'loss_em'), ewm_col=0, log_name='log.txt'):
    '''
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    '''
    # func_name = "plot_utils.py::plot_logs"

    # # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # # convert single Path to list to avoid 'not iterable' error

    # if not isinstance(logs, list):
    #     if isinstance(logs, PurePath):
    #         logs = [logs]
    #         print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
    #     else:
    #         raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
    #         Expect list[Path] or single Path obj, received {type(logs)}")

    # # verify valid dir(s) and that every item in list is Path object
    # for i, dir in enumerate(logs):
    #     if not isinstance(dir, PurePath):
    #         raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
    #     if dir.exists():
    #         continue
    #     raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    plt.rcParams["font.family"] = "serif"
    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    # for df, color, log, coeff in zip(dfs, sns.color_palette(n_colors=len(logs)), logs, [2, 5, 10, 15]):
    #     for j, field in enumerate(fields):
    #         if j == 1:
    #             df[f'train_{field}'] = df[f'train_{field}']/coeff
    #             df[f'test_{field}'] = df[f'test_{field}']/coeff

    #         df.interpolate().ewm(com=ewm_col).mean().plot(
    #             y=[f'train_{field}', f'test_{field}'],
    #             ax=axs[j],
    #             color=[color] * 2,
    #             style=['-', '--']
    #         )

    for df, color, log, coeff in zip(dfs, sns.color_palette(n_colors=len(logs)), logs, [1, 5, 10, 15]):
        for j, field in enumerate(fields):
            if j == 0:
                df[f'train_{field}'] = df[f'train_{field}']/coeff
                df[f'test_{field}'] = df[f'test_{field}']/coeff

            df.interpolate().ewm(com=ewm_col).mean().plot(
                y=[f'train_{field}', f'test_{field}'],
                ax=axs[j],
                color=[color] * 2,
                style=['-', '--']
            )


    for ax, field in zip(axs, ['localization', 'classification']):
        # legends = list(product([Path(p).name for p in logs], ['train_', 'test_']))
        legends = list(product([r'$\lambda_{box}=1$', r'$\lambda_{box}=5$', r'$\lambda_{box}=10$', r'$\lambda_{box}=15$'], [' train', ' val']))
        legends = [l[0]+l[1] for l in legends]
        ax.legend(legends)
        ax.set_title(field)
        ax.set_xlabel("epochs")
        ax.set_ylabel("loss")

    plt.savefig("loss_plot.png", dpi=300, bbox_inches='tight')
    plt.savefig("loss_plot.pdf", bbox_inches='tight')

    plt.show()

def plot_precision_recall(files, naming_scheme='iter'):
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [f.stem for f in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        recall = data['params'].recThrs
        scores = data['scores']
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data['recall'][0, :, 0, -1].mean()
        print(f'{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, ' +
              f'score={scores.mean():0.3f}, ' +
              f'f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}'
              )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title('Precision / Recall')
    axs[0].legend(names)
    axs[1].set_title('Scores / Recall')
    axs[1].legend(names)
    return fig, axs


def plot_acc_speed():
    np.random.seed(42)
    # data = np.random.randint(0, 5, size=(3, 5))
    # print(data)
    data_emotic = np.array([[84, 84, 85, 108, 135, 139, 146, 173, 183, 189, 219],
                    [27.93, 23.85, 24.06, 29.63, 22.25, 31.53, 29.23, 36.18, 35.48, 28.42, 37.76]])
    name_emotic = ['EMOT-Net', 'CAER-Net', 'Emoticon (1)', 'Emoticon (1,3)', 'HECO (1)', 'Emoticon (1,2)', 'HECO (1,2)', 'HECO (1,2,3)', 'Emoticon (1,2,3)', 'GNN-CNN', 'HECO (1,2,3,4)']

    data_emotic_ours = np.array([[53, ], [37.80, ]])
    name_emotic_ours = ['Ours-R101', ]

    data_caer = np.array([[22, 23, 32, 45, 48, 52, 181, 223],
                    [73.51, 77.21, 74.51, 81.31, 84.82, 88.42, 88.65, 91.17]])
    name_caer = ['CAER-Net', 'GNN-CNN', 'EMOT-Net', 'GRERN', 'RRLA', 'MA-Net', 'EmotiCon (1,2,3)', 'CCIM+EmotiCon']

    data_caer_ours = np.array([[22, 39, ],
                    [86.03, 91.81, ]])
    name_caer_ours = ['Ours-R18', 'Ours-R50', ]

    sns.set(style="whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize=(8, 6))

    scatter1 = sns.scatterplot(x=data_emotic[0,:], y=data_emotic[1,:], color='red', marker='o', s=60, label='EMOTIC', legend=False)
    for idx in range(data_emotic.shape[1]):
        plt.text(data_emotic[0, idx]-15, data_emotic[1, idx]+0.2, name_emotic[idx], va="bottom", fontsize=9)

    scatter1_ours = sns.scatterplot(x=data_emotic_ours[0,:], y=data_emotic_ours[1,:], color='red', marker='*', s=100, label='EMOTIC', legend=False)
    for idx in range(data_emotic_ours.shape[1]):
        plt.text(data_emotic_ours[0, idx]-15, data_emotic_ours[1, idx]+0.2, name_emotic_ours[idx], va="bottom", fontsize=9)

    plt.xlim(0, 250)
    plt.ylim(20, 42)
    plt.xlabel("number of parameters (M)")
    plt.ylabel("mean Average Precision (%) on EMOTIC")

    ax2 = plt.twinx()
    scatter2 = sns.scatterplot(x=data_caer[0,:], y=data_caer[1,:], color='blue', marker='o', s=60, label='CAER-S', legend=False)
    for idx in range(data_caer.shape[1]):
        ax2.text(data_caer[0, idx]-15, data_caer[1, idx]+0.5, name_caer[idx], va="bottom", fontsize=9)

    scatter2_ours = sns.scatterplot(x=data_caer_ours[0,:], y=data_caer_ours[1,:], color='blue', marker='*', s=100, label='CAER-S', legend=False)
    for idx in range(data_caer_ours.shape[1]):
        ax2.text(data_caer_ours[0, idx]-15, data_caer_ours[1, idx]+0.5, name_caer_ours[idx], va="bottom", fontsize=9)

    ax2.set_ylabel("accuracy (%) on CAER-S")
    plt.ylim(70, 95)

    handles, labels = scatter1.get_legend_handles_labels()
    # handles_ours, labels_ours = scatter1_ours.get_legend_handles_labels()
    handles2, labels2 = scatter2.get_legend_handles_labels()
    # handles2_ours, labels2_ours = scatter2_ours.get_legend_handles_labels()
    # handles.extend(handles_ours)
    handles.extend(handles2)
    # handles.extend(handles2_ours)
    # labels.extend(labels_ours)
    labels.extend(labels2)
    # labels.extend(labels2_ours)
    # embed()
    plt.legend(handles, labels, loc='lower right')
    plt.grid()

    plt.savefig("performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig("performance_comparison.pdf", bbox_inches='tight')

    plt.show()

def plot_setnum():
    sns.set(style="whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize=(2, 1))

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    caer = [91.62, 91.57, 91.39, 91.57, 91.47, 91.52, 91.59, 91.42, 91.80, 91.44]
    emotic = [34.01, 35.78, 35.12, 35.50, 34.81, 35.53, 35.05, 35.50, 34.93, 35.02]
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Set Number')
    ax1.set_ylabel('Acc (%) on CAER-S', color=color)
    ax1.plot(x, caer, color=color, label='CAER-S')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(88,93)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('mAP (%) on EMOTIC', color=color)
    ax2.plot(x, emotic, color=color, linestyle='--', label='EMOTIC')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(30,40)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines = lines1 + lines2
    labels = labels1 + labels2
    ax1.legend(lines, labels, loc=0)
    plt.grid()

    # plt.title('Dual Axis Plot with Connected Lines')
    fig.tight_layout()
    plt.savefig("set_number.png", dpi=300, bbox_inches='tight')
    plt.savefig("set_number.pdf", bbox_inches='tight')

    plt.show()

def plot_Knum():
    sns.set(style="whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize=(8, 16))
    fig, ax = plt.subplots(figsize=(8, 4))

    x = [25, 50, 100, 150, 200, 250, 300]
    K_sp = [37.01, 37.25, 37.26, 37.01, 36.94, 36.91, 36.85]
    K_se = [35.81, 36.31, 35.89, 36.21, 36.01, 35.64, 35.92]

    plt.ylabel('mAP (%) on EMOTIC')

    plt.plot(x, K_sp, 'ro-', label='K_sp')
    plt.plot(x, K_se, 'bo-', label='K_se')
    for i in range(len(x)):
        plt.text(x[i], K_sp[i]+0.1, f'{K_sp[i]}', fontsize=10, ha='center', va='bottom')
    for i in range(len(x)):
        plt.text(x[i], K_se[i]+0.1, f'{K_se[i]}', fontsize=10, ha='center', va='bottom')

    plt.tick_params(axis='y')
    plt.ylim(35,38)

    plt.legend()
    plt.grid(True)
    fig.tight_layout()
    
    plt.savefig("K_number.png", dpi=300, bbox_inches='tight')
    plt.savefig("K_number.pdf", bbox_inches='tight')
    plt.show()


def multi_faces():
    data_path = '/data/xinpeng/EMOTIC/images/'
    output_file = '/data/xinpeng/EMOTIC/annotations/test_bi.json'
    dataset = torchvision.datasets.CocoDetection(data_path, output_file)

    fn_list = []
    for img, target in dataset:
        fn_list.append(len(target))
    result = Counter(fn_list)
    print(result)

if __name__ == "__main__":
    # plot_Knum()
    # multi_faces()
    plot_logs(['box1', 'box5', 'box10', 'box15',])
