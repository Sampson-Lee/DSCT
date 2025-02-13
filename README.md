# DSCT
By Xinpeng Li, Teng Wang, Jian Zhao, Shuyi Mao, Jinbao Wang, Feng Zheng, Xiaojiang Peng†, Xuelong Li†

This repository contains the official implementation of the paper [Two in One Go: Single-stage Emotion Recognition with Decoupled Subject-context Transformer (ACM MM 2024)](https://arxiv.org/abs/2404.17205).


## Introduction
Decoupled  Subject-Context Transformer (DSCT) is a single-stage emotion recognition approach for simultaneous subject localization and
emotion classification. 

![introduction](./imgs/intro.jpg)


## License

This project is released under the [Apache 2.0 license](./LICENSE).


## Installation

### Acknowledgement
The project is built upon [deformable_detr](https://github.com/fundamentalvision/Deformable-DETR).

### Requirements

* Linux
* CUDA >= 9.2
* GCC: 5.4 <= GCC <= 9.5
* Python >= 3.7
* PyTorch >= 1.5.1
* torchvision >= 0.6.1


### Installation
Clone the repo:
```bash
git clone git@github.com:Sampson-Lee/DSCT.git
cd DSCT
```

Create and activate the environment:
```bash
conda create -n dsct python=3.10 pip
conda activate dsct 
```

Install PyTorch and dependencies:
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

Compile CUDA operators:
```bash
cd ./models/ops
sh ./make.sh
```

Test the operators (ensure all checks pass until CUDA runs out of memory):
```bash
python test.py
```    

## Usage

### Dataset
Please download [EMOTIC dataset](https://github.com/rkosti/emotic) and [CAER-S dataset](https://caer-dataset.github.io/). 

We provided the coco-format annotations, i.e. `emotic_{train, val, test}_bi.json` and `caer_{train, val, test}.json`, and preprocessing scripts in `./datasets`.

Modify the dataset code or directory. Here is the directory tree I am following:
```
emotic
└── images
    ├── ade20k
    │   └── images
    │        ├── xxx.jpg
    ├── framesdb
    │   └── images
    │        ├── xxx.jpg
    ├── mscoco
    │   └── images
    │        ├── xxx.jpg
    └── emodb_small
        └── images
             ├── xxx.jpg
caer
├── train
│   ├── Anger
│   │   ├── xxx.jpg
│   ├── Disgust
│   ├── Fear
│   ├── Happy
│   ├── Neutral
│   ├── Sad
│   └── Surprise
└── test
    ├── Anger
    ├── Disgust
    ├── Fear
    ├── Happy
    ├── Neutral
    ├── Sad
    └── Surprise                        
```

### Running
Please place the [pretrained weights of deformable detr](https://drive.google.com/file/d/1nDWZWHuRwtwGden77NLM9JoWe-YisJnA/view?usp=sharing) in the working directory.

Then, follow `run.sh` to conduct training, testing, or visualization. 

Below is an explained example command:
```
YOUR_DATA_PATH=/home/lxp/data/emotic
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=29507 \
    --use_env main.py \
    --dataset_file=emotic \   # Choose the dataset between emotic and caer
    --binary_flag=1 \         # Set 1 for multi-label tasks; 0 for multi-class tasks
    --backbone=resnet50 \     # Select resnet50 or resnet101
    --detr=deformable_detr_dsct \
    --model=deformable_transformer_dsct \
    --batch_size=1 \          # Adjust the batch size
    --cls_loss_coef=5 \       # Determine the coefficient of the classification loss
    --data_path=$YOUR_DATA_PATH \
    --output_dir=$YOUR_DATA_PATH/checkpoints \
    --epochs=50 \
    --lr_drop=40 \
    --num_queries=4 \         # Configure the number of queries
    --pretrained_weights=./r50_deformable_detr-checkpoint.pth \ # leverage the pretrained weights of deformable detr
```

We conducted the experiments on a single NVIDIA GeForce RTX 4090 GPU, using the configuration specified in the provided `run.sh` file, which includes 50 epochs and a batch size of 4. The results remain competitive, achieving 89.19% on the CAER-S dataset and 36.21% on the Emotic dataset. The pretrained weights are available for download [here](https://drive.google.com/drive/folders/12PqXjm7XMuVq-rNjyXM9mbSWVHiOGJtL?usp=drive_link).

## Citation
If you find this project helpful, please consider citing our paper and starring our repository.

```
@inproceedings{li2024two,
  title={Two in One Go: Single-stage Emotion Recognition with Decoupled Subject-context Transformer},
  author={Li, Xinpeng and Wang, Teng and Zhao, Jian and Mao, Shuyi and Wang, Jinbao and Zheng, Feng and Peng, Xiaojiang and Li, Xuelong},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={9340--9349},
  year={2024}
}
```
