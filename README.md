# DSCT
By Xinpeng Li, Teng Wang, Jian Zhao, Shuyi Mao, Jinbao Wang, Feng Zheng, Xiaojiang Peng†, Xuelong Li†

This repository is an official implementation of the paper [Two in One Go: Single-stage Emotion Recognition with Decoupled Subject-context Transformer (ACM MM 2024)](https://arxiv.org/abs/2404.17205).


## Introduction
Decoupled  Subject-Context Transformer (DSCT) is a single-stage emotion recognition approach for simultaneous subject localization and
emotion classification. 

![introduction](./imgs/intro.jpg)

<!-- Abstract. Emotion recognition aims to discern the emotional state of subjects within an image, relying on subject-centric and contextual visual cues. Current approaches typically follow a two-stage pipeline: first
localize subjects by off-the-shelf detectors, then perform emotion classification through the late fusion of subject and context features. However, the complicated paradigm suffers from disjoint training stages and limited fine-grained interaction between subject-context
elements. To address the challenge, we present a single-stage emotion recognition approach, employing a Decoupled Subject-Context Transformer (DSCT), for simultaneous subject localization and
emotion classification. Rather than compartmentalizing training stages, we jointly leverage box and emotion signals as supervision to enrich subject-centric feature learning. Furthermore, we introduce DSCT to facilitate interactions between fine-grained subjectcontext
cues in a “decouple-then-fuse” manner. The decoupled
query tokens—subject queries and context queries—gradually intertwine across layers within DSCT, during which spatial and semantic relations are exploited and aggregated. We evaluate our single-stage framework on two widely used context-aware emotion recognition datasets, CAER-S and EMOTIC. Our approach surpasses two-stage alternatives with fewer parameter numbers, achieving a 3.39% accuracy
improvement and a 6.46% average precision gain on CAER-S
and EMOTIC datasets, respectively. -->

## License

This project is released under the [Apache 2.0 license](./LICENSE).


## Installation
The project is based on [deformable_detr](https://github.com/fundamentalvision/Deformable-DETR).

### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n deformable_detr python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate deformable_detr
    ```
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

### Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```


## Usage

### Dataset
Please download [EMOTIC dataset](https://github.com/rkosti/emotic) and [CAER-S dataset](https://caer-dataset.github.io/). 

We provided the coco-format annotations, i.e. `emotic_{train, val, test}_bi.json` and `caer_{train, test}.json`, and preprocessing scripts.

### Running
Please follow `run.sh` to conduct training, testing, or visualization. 

We provided the latest pre-trained weights [here]() for your convenience.


## Citation
```
@inproceedings{li2024two,
  title={Two in One Go: Single-stage Emotion Recognition with Decoupled Subject-context Transformer},
  author={Li, Xinpeng and Wang, Teng and Zhao, Jian and Mao, Shuyi and Wang, Jinbao and Zheng, Feng and Peng, Xiaojiang and Li, Xuelong},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={9340--9349},
  year={2024}
}
```