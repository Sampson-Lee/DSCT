# ##### emotic training scripts #####
# YOUR_DATA_PATH=/home/lxp/data/emotic
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
#     --nproc_per_node=1 \
#     --master_port=29507 \
#     --use_env main.py \
#     --dataset_file=emotic \ 
#     --backbone=resnet50 \ 
#     --binary_flag=1 \     
#     --detr=deformable_detr_dsct \
#     --model=deformable_transformer_dsct \
#     --batch_size=4 \
#     --cls_loss_coef=5 \
#     --data_path=$YOUR_DATA_PATH \
#     --output_dir=$YOUR_DATA_PATH/checkpoints \
#     --epochs=50 \
#     --lr_drop=40 \
#     --num_queries=4;

# ##### emotic testing scripts #####
# YOUR_DATA_PATH=/home/lxp/data/emotic
# YOUR_MODEL_PATH=/home/lxp/data/emotic/checkpoints
# CUDA_VISIBLE_DEVICES=0 python test.py \
#         --dataset_file=emotic \
#         --detr=deformable_detr_dsct \
#         --model=deformable_transformer_dsct \
#         --num_queries=4 \
#         --binary_flag=1 \
#         --data_path=$YOUR_DATA_PATH/images \
#         --json_path=./datasets/annotations/emotic_test_bi.json \
#         --resume=$YOUR_MODEL_PATH/checkpoint.pth;

# ##### emotic visualization scripts #####
# YOUR_DATA_PATH=/home/lxp/data/emotic
# YOUR_MODEL_PATH=/home/lxp/data/emotic/checkpoints
# CUDA_VISIBLE_DEVICES=0 python vis.py \
#         --dataset_file=emotic \
#         --num_queries=4 \
#         --binary_flag=1 \
#         --model=deformable_transformer_dsct \
#         --detr=deformable_detr_dsct \
#         --data_path=$YOUR_DATA_PATH/images \
#         --json_path=./datasets/annotations/emotic_test_bi.json \
#         --resume=$YOUR_MODEL_PATH/checkpoint.pth;

##### caer-s training scripts #####
YOUR_DATA_PATH=/home/lxp/data/CAER_S
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29507 \
        --use_env main.py \
        --dataset_file=caer \
        --binary_flag=0 \
        --detr=deformable_detr_dsct \
        --model=deformable_transformer_dsct \
        --batch_size=4 \
        --cls_loss_coef=5 \
        --data_path=$YOUR_DATA_PATH \
        --output_dir=$YOUR_DATA_PATH/checkpoints \
        --epochs=50 \
        --lr_drop=40 \
        --num_queries=9; \

# ##### caer-s testing scripts #####
# YOUR_DATA_PATH=/home/lxp/data/CAER_S
# YOUR_MODEL_PATH=/home/lxp/data/CAER_S/checkpoints
# CUDA_VISIBLE_DEVICES=0 python test.py \
#         --dataset_file=caer \
#         --detr=deformable_detr_dsct \
#         --model=deformable_transformer_dsct \
#         --num_queries=9 \
#         --binary_flag=0 \
#         --data_path=$YOUR_DATA_PATH/test \
#         --json_path=./datasets/annotations/caer_test.json \
#         --resume=$YOUR_MODEL_PATH/checkpoint.pth;

# ##### caer-s visualization scripts #####
# YOUR_DATA_PATH=/home/lxp/data/CAER_S
# YOUR_MODEL_PATH=/home/lxp/data/CAER_S/checkpoints
# CUDA_VISIBLE_DEVICES=0 python vis.py \
#         --dataset_file=caer \
#         --binary_flag=0 \
#         --num_queries=9 \
#         --model=deformable_transformer_dsct \
#         --detr=deformable_detr_dsct \
#         --data_path=$YOUR_DATA_PATH/test \
#         --json_path=./datasets/annotations/caer_test.json \
#         --resume=$YOUR_MODEL_PATH/checkpoint.pth;