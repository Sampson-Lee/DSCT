# # emotic training scripts
# YOUR_DATA_PATH=./
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
#     --nproc_per_node=8 \
#     --master_port=29507 \
#     --use_env main.py \
#     --dataset_file=emotic \
#     --binary_flag=1 \
#     --detr=deformable_detr_dsct \
#     --model=deformable_transformer \
#     --batch_size=4 \
#     --cls_loss_coef=5 \
#     --data_path=$YOUR_DATA_PATH/EMOTIC/images/ \
#     --output_dir=$YOUR_DATA_PATH/EMOTIC/checkpoints \
#     --epochs=50 \
#     --lr_drop=40 \
#     --num_queries=4;

# # emotic testing scripts
# YOUR_DATA_PATH=./
# YOUR_MODEL_PATH=./
# CUDA_VISIBLE_DEVICES=1 python test.py \
#         --dataset_file=emotic \
#         --detr=deformable_detr_dsct \
#         --model=deformable_transformer \
#         --num_queries=4 \
#         --binary_flag=1 \
#         --data_path=$YOUR_DATA_PATH/EMOTIC/images/ \
#         --json_path=$YOUR_DATA_PATH/EMOTIC/annotations/test_bi.json \
#         --resume=$YOUR_MODEL_PATH/checkpoint.pth;

# # emotic visualization scripts
# YOUR_DATA_PATH=./
# YOUR_MODEL_PATH=./
# CUDA_VISIBLE_DEVICES=7 python vis.py \
#         --dataset_file=emotic \
#         --num_queries=4 \
#         --binary_flag=1 \
#         --model=deformable_transformer \
#         --detr=deformable_detr_dsct \
#         --data_path=$YOUR_DATA_PATH/EMOTIC/images/ \
#         --json_path=$YOUR_DATA_PATH/EMOTIC/annotations/test_bi.json \
#         --resume=$YOUR_MODEL_PATH/checkpoint.pth;

# caer-s training scripts
YOUR_DATA_PATH=/home/lxp/data
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=29507 \
        --use_env main.py \
        --dataset_file=caer \
        --binary_flag=0 \
        --detr=deformable_detr_dsct \
        --model=deformable_transformer_dsct \
        --batch_size=1 \
        --cls_loss_coef=5 \
        --data_path=$YOUR_DATA_PATH/CAER_S \
        --output_dir=$YOUR_DATA_PATH/CAER_S/checkpoints \
        --epochs=50 \
        --lr_drop=40 \
        --num_queries=4; \