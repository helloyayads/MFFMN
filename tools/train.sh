#! /bin/bash

## kitti 
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=29987 train.py  --tcp_port 29987  --launcher pytorch  \
--cfg_file ${CONFIG_FILE} \
--batch_size 2  --epochs 80  --workers 4 --sync_bn