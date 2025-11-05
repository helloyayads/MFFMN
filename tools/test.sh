## kitti

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=29987 test.py  --tcp_port 29987  --launcher pytorch  \
--cfg_file ${CONFIG_FILE} \
--ckpt_dir ${CKPT_DIR}
--batch_size 2 --workers 4 \