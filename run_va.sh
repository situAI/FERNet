CUDA_VISIBLE_DEVICES=6 nohup python -m torch.distributed.launch --master_port 9998 --nproc_per_node=1 main.py --config ./config/va_tdnn.yaml >> va_history.out &
