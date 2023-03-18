CUDA_VISIBLE_DEVICES=5,6 nohup python -m torch.distributed.launch --master_port 9998 --nproc_per_node=2 main.py --config ./config/au_baseline.yaml >> au_history.out &
