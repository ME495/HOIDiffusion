#! /bin/bash

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
       --nproc_per_node 1 --master_port 47771 test_dex.py \
       --which_cond dex --bs 2 --cond_weight 1 --sd_ckpt model_sd_finetuned.ckpt \
       --cond_tau 1 --adapter_ckpt model_condition.pth --cond_inp_type image \
       --input /home/user/q_16T_2024/cj/GrabNet/output --file train.csv \
       --outdir results