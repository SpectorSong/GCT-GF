#!/bin/bash

#SBATCH --account=zhangwen
#SBATCH --partition=gpu

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5

module load nvidia/cuda/11.6

python train.py --use_amp --name wpgan --batch_size 4 --netG wpgan --ngf 64 --embed_dim 64 --netD sn --use_perceptual_loss --lambda_GAN 1 --lambda_L1 100 --lambda_percep 1 --gpu_ids 0 --max_dataset_size 4000 --num_threads 5 --input_type train --niter 3 --niter_decay 27 --lr 0.0001 --lr_policy linear --print_freq 800 --save_latest_freq 800 --save_epoch_freq 5 --display_freq 800 --checkpoints_dir /project/songyanjiao/reconstruction/results --dataroot /project/songyanjiao/reconstruction/DW_S1_dataset_nearest_input --vgg16_path /project/songyanjiao/reconstruction/pretrained/vgg16_13C.pth

