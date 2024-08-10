#!/bin/bash

#SBATCH --account=zhangwen
#SBATCH --partition=gpu

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5

module load nvidia/cuda/11.6

python test.py --name wpgan --batch_size 4 --netG wpgan --ngf 64 --embed_dim 64 --gpu_ids 0 --num_threads 5 --max_dataset_size 2000 --input_type test --phase test --epoch 30 --n_input_samples 3 --checkpoints_dir /project/songyanjiao/reconstruction/results --dataroot /project/songyanjiao/reconstruction/DW_S1_dataset_nearest_input --vgg16_path /project/songyanjiao/reconstruction/pretrained/vgg16_13C.pth

