#!/bin/bash -l

#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --mail-user=mreza025@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=16
#SBATCH -p batch
#SBATCH --output=output_%j-%N.txt # logging per job and per host in the current directory. Both stdout and stderr are logged
#SBATCH --gres=gpu:1

conda activate mml

CUDA_VISIBLE_DEVICES=0 python patched_nyudv2_train_for_cl.py \
  --backbone resnet50 \
  --lr 0.05 \
  --workers 1 \
  --epochs 500 \
  --batch-size 64 \
  --ratio 3 \
  --gpu-ids 0 \
  --checkname MMSNetAttnCL \
  --model-name Patched-NYU-P64-B64-BlurredGrayscale+HHA-Std-Avg-R50-T-RGB+HHA \
  --eval-interval 1 \
  --loss-type ce \
  --dataset SmallDataset \
  --list-folder list_folder \
  --use-pretrained-resnet \
  --is-multimodal \
  --norm avg \
  --use-rgb \
  --use-depth
