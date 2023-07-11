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

CUDA_VISIBLE_DEVICES=0 python patched_nyudv2_visualize.py \
  --backbone resnet50 \
  --lr 0.05 \
  --workers 2 \
  --epochs 1 \
  --batch-size 1 \
  --gpu-ids 0 \
  --pth-path ./run/multimodal_dataset/MCubeSNet/experiment_10/checkpoint-latest-pytorch.pth.tar \
  --eval-interval 1 \
  --ratio 3 \
  --loss-type ce \
  --dataset SmallDataset \
  --list-folder list_folder \
  --use-pretrained-resnet \
  --is-multimodal \
  --use-rgb \
  --norm avg 
