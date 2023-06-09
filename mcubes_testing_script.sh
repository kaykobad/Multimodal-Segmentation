CUDA_VISIBLE_DEVICES=0,1 python my_testing_script_2.py \
  --backbone resnet \
  --lr 0.05 \
  --workers 1 \
  --epochs 1000 \
  --batch-size 8 \
  --pth-path ./run/multimodal_dataset/MMSNet/experiment_1/MMSNet-M3S-B8-RGB+NIR+Pol-Avg-2_best_test.pth.tar \
  --ratio 3 \
  --gpu-ids 0,1 \
  --eval-interval 1 \
  --loss-type ce \
  --dataset multimodal_dataset \
  --list-folder list_folder \
  --use-pretrained-resnet \
  --is-multimodal \
  --norm avg \
  --use-pol \
  --use-nir
