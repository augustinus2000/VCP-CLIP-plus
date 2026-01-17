#!/bin/bash

echo "===== VCP-CLIP+_train.py start ====="
python -u VCP-CLIP+_train.py --dataset visa --train_data_path ./dataset/mvisa/data \
--val_data_path ./dataset/mvisa/data \
--save_path ./my_exps/train_visa --pretrained_path ./pretrained_weight/ViT-L-14-336px.pt \
--prompt_len 2 --deep_prompt_len 1 --device_id 0 --learning_rate 0.00004 --features_list 6 12 18 24 --pretrained openai --image_size 518 \
--batch_size 16 --epoch 10 --group_id_list 2 --seed 333 --config_path ./models/model_configs/ViT-L-14-336.json --model ViT-L-14-336
echo "===== Training completed ====="