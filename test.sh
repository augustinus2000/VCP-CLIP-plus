# test on the MVTec-AD dataset

echo "===== VCP-CLIP+_test.py start ====="
python -u VCP-CLIP+_test.py --dataset mvtec --data_path ./dataset/mvisa/data \
--checkpoint_path ./my_exps/train_visa/epoch_10_group_id_2.pth \
--save_path ./results/test_mvtec --pretrained_path ./pretrained_weight/ViT-L-14-336px.pt \
--prompt_len 2 --deep_prompt_len 1 --device_id 1 --features_list 6 12 18 24 --pretrained openai --image_size 518 \
--seed 333 --config_path ./models/model_configs/ViT-L-14-336.json --model ViT-L-14-336
echo "===== Testing completed ====="