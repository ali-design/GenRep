# Train 
CUDA_VISIBLE_DEVICES=3,4 python main_linear.py \
--ckpt CKPT_HERE  \
--data_folder /data/vision/torralba/datasets/imagenet_pytorch_new/ImageNet  --learning_rate 0.01 --batch_size 256

