
CUDA_VISIBLE_DEVICES=1,2 python main_supcon.py --cosine --batch_size 256 --method SupCon --ganrndwalk --zstd 1.2 --data_folder  /data/scratch/jahanian/ganclr_results/biggan256tr1-png_steer_rnd_100


CUDA_VISIBLE_DEVICES=1,2 python main_linear.py \
--ckpt /data/scratch-oc40/jahanian/ganclr_results/SupCon/biggan_models/SupCon_biggan_ganrndwalk_zstd_1.1_resnet50_lr_0.03_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine/last.pth \
--data_folder /data/scratch-oc40/jahanian/ganclr_results/ImageNet100/ --learning_rate 0.01 --batch_size 128

