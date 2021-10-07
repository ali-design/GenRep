
#CUDA_VISIBLE_DEVICES=1,2 python main_supcon.py --cosine --batch_size 256 --method SupCon --ganrndwalk --zstd 1.2 --data_folder  /data/scratch/jahanian/ganclr_results/biggan256tr1-png_steer_rnd_100


#CUDA_VISIBLE_DEVICES=3,4 python main_linear.py \
#--ckpt /data/vision/torralba/frames/data_acquisition/GANContrastive/release/GenRep/your_ckpts_path_200_2/SupCE/gan_models/SupCE_gan_biggan_classif_resnet50_ncontrast.1_ratiodata.1.0_lr_0.03_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_biggan-deep-256_tr2.0_gauss1_std1.0_imagenet1000_NS13000_NN1/ckpt_epoch_200.pth  \
#--data_folder /data/vision/torralba/datasets/imagenet_pytorch_new/ImageNet  --learning_rate 0.01 --batch_size 256


CUDA_VISIBLE_DEVICES=3,4 python main_linear.py \
--ckpt /data/vision/torralba/frames/data_acquisition/GANContrastive/release/GenRep/biggan_ce_1N/SupCE/gan_models/SupCE_gan_biggan_classif_resnet50_ncontrast.1_ratiodata.1.0_lr_0.03_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_biggan-deep-256_tr2.0_gauss1_std1.0_imagenet1000_NS13000_NN1/ckpt_epoch_200.pth  \
--data_folder /data/vision/torralba/datasets/imagenet_pytorch_new/ImageNet  --learning_rate 0.01 --batch_size 256


#CUDA_VISIBLE_DEVICES=1,2 python main_linear.py \
#--ckpt /data/scratch-oc40/jahanian/ganclr_results/SupCon/biggan_models/SupCon_biggan_ganrndwalk_zstd_1.1_resnet50_lr_0.03_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine/last.pth \
#--data_folder /data/vision/torralba/datasets/pascal07/VOCdevkit/VOC2007/  --learning_rate 0.01 --batch_size 128 --dataset voc2007 

