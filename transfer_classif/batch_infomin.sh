#!/bin/bash

## simclr lr0.03
./run_svm2.sh 3 /data/vision/phillipi/ganclr/models/train1k/samples_size/InfoMin/SimCLR/gan_models/SimCLR_gan_gauss1_resnet50_ncontrast.1_ratiodata.1.0_lr_0.03_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_bigbigan_resnet_128_tr2.0_gauss1_std0.05_imagenet1000_NS1300_NN1/last.pth out_simclr_gauss1_lr0.03_r1_std0.05/epoch_20 

./run_svm2.sh 5 /data/vision/phillipi/ganclr/models/train1k/samples_size/InfoMin/SimCLR/gan_models/SimCLR_gan_gauss1_resnet50_ncontrast.1_ratiodata.1.0_lr_0.03_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_bigbigan_resnet_128_tr2.0_gauss1_std0.1_imagenet1000_NS1300_NN1/last.pth out_simclr_gauss1_lr0.03_r1_std0.1/epoch_20 

./run_svm2.sh 1 /data/vision/phillipi/ganclr/models/train1k/samples_size/train1k/SimCLR/gan_models/SimCLR_gan_gauss1_resnet50_ncontrast.1_ratiodata.1.0_lr_0.03_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_bigbigan_resnet_128_tr2.0_gauss1_std0.2_imagenet1000_NS13000_NN1/last.pth out_simclr_gauss1_lr0.03_r1_std0.2/epoch_20

./run_svm2.sh 1 /data/vision/phillipi/ganclr/models/train1k/samples_size/InfoMin/SimCLR/gan_models/SimCLR_gan_gauss1_resnet50_ncontrast.1_ratiodata.1.0_lr_0.03_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_bigbigan_resnet_128_tr2.0_gauss1_std0.3_imagenet1000_NS1300_NN1/last.pth out_simclr_gauss1_lr0.03_r1_std0.3/epoch_20

./run_svm2.sh 3 /data/vision/phillipi/ganclr/models/train1k/samples_size/InfoMin/SimCLR/gan_models/SimCLR_gan_gauss1_resnet50_ncontrast.1_ratiodata.1.0_lr_0.03_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_bigbigan_resnet_128_tr2.0_gauss1_std0.4_imagenet1000_NS1300_NN1/last.pth out_simclr_gauss1_lr0.03_r1_std0.4/epoch_20 

## supcon lr0.01
./run_svm2.sh 1 /data/vision/phillipi/ganclr/models/train1k/samples_size/InfoMin/SupCon/gan_models/SupCon_gan_gauss1_resnet50_ncontrast.1_ratiodata.1.0_lr_0.01_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_biggan-deep-256_tr2.0_gauss1_std0.5_imagenet1000_NS1300_NN1/last.pth out_supcon_gauss1_lr0.01_r1_std0.5/epoch_last 

./run_svm2.sh 3 /data/vision/phillipi/ganclr/models/train1k/samples_size/InfoMin/SupCon/gan_models/SupCon_gan_gauss1_resnet50_ncontrast.1_ratiodata.1.0_lr_0.01_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_biggan-deep-256_tr2.0_gauss1_std0.8_imagenet1000_NS1300_NN1/last.pth out_supcon_gauss1_lr0.01_r1_std0.8/epoch_last 

./run_svm2.sh 5 /data/vision/phillipi/ganclr/models/train1k/samples_size/InfoMin/SupCon/gan_models/SupCon_gan_gauss1_resnet50_ncontrast.1_ratiodata.1.0_lr_0.01_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_biggan-deep-256_tr2.0_gauss1_std1.0_imagenet1000_NS13000_NN1/last.pth out_supcon_gauss1_lr0.01_r1_std1.0/epoch_last 

./run_svm2.sh 1 /data/vision/phillipi/ganclr/models/train1k/samples_size/InfoMin/SupCon/gan_models/SupCon_gan_gauss1_resnet50_ncontrast.1_ratiodata.1.0_lr_0.01_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_biggan-deep-256_tr2.0_gauss1_std1.2_imagenet1000_NS1300_NN1/last.pth out_supcon_gauss1_lr0.01_r1_std1.2/epoch_last 

./run_svm2.sh 3 /data/vision/phillipi/ganclr/models/train1k/samples_size/InfoMin/SupCon/gan_models/SupCon_gan_gauss1_resnet50_ncontrast.1_ratiodata.1.0_lr_0.01_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_biggan-deep-256_tr2.0_gauss1_std1.5_imagenet1000_NS1300_NN1/last.pth out_supcon_gauss1_lr0.01_r1_std1.5/epoch_last 