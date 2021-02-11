#!/bin/bash
./run_svm2.sh 0 /data/scratch/xavierpuig/ganclr/SupInv/biggan_models/SupInv_biggan_resnet50_lr_0.03_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_nn_biggan256tr1-png_steer_rnd_std1.0_100_1_samples/last.pth  sup_inverter &

./run_svm2.sh 1 /data/scratch/xavierpuig/ganclr/UnsupInv/biggan_models/UnsupInv_biggan_resnet50_lr_0.03_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_new_bigbi_resnet_128_gauss1_std0.3_imagenet100_N1/last.pth  unsup_inverter &
