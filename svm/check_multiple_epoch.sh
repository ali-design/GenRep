#!/bin/sh
for i in 40 80 120 160 200
do
	./run_svm.sh 0 /data/scratch-oc40/jahanian/ganclr_results/SupCon/imagenet100_models/SupCon_imagenet100_resnet50_lr_0.03_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_ImageNet100/ckpt_epoch_$i.pth  supcon_imagenet100_epoch$i
done
