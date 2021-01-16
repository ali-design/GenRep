#!/bin/bash
cont=0
gpus=(0 1 2 3 4 5)
for i in 20 40 60 80 100
do
    gpuid=${gpus[$cont]}
    echo "Run gpu "$gpuid" "$cont
    ./run_svm.sh $gpuid /data/scratch/xavierpuig/ganclr/SupCon/imagenet100_models/SupCon_imagenet100_real_100_resnet50_ncontrast.0_lr_0.03_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_/ckpt_epoch_$i.pth  supcon_real_total100_epoch$i &
    cont=$(($cont + 1))
done
