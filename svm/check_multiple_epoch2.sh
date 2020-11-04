#!/bin/bash
cont=0
gpus=(0 1 2 3 4 5)
for i in 40 80 120 160 200
do
    gpuid=${gpus[$cont]}
    echo "Run gpu "$gpuid
    ./run_svm.sh $gpuid  /data/scratch/xavierpuig/ganclr/SupCon/biggan_models/SupCon_biggan_independent_resnet50_ncontrast.1_lr_0.03_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_/ckpt_epoch_$i.pth  supcon_independent_contrast1_total200_epoch$i &
    cont=$(($cont + 1))
done
