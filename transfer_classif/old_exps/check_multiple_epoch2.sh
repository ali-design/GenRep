#!/bin/bash
cont=0
gpus=(0 1 2 3 4 5)
for i in 900
do
    gpuid=${gpus[$cont]}
    echo "Run gpu "$gpuid
    ./run_svm2.sh $gpuid  /data/scratch/xavierpuig/ganclr/SupCon/biggan_models/SupCon_biggan_gauss1.1_2000_resnet50_ncontrast.20_lr_0.03_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_big_deep256_tr1.0_gauss1_std1.0_imagenet100_N20/ckpt_epoch_$i.pth  :wsupcon_gangauss1.1_total2000_epoch$i &
    cont=$(($cont + 1))
done
