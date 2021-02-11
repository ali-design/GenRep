#!/bin/bash
cont=0
gpus=(0 1 2 3 4 5)
for i in 1000
do
    gpuid=${gpus[$cont]}
    echo "Run gpu "$gpuid" "$cont
    ./run_svm.sh $gpuid /data/scratch/xavierpuig/ganclr/SupCon/biggan_models/SupCon_biggan_independent2_2000_resnet50_ncontrast.20_lr_0.03_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_/ckpt_epoch_$i.pth  supcon_indep_total2000_epoch$i &
    cont=$(($cont + 1))
done
