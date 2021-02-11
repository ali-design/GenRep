#!/bin/bash
cont=0
gpus=(0 1 2 3 4 5)
for i in 480 # 80 120 160 200 240 280 320 360 400 440 480
do
    gpuid=${gpus[$cont]}
    echo "Run gpu "$gpuid" "$cont
    #./run_svm2.sh $gpuid /data/scratch/xavierpuig/ganclr/SupCon/biggan_models/SupCon_bigganonlineMPindep_none_resnet50_ncontrast.20_lr_0.03_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_ImageNet100/ckpt_epoch_$i.pth  supcon_onlineindep_total500/epoch$i &
    ./run_svm2.sh $gpuid /data/scratch/xavierpuig/ganclr/SupCon/biggan_models/SupCon_bigganonlineMP_none_resnet50_ncontrast.20_lr_0.03_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_ImageNet100/ckpt_epoch_$i.pth  supcon_onlinegauss_total500/epoch$i 

    #./run_svm2.sh $gpuid /data/scratch/xavierpuig/ganclr/SupCon/imagenet100_models/SupCon_imagenet100_real_500_resnet50_ncontrast.0_lr_0.03_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_/ckpt_epoch_$i.pth  supcon_real_total500/epoch$i 

    #./run_resnet.sh $gpuid /data/scratch/xavierpuig/ganclr/SupCon/imagenet100_models/SupCon_imagenet100_real_500_resnet50_ncontrast.0_lr_0.03_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_/ckpt_epoch_$i.pth  imagenet_resnet/epoch_unk4
    cont=$(($cont + 1))
done
