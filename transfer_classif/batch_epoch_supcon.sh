#!/bin/bash
cont=0
gpus=(1 3 5)
# for i in 1 10 20 30 40
# for i in 1 11 21 31
for i in 1 10 20
do
    gpuid=${gpus[$cont]}
    echo "Run gpu "$gpuid" "$cont
    ./run_svm2.sh $gpuid /data/vision/phillipi/ganclr/models/train1k/samples_size/InfoMin/SupCon/gan_models/SupCon_gan_gauss1_resnet50_ncontrast.1_ratiodata.1.0_lr_0.01_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_biggan-deep-256_tr2.0_gauss1_std0.5_imagenet1000_NS1300_NN1/ckpt_epoch_$i.pth out_supcon_gauss1_lr0.01_r1_std0.5/epoch$i 

#     ./run_svm2.sh $gpuid /data/scratch/jahanian/train1k_debug40/samples_size/SupCon/gan_models/SupCon_gan_gauss1_resnet50_ncontrast.1_ratiodata.10.0_lr_0.01_decay_0.0001_bsz_256_temp_0.1_trial_47_cosine_biggan-deep-256_tr2.0_gauss1_std1.0_imagenet1000_NS13000_NN1/ckpt_epoch_$i.pth out_supcon_gauss_loss_avg_e40_scratch_w64_lr0.01_r10/epoch$i 

#     ./run_svm2.sh $gpuid /data/scratch/jahanian/train1k_debug40/samples_size/SupCon/gan_models/SupCon_gan_gauss1_resnet50_ncontrast.1_ratiodata.1.0_lr_0.03_decay_0.0001_bsz_256_temp_0.1_trial_44_cosine_biggan-deep-256_tr2.0_gauss1_std1.0_imagenet1000_NS13000_NN1/ckpt_epoch_$i.pth out_supcon_gauss_loss_avg_e40_scratch_w64_lr0.0002_adam_r1/epoch$i 
    
    cont=$(($cont + 1))
done