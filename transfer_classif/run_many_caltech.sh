initpath="/data/scratch/xavierpuig/ganclr/SupCon/biggan_models/"
initpath2="/data/scratch/xavierpuig/ganclr/SupCon/imagenet100_models/"
./run_class_caltech.sh 0 $initpath"SupCon_biggan_gauss1.1_resnet50_ncontrast.0_lr_0.03_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_/last.pth" caltech_ce_0_v4 caltech101
#./run_class_caltech.sh 1 /data/scratch/xavierpuig/ganclr/SupCE/biggan_models/SupCE_biggan_gauss1.1_resnet50_ncontrast.0_lr_0.1_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_big_deep256_tr1.0_gauss1_std1.0_imagenet100_N20/last.pth caltech_supcon_0
