
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_net.py \
#      --num-gpus 8 \
#      --config-file config/pascal_voc_R_50_C4_transfer.yaml \
#      MODEL.WEIGHTS ckpts/simclr_bigbi_gauss.pth \
#      OUTPUT_DIR outputs/simclr_bigbi_gauss
#    
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_net.py \
#      --num-gpus 8 \
#      --config-file config/pascal_voc_R_50_C4_transfer.yaml \
#      MODEL.WEIGHTS ckpts/supcon_biggan_gauss.pth \
#      OUTPUT_DIR outputs/supcon_biggan_gauss





#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_net.py \
#      --num-gpus 8 \
#      --config-file config/pascal_voc_R_50_C4_transfer.yaml \
#      MODEL.WEIGHTS ckpts/simclr_bigbi_indep.pth \
#      OUTPUT_DIR outputs/simclir_bigbi_indep
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_net.py \
#      --num-gpus 8 \
#      --config-file config/pascal_voc_R_50_C4_transfer.yaml \
#      MODEL.WEIGHTS ckpts/simclr_imagenet100.pth \
#      OUTPUT_DIR outputs/simclr_imagenet100
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_net.py \
#      --num-gpus 8 \
#      --config-file config/pascal_voc_R_50_C4_transfer.yaml \
#      MODEL.WEIGHTS ckpts/supcon_imagenet100.pth \
#      OUTPUT_DIR outputs/supcon_imagenet100
#
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_net.py \
#      --num-gpus 8 \
#      --config-file config/pascal_voc_R_50_C4_transfer.yaml \
#      MODEL.WEIGHTS ckpts/imagenet1000_ce.pth \ 
#      OUTPUT_DIR outputs/imagenet_1000_ce
#    
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_net.py \
#      --num-gpus 8 \
#      --config-file config/pascal_voc_R_50_C4_transfer.yaml \
#      MODEL.WEIGHTS ckpts/random_init.pth \
#      OUTPUT_DIR outputs/random_init

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_net.py \
      --num-gpus 8 \
      --config-file config/pascal_voc_R_50_C4_transfer.yaml \
      MODEL.WEIGHTS ckpts/supcon_biggan_indep.pth \
      OUTPUT_DIR outputs/supcon_biggan_indep
