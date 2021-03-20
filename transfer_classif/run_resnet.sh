#!/bin/bash
echo "Extracting..."
CUDA_VISIBLE_DEVICES=$1 python extract_features.py --dataset voc2007 --data_folder /data/vision/torralba/datasets/PASCAL2007/ --ckpt $2 --expname $3 --model_weight pretrained
NAMEEXP=$3
OUT_PATH="../../scratch/svm_models/voc2007/${NAMEEXP}"
OUT_FEATS="../../scratch/features/voc2007/data_${NAMEEXP}"
echo $OUT_FEATS
echo "Training..."

python train_svm_kfold.py --data_file $OUT_FEATS --output_path $OUT_PATH --costs_list "0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0"
echo "Testing..."
python test_svm.py --data_file $OUT_FEATS --output_path $OUT_PATH --costs_list "0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0"