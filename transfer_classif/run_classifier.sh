#!/bin/bash
echo "Extracting..."
# $4 is the dataset
CUDA_VISIBLE_DEVICES=$1 python extract_features.py --dataset $4 --data_folder ./datasets/ --ckpt $2 --expname $3
NAMEEXP=$3
OUT_PATH="../../scratch/logre_models/$4/${NAMEEXP}"
OUT_FEATS="../../scratch/features/$4/data_${NAMEEXP}"
echo $OUT_FEATS
echo "Training..."
python train_logistic_kfold.py --data_file $OUT_FEATS --output_path $OUT_PATH --costs_list "0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0"
echo "Testing..."
python test_logistic.py --data_file $OUT_FEATS --output_path $OUT_PATH --costs_list "0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0"
