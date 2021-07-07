# GenRep readme

Before running, setting the path via lines [here](https://github.com/ali-design/ganCLR/blob/master/main_supcon.py#L79-L81)

To run SimCLR, training:
```
python main_supcon.py --cosine --syncBN
```
`--syncBN` is required for SimCLR, other parameters such as `batch_size` and `learning_rate` can be changed accordingly.
To do linear testing:
```
python main_linear.py --ckpt /path/to/model.pth
```
## SupCon biggan ganrndwalk train
CUDA_VISIBLE_DEVICES=1,2 python main_supcon.py --cosine --batch_size 256 --method SupCon --ganrndwalk --zstd 1.2 --data_folder /data/scratch/jahanian/ganclr_results_2/biggan256tr1-png_steer_rnd_100
## SupCon biggan ganrndwalk test
CUDA_VISIBLE_DEVICES=1,2 python main_linear.py --ckpt /data/scratch-oc40/jahanian/ganclr_results/SupCon/biggan_models/SupCon_biggan_ganrndwalk_zstd_1.1_resnet50_lr_0.03_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine/last.pth --data_folder /data/scratch-oc40/jahanian/ganclr_results/ImageNet100/ --learning_rate 0.01 --batch_size 128
