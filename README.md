ganCLR readme

To run SimCLR, training:
```
python main_supcon.py --cosine --syncBN
```
`--syncBN` is required for SimCLR, other parameters such as `batch_size` and `learning_rate` can be changed accordingly.
To do linear testing:
```
python main_linear.py --ckpt /path/to/model.pth
```
