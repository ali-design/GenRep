# ganCLR readme

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
