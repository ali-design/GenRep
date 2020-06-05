import os
import json

data_dir = './utils/train'
dirlist = os.listdir(data_dir)
with open('./utils/imagenet_class_index.json', 'rb') as fid:
    jdict = json.load(fid)
jkeys = jdict.keys()
for key in jkeys:
    dir_name = str('%04d' % int(key))
    dir_name_idx = jdict[key][0]
    os.rename(os.path.join(data_dir, dir_name), os.path.join(data_dir, dir_name_idx))