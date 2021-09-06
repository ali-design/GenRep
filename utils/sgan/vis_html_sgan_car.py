from os import listdir
from os.path import isfile, join, basename
import glob
import numpy as np
import json
import random

def make_html(home_dir):
    try:
        ## load the imgList
        imgList_filename = join(home_dir, 'ratiodata0.1_imgList.txt')
#         imgList_filename = join(home_dir, 'full_imgList.txt')
        if isfile(imgList_filename):
            print('Listing images by reading from ', imgList_filename)
            with open(imgList_filename, 'r') as fid:
                imgList = fid.readlines()
            imgList = [x.rstrip() for x in imgList]
        else:
            print('Error: cannot find full_imgList.txt file')
        
#         ## dir_name to class_name
#         with open('./imagenet100_class_index.json', 'rb') as fid:
#             imagenet_class_name_dict = json.load(fid)

        index_html_path = join(home_dir, "index.html")
        print('Saving index file in', index_html_path)
        fid = open(index_html_path, 'w', encoding = 'utf-8')
        fid.write('<table style="text-align:center;">')
        
#         random.shuffle(imgList)

        for anchor_name in imgList:                
            neighbor_name = anchor_name.replace('anchor', '1.0_1')
            dir_name = anchor_name.split('/')[-2]
#             class_name = imagenet_class_name_dict[dir_name] 

            row_str = '<tr><td>(anchor, neighbor): </td>'       
            anchor_name = anchor_name.replace('/data/vision/phillipi/ganclr/datasets/', '/jahanian/gcd/')
            neighbor_name = neighbor_name.replace('/data/vision/phillipi/ganclr/datasets/', '/jahanian/gcd/')
#             row_str += '<td>{}</td>'.format(class_name)
            row_str += '<td><a href="{}"><img style="width:128px;" src="{}"/></a></td>'.format(anchor_name, anchor_name)
            row_str += '<td><a href="{}"><img style="width:128px;" src="{}"/></a></tr></td></tr>'.format(neighbor_name, neighbor_name)
            fid.write(row_str)  
        fid.write('</table>')
    finally:
        fid.close()

# make_html('/data/vision/phillipi/ganclr/datasets/biggan-deep-256_tr2.0_gauss1_std1.0_imagenet1000_NS13000_NN1')
make_html('/data/vision/phillipi/ganclr/datasets/stylegan2_cars893_tr0.9_gauss1_std0.25_NS1000_NN1')