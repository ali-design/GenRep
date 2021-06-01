from os import listdir
from os.path import isfile, join, basename, isfile
import glob
import numpy as np

def make_html(home_dir):
#     try:
#         index_html_path = join(home_dir, "index.html")
#         img_dir = join(home_dir, 'images')
#         print('Saving index file in', index_html_path)
#         fid = open(index_html_path, 'w', encoding = 'utf-8')
#         fid.write('<table style="text-align:center;">')
#         im_name_list = glob.glob(join(img_dir,'image_epoch_*anchor*.png'))

#         for anchor_name in im_name_list[:1000]:                
#             neighbor_name = anchor_name.replace('anchor', 'neighbor')
#             if not isfile(neighbor_name):
#                 continue

#             class_name = anchor_name.split('anchor_')[-1]
#             class_name = class_name.split('.png')[0]
#             epoch = anchor_name.split('epoch_')[-1].split('_')[0]
#             epoch = epoch.split('_')[0]
#             row_str = '<tr><td>epoch{}:    </td>'.format(epoch)            
#             anchor_name = anchor_name.replace('/data/vision/phillipi/ganclr/datasets/', '/jahanian/gcd/')
#             neighbor_name = neighbor_name.replace('/data/vision/phillipi/ganclr/datasets/', '/jahanian/gcd/')
#             row_str += '<td>{}</td>'.format(class_name)
#             row_str += '<td><a href="{}"><img style="width:128px;height:128px;" src="{}"/></a></td>'.format(anchor_name, anchor_name)
#             row_str += '<td><a href="{}"><img style="width:128px;height:128px;" src="{}"/></a></tr></td></tr>'.format(neighbor_name, neighbor_name)
#             fid.write(row_str)  
#         fid.write('</table>')
#     finally:
#         fid.close()

    ## only epoch 11 and 17
    try:
        index_html_path = join(home_dir, "index.html")
        img_dir = join(home_dir, 'images')
        print('Saving index file in', index_html_path)
        fid = open(index_html_path, 'w', encoding = 'utf-8')
        fid.write('<table style="text-align:center;">')
        ## epoch 11
        im_name_list = glob.glob(join(img_dir,'image_epoch_11*anchor*.png'))
        print('size epoch 11:', len(im_name_list))
        for anchor_name in im_name_list:                
            neighbor_name = anchor_name.replace('anchor', 'neighbor')
            if not isfile(neighbor_name):
                continue

            class_name = anchor_name.split('anchor_')[-1]
            class_name = class_name.split('.png')[0]
            epoch = anchor_name.split('epoch_')[-1].split('_')[0]
            epoch = epoch.split('_')[0]
            row_str = '<tr><td>epoch{}:    </td>'.format(epoch)            
            anchor_name = anchor_name.replace('/data/vision/phillipi/ganclr/datasets/', '/jahanian/gcd/')
            neighbor_name = neighbor_name.replace('/data/vision/phillipi/ganclr/datasets/', '/jahanian/gcd/')
            row_str += '<td>{}</td>'.format(class_name)
            row_str += '<td><a href="{}"><img style="width:128px;height:128px;" src="{}"/></a></td>'.format(anchor_name, anchor_name)
            row_str += '<td><a href="{}"><img style="width:128px;height:128px;" src="{}"/></a></tr></td></tr>'.format(neighbor_name, neighbor_name)
            fid.write(row_str)  

        ## epoch 17
        im_name_list = glob.glob(join(img_dir,'image_epoch_17*anchor*.png'))
        print('size epoch 17:', len(im_name_list))
        for anchor_name in im_name_list:                
            neighbor_name = anchor_name.replace('anchor', 'neighbor')
            if not isfile(neighbor_name):
                continue

            class_name = anchor_name.split('anchor_')[-1]
            class_name = class_name.split('.png')[0]
            epoch = anchor_name.split('epoch_')[-1].split('_')[0]
            epoch = epoch.split('_')[0]
            row_str = '<tr><td>epoch{}:    </td>'.format(epoch)            
            anchor_name = anchor_name.replace('/data/vision/phillipi/ganclr/datasets/', '/jahanian/gcd/')
            neighbor_name = neighbor_name.replace('/data/vision/phillipi/ganclr/datasets/', '/jahanian/gcd/')
            row_str += '<td>{}</td>'.format(class_name)
            row_str += '<td><a href="{}"><img style="width:128px;height:128px;" src="{}"/></a></td>'.format(anchor_name, anchor_name)
            row_str += '<td><a href="{}"><img style="width:128px;height:128px;" src="{}"/></a></tr></td></tr>'.format(neighbor_name, neighbor_name)
            fid.write(row_str)  
            
        fid.write('</table>')
    finally:
        fid.close()
make_html('/data/vision/phillipi/ganclr/datasets/SupCon_gan_gauss1_resnet50_ncontrast.1_ratiodata.10.0_lr_0.03_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_biggan-deep-256_tr2.0_gauss1_std1.0_imagenet1000_NS13000_NN1')


