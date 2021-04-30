import glob
import os
from tqdm import tqdm 
import time

## BigBiGAN
dataset = 'bigbigan_resnet_128_tr2.0_gauss1_std0.05_imagenet1000_NS1300_NN1'
# dataset = 'bigbigan_resnet_128_tr2.0_gauss1_std0.1_imagenet1000_NS1300_NN1'
# dataset = 'bigbigan_resnet_128_tr2.0_gauss1_std0.2_imagenet1000_NS13000_NN1'
# dataset = 'bigbigan_resnet_128_tr2.0_gauss1_std0.3_imagenet1000_NS1300_NN1'
# dataset = 'bigbigan_resnet_128_tr2.0_gauss1_std0.4_imagenet1000_NS1300_NN1'

# dataset = 'steer!'
##----------------------------------------------------------------------------
## BigGAN
# dataset = 'biggan-deep-256_tr1.0_steer_composite_pth_imagenet1000_N2'

# dataset = 'biggan-deep-256_tr1.0_indept_imagenet1000_N1'

# dataset = 'biggan-deep-256_tr2.0_gauss1_std1.0_imagenet1000_NS13000_NN1'
# dataset = 'biggan-deep-256_tr2.0_gauss1_std0.5_imagenet1000_NS1300_NN1'
# dataset = 'biggan-deep-256_tr2.0_gauss1_std0.8_imagenet1000_NS1300_NN1'
# dataset = 'biggan-deep-256_tr2.0_gauss1_std1.2_imagenet1000_NS1300_NN1'
# dataset = 'biggan-deep-256_tr2.0_gauss1_std1.5_imagenet1000_NS1300_NN1'
    
root_dir = os.path.join('/data/vision/phillipi/ganclr/datasets', dataset)
print('Working on dataset ', root_dir)
print('Listing images ...')
time_start = time.time()
imgList = glob.glob(os.path.join(root_dir, '*/*_anchor.png'))
print('number of images: {} and spent time: {}'.format(len(imgList), time.time() - time_start))

print('creating the full image list ...')
with open(os.path.join(root_dir,'full_imgList.txt'), 'w') as fid:
    for anchor_name in tqdm(imgList):
        fid.write(anchor_name+'\n')

#trim
ratiodata_list = [10.0, 5.0, 2.0, 1.0, 0.5, 0.1]
for ratiodata in ratiodata_list:
    max_per_class = int(1300 * ratiodata)
    if len(imgList) < max_per_class * 1000:
        continue
        
    indices = [int(x.split('sample')[-1].split('_')[0]) for x in imgList]
    imgList = [imname for imname, ind in zip(imgList, indices) if ind < max_per_class]
    print('creating {} ratio of the image list ...'.format(ratiodata))
    with open(os.path.join(root_dir,'ratiodata{}_imgList.txt'.format(ratiodata)), 'w') as fid:
        for anchor_name in imgList:
            fid.write(anchor_name+'\n')
        
