''' 
code adapted from lucy's code
you will need to pip install pytorch_pretrained_biggan, see https://github.com/huggingface/pytorch-pretrained-BigGAN
'''

import torch
import tqdm
from pytorch_pretrained_biggan import (
    BigGAN,
    truncated_noise_sample,
    one_hot_from_int
)
import PIL.Image
import numpy as np
import os
import argparse
from tqdm import tqdm
import json
import pickle
from scipy.stats import truncnorm

def convert_to_images(obj):
    """ Convert an output tensor from BigGAN in a list of images.
    """
    # need to fix import, see: https://github.com/huggingface/pytorch-pretrained-BigGAN/pull/14/commits/68a7446951f0b9400ebc7baf466ccc48cdf1b14c
    if not isinstance(obj, np.ndarray):
        obj = obj.detach().numpy()
    obj = obj.transpose((0, 2, 3, 1))
    obj = np.clip(((obj + 1) / 2.0) * 256, 0, 255)
    img = []
    for i, out in enumerate(obj):
        out_array = np.asarray(np.uint8(out), dtype=np.uint8)
        img.append(PIL.Image.fromarray(out_array))
    return img

def truncated_noise_sample_neighbors(batch_size=1, dim_z=128, truncation=1., seed=None, num_neighbors=20, scale=1.0):
    """ Create a truncated noise vector.
        Params:
            batch_size: batch size.
            dim_z: dimension of z
            truncation: truncation value to use
            seed: seed for the random generator
        Output:
            array of shape (batch_size, dim_z)
    """
    list_results = []
    list_results2 = []
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    zs = truncation * values
    list_results.append(zs) # these are anchors
    list_results2.append(zs)
    
    state_neighbors = None if seed is None else np.random.RandomState(seed+1000)
    for i in range(num_neighbors):
        state_neighbors = None if seed is None else np.random.RandomState(seed+1000+i)
        values_neighbors = truncation * truncnorm.rvs(-2, 2, size=(batch_size, dim_z), scale=scale, random_state=state_neighbors).astype(np.float32)
        zs_norm = np.linalg.norm(zs)
        zs_new = zs + values_neighbors
        zs_angular = zs_norm * zs_new/np.linalg.norm(zs_new)
#         print('z_new.shape, zs_angular.shape:', zs_new.shape, zs_angular.shape)
        list_results.append(zs_angular)
        list_results2.append(zs_new)
        
    return list_results, list_results2

def sample(opt):
    output_path = (os.path.join(opt.out_dir, 'biggan%dtr%d-%s_%s_%d_samples' %
                   (opt.size, int(opt.truncation), opt.imformat, opt.desc, opt.num_neighbors)))
    partition = opt.partition
    # start_seed, nimg = constants.get_seed_nimg(partition)
    start_seed = opt.start_seed
    nimg = opt.num_imgs
    model_name = 'biggan-deep-%s' % opt.size
    truncation = opt.truncation
    imformat = opt.imformat
    batch_size = opt.batch_size
    with open('./imagenet_class_index.json', 'rb') as fid:
        imagenet_class_index_dict = json.load(fid)
    imagenet_class_index_keys = list(imagenet_class_index_dict.keys())
    print('Loading the model ...')
    model = BigGAN.from_pretrained(model_name).cuda()
    
    list100 = os.listdir('/data/scratch-oc40/jahanian/ganclr_results/ImageNet100/train')
    import random
    random.shuffle(imagenet_class_index_keys)
    for key in tqdm(imagenet_class_index_keys):
        if imagenet_class_index_dict[key][0] not in list100:
            continue

        class_dir_name = os.path.join(output_path, partition, imagenet_class_index_dict[key][0])
        print(class_dir_name)
        if os.path.isdir(class_dir_name):
            continue

        os.makedirs(class_dir_name, exist_ok=True)
        idx = int(key)
        z_dict = dict()

        print('Generating images for class {}'.format(idx))
        class_vector = one_hot_from_int(idx, batch_size=nimg)
        seed = start_seed + idx
        noise_vector_neighbors, nvn2 = truncated_noise_sample_neighbors(truncation=truncation,
                                                        batch_size=nimg,
                                                        seed=seed, num_neighbors=opt.num_neighbors)
        class_vector = torch.from_numpy(class_vector).cuda()
        for ii in range(len(noise_vector_neighbors)):
            noise_vector = noise_vector_neighbors[ii]
            noise_vector = torch.from_numpy(noise_vector).cuda()
            nv2 = nvn2[ii]
            nv2 = torch.from_numpy(nv2).cuda()
            for batch_start in range(0, nimg, batch_size):
                s = slice(batch_start, min(nimg, batch_start + batch_size))

                with torch.no_grad():
                    output = model(noise_vector[s], class_vector[s], truncation)
                    output2 = model(nv2[s], class_vector[s], truncation)
                    
                output = output.cpu()
                ims = convert_to_images(output)
                output2 = output2.cpu()
                ims2 = convert_to_images(output2)
                for i, im in enumerate(ims):
                    if ii == 0: #anchors
                        im_name = 'seed%04d_sample%05d_anchor.%s' % (seed, batch_start+i, imformat)
                        im.save(os.path.join(class_dir_name, im_name))
                        z_dict[im_name] = [noise_vector[batch_start+i].cpu().numpy(), idx]
                    else:
                        im_name1 = 'seed%04d_sample%05d_ang_%d.%s' % (seed, batch_start+i, ii, imformat)
                        im.save(os.path.join(class_dir_name, im_name1))
                        z_dict[im_name1] = [noise_vector[batch_start+i].cpu().numpy(), idx]
                        im_name2 = 'seed%04d_sample%05d_iso_%d.%s' % (seed, batch_start+i, ii, imformat)
                        im2 = ims2[i]
                        im2.save(os.path.join(class_dir_name, im_name2))
                        z_dict[im_name2] = [nv2[batch_start+i].cpu().numpy(), idx]
                        
        with open(os.path.join(class_dir_name, 'z_dataset.pkl'), 'wb') as fid:
            pickle.dump(z_dict,fid)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Sample from biggan")
    parser.add_argument('--out_dir', default='/data/vision/phillipi/ganclr/datasets', type=str)
    parser.add_argument('--partition', default='train', type=str)
    parser.add_argument('--truncation', default=1.0, type=float)
    parser.add_argument('--size', default=256, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--imformat', default='png', type=str)
    parser.add_argument('--num_imgs', default=10, type=int, help='num imgs per class')
    parser.add_argument('--start_seed', default=0, type=int)
    parser.add_argument('--scale', default=1.0, type=int, help='scale of std')
    parser.add_argument('--num_neighbors', default=10, type=int, help='num samples per anchor')
    parser.add_argument('--desc', default='paper_figure', type=str, help='this will be the tag of this specfic dataset, added to the end of the dataset name')
    opt = parser.parse_args()
    sample(opt)
