import io
import IPython.display
import PIL.Image
from pprint import pformat

import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tensorflow_hub as hub
from scipy.stats import truncnorm

import utils_bigbigan as ubigbi

import os
import argparse
from tqdm import tqdm
import json
import pickle

def truncated_noise_sample_neighbors(batch_size=1, dim_z=120, truncation=1., seed=None, num_neighbors=20, scale=1.0):
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
    state = None if seed is None else np.random.RandomState(seed)
    zs = truncation * truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    list_results.append(zs) # these are anchors

    for i in range(num_neighbors):
        state_neighbors = None if seed is None else np.random.RandomState(seed+1000+i)
        values_neighbors = truncation * truncnorm.rvs(-2, 2, size=(batch_size, dim_z), 
                                                      scale=scale, 
                                                      random_state=state_neighbors).astype(np.float32)
        zs_norm = np.linalg.norm(zs)
        zs_new = zs + values_neighbors
        zs_angular = zs_norm * zs_new/np.linalg.norm(zs_new)
#         print('z_new.shape, zs_angular.shape:', zs_new.shape, zs_angular.shape)
        list_results.append(zs_angular)

    return list_results

    
def sample(opt):
    output_path = (os.path.join(opt.out_dir, 
                                'bigbi-{}_128_gauss_angular_std{}_imagenet100_N{}'.format(opt.encoder_type, 
                                                                                          opt.scale, 
                                                                                          opt.num_neighbors)))
    partition = opt.partition
    # start_seed, nimg = constants.get_seed_nimg(partition)
    start_seed = opt.start_seed
    nimg = opt.num_imgs
    
    if opt.encoder_type == 'resnet':
        module_path = 'https://tfhub.dev/deepmind/bigbigan-resnet50/1'  # ResNet-50
    elif opt.net_type == 'revnet':
        module_path = 'https://tfhub.dev/deepmind/bigbigan-revnet50x4/1'  # RevNet-50 x4
    
    module = hub.Module(module_path)  # inference
    bigbigan = ubigbi.BigBiGAN(module)
    # Make input placeholders for z (`gen_ph`).
    gen_ph = bigbigan.make_generator_ph()
    # Compute samples G(z) from encoder input z (`gen_ph`).
    gen_samples = bigbigan.generate(gen_ph)
    ## Create a TensorFlow session and initialize variables
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)   

    batch_size = opt.batch_size
    with open('./imagenet_class_index.json', 'rb') as fid:
        imagenet_class_index_dict = json.load(fid)
    imagenet_class_index_keys = list(imagenet_class_index_dict.keys())
    
    list100 = os.listdir('/data/scratch-oc40/jahanian/ganclr_results/ImageNet100/train')
    
    z_dict = dict()
    import random
    random.shuffle(imagenet_class_index_keys)
    for key in tqdm(imagenet_class_index_keys):
        if imagenet_class_index_dict[key][0] not in list100:
            continue
        class_dir_name = os.path.join(output_path, partition, imagenet_class_index_dict[key][0])
        if os.path.isdir(class_dir_name):
            continue
        os.makedirs(class_dir_name, exist_ok=True)
        idx = int(key)
        print('Generating images for class {}'.format(idx))
#         class_vector = one_hot_from_int(idx, batch_size=nimg)
        seed = start_seed + idx
        noise_vector_neighbors = truncated_noise_sample_neighbors(batch_size=nimg,
                                                                  seed=seed,
                                                                  scale=opt.scale, 
                                                                  num_neighbors=opt.num_neighbors)
#         class_vector = torch.from_numpy(class_vector).cuda()
        for ii in range(len(noise_vector_neighbors)):
            noise_vector = noise_vector_neighbors[ii]
            for batch_start in range(0, nimg, batch_size):
                s = slice(batch_start, min(nimg, batch_start + batch_size))
                
                feed_dict = {gen_ph: noise_vector[s]}
                anchor_out = sess.run(gen_samples, feed_dict=feed_dict)
                ims = ubigbi.image_to_uint8(anchor_out)
                for i, im in enumerate(ims):
                    if ii == 0: #anchors
                        im_name = 'seed%04d_sample%05d_anchor.%s' % (seed, batch_start+i, opt.imformat)
                    else:
                        im_name = 'seed%04d_sample%05d_1.0_%d.%s' % (seed, batch_start+i, ii, opt.imformat)

                    im = PIL.Image.fromarray(im)
                    im.save(os.path.join(class_dir_name, im_name))
                    z_dict[im_name] = [noise_vector[i], idx]
        with open(os.path.join(class_dir_name, 'z_dataset.pkl'), 'wb') as fid:
            pickle.dump(z_dict,fid)                                                        
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Sample from biggan")
    parser.add_argument('--out_dir', default='/data/vision/phillipi/ganclr/datasets', type=str)
    parser.add_argument('--partition', default='train', type=str)
#     parser.add_argument('--truncation', default=1.0, type=float)
#     parser.add_argument('--size', default=256, type=int)
    parser.add_argument('--encoder_type', default='resnet', type=str, help='bigbigan encoder type')
    parser.add_argument('--scale', default=0.3, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--imformat', default='png', type=str)
    parser.add_argument('--num_imgs', default=1300, type=int, help='num imgs per class')
    parser.add_argument('--start_seed', default=0, type=int)
    parser.add_argument('--num_neighbors', default=20, type=int, help='num samples per anchor')
#     parser.add_argument('--desc', default='steer_rnd_std1.0_100', type=str, help='this will be the tag of this specfic dataset, added to the end of the dataset name')
    opt = parser.parse_args()
    sample(opt)


