import os
import glob
import argparse
from tqdm import tqdm
import json
import pickle
import random
import oyaml as yaml

# for the model
import torch
from stylegan2_utils import renormalize, nethook
from stylegan2_utils.stylegan2 import load_seq_stylegan
from PIL import Image
import numpy as np

if torch.cuda.is_available():
    print('cuda is available.')
    device = 'cuda'
else:
    print('No cuda available!')
    device = 'cpu'
# load the model
gan_model = load_seq_stylegan('car', mconv='seq', truncation=0.90)
nethook.set_requires_grad(False, gan_model)


def truncated_noise_sample_neighbors(batch_size=1, dim_z=512, truncation=1., seed=None, num_neighbors=0, scale=0.25):
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

    np.random.seed(seed)
    zs = torch.tensor(np.random.normal(0.0, 1.0, [batch_size, 512]), dtype=torch.float32, requires_grad=False).to(device)
    list_results.append(zs) # these are anchors

    for i in range(num_neighbors):
        np.random.seed(seed+1000+i)
        values_neighbors = torch.tensor(np.random.normal(0.0, scale, [batch_size, 512]), dtype=torch.float32,
                                        requires_grad=False).to(device)

        list_results.append(zs + values_neighbors)

    return list_results
    
def sample(opt):
    output_path = opt.output_path
    partition = opt.partition
    # start_seed, nimg = constants.get_seed_nimg(partition)
    start_seed = opt.start_seed
    nimg = opt.num_imgs
    
  

    batch_size = opt.batch_size
    if opt.dataset_type == 100:
        with open('./imagenet100_class_index.json', 'rb') as fid:
            imagenet_class_index_dict = json.load(fid)
    elif opt.dataset_type == 1000:
        with open('./imagenet_class_index.json', 'rb') as fid:
            imagenet_class_index_dict = json.load(fid)
    elif opt.dataset_type == 893:
        with open('./imagenet893_class_index.json', 'rb') as fid:
            imagenet_class_index_dict = json.load(fid)
            
    imagenet_class_index_keys = list(imagenet_class_index_dict.keys())
        
    random.shuffle(imagenet_class_index_keys)
    for key in tqdm(imagenet_class_index_keys): 
        class_dir_name = os.path.join(output_path, partition, imagenet_class_index_dict[key][0])
        if os.path.isdir(class_dir_name):
            continue
        os.makedirs(class_dir_name, exist_ok=True)
        idx = int(key)
        z_dict = dict()
                
        print('Generating images for class {}, with number of images to be {}'.format(idx, nimg))
        seed = start_seed + idx
        noise_vector_neighbors = truncated_noise_sample_neighbors(batch_size=nimg,
                                                                  seed=seed, 
                                                                  truncation = opt.truncation,
                                                                  num_neighbors=opt.num_neighbors,
                                                                  scale=opt.std)
        for ii in range(len(noise_vector_neighbors)):
            noise_vector = noise_vector_neighbors[ii]
            for batch_start in range(0, nimg, batch_size):
                s = slice(batch_start, min(nimg, batch_start + batch_size))
                with torch.no_grad():
                    ims = gan_model(noise_vector[s])
    
                for i, im in enumerate(ims):
                    if ii == 0: #anchors
                        im_name = 'seed%04d_sample%05d_anchor.%s' % (seed, batch_start+i, opt.imformat)
                    else:
                        im_name = 'seed%04d_sample%05d_1.0_%d.%s' % (seed, batch_start+i, ii, opt.imformat)

                    im = renormalize.as_image(im)
                    im = Image.fromarray(np.array(im)[64:448,:,:])
                    im.save(os.path.join(class_dir_name, im_name))
                    z_dict[im_name] = [noise_vector[batch_start+i].cpu().numpy(), idx]
        with open(os.path.join(class_dir_name, 'z_dataset.pkl'), 'wb') as fid:
            pickle.dump(z_dict,fid)                                                        
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Sample from stylegan2 cars")
    parser.add_argument('--out_dir', default='/data/vision/phillipi/ganclr/datasets', type=str)
    parser.add_argument('--partition', default='train', type=str)
    parser.add_argument('--truncation', default=0.9, type=float)
#     parser.add_argument('--size', default=256, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--imformat', default='png', type=str)
    parser.add_argument('--num_imgs', default=1000, type=int, help='num imgs per class')
    parser.add_argument('--start_seed', default=0, type=int)
    parser.add_argument('--dataset_type', default=893, type=int, choices=[100, 893, 1000],help='choices: 100, 893, or 1000, equivalent to number of images in image100, cars, or imagenet1000')
    parser.add_argument('--num_neighbors', default=1, type=int, help='num samples per anchor')
    parser.add_argument('--std', default=0.25, type=float, help='std for gaussian in z space')


    opt = parser.parse_args()
    if opt.num_neighbors == 0:
        output_path = (os.path.join(opt.out_dir, 'stylegan2_cars{}_tr{}'.format(opt.dataset_type, opt.truncation)))
        opt.std = 1.0 # we only sample from normal
    else:
        output_path = (os.path.join(opt.out_dir, 
                       'stylegan2_cars{}_tr{}_gauss1_std{}_NS{}_NN{}'.format(opt.dataset_type, opt.truncation, 
                                                                       opt.std, opt.num_imgs, opt.num_neighbors)))
    opt.output_path = output_path
    print(opt)
    if not os.path.isdir(opt.output_path):
        os.makedirs(opt.output_path, exist_ok=True)
    with open(os.path.join(opt.output_path, 'opt_summary.yml'), 'w') as fid:
        yaml.dump(vars(opt), fid, default_flow_style=False)
    sample(opt)