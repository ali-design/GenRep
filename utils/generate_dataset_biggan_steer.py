''' 
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
import random
import pixel_transformations
import oyaml as yaml

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

def sample(opt):
    model_name = opt.model_name
    output_path = opt.output_path
    partition = opt.partition
    # start_seed, nimg = constants.get_seed_nimg(partition)
    start_seed = opt.start_seed
    nimg = opt.num_imgs
    model_name = 'biggan-deep-%s' % opt.size
    truncation = opt.truncation
    imformat = opt.imformat
    batch_size = opt.batch_size
    
#     with open('./imagenet_class_index.json', 'rb') as fid:
    with open('./imagenet100_class_index.json', 'rb') as fid:
        imagenet_class_index_dict = json.load(fid)
    imagenet_class_index_keys = list(imagenet_class_index_dict.keys())
    print('Loading the model ...')
    
    model = BigGAN.from_pretrained(model_name)
    if torch.cuda.device_count() > 1:
        print('Using {} gpus for G'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    model.to('cuda')
    
    # load pretrained walk
    walk_composed_final_path = ('./walk_weights_biggan_deep/w_composed_160k_lr0.001_final.pth')
    walk_composed_final = torch.load(walk_composed_final_path)
    walk_color = walk_composed_final['walk_color'].to('cuda')
    walk_rot3d = walk_composed_final['walk_rot3d'].to('cuda')
    walk_rot2d = walk_composed_final['walk_rot2d'].to('cuda')
    walk_zoom = walk_composed_final['walk_zoom'].to('cuda')
    walk_shiftx = walk_composed_final['walk_shiftx'].to('cuda')
    walk_shifty = walk_composed_final['walk_shifty'].to('cuda')
    # transforms
    rot3d_transform = pixel_transformations.Rot3dTransform()
    rot2d_transform = pixel_transformations.Rot2dTransform()
    zxy_transform = pixel_transformations.ZoomShiftXYTransform()
    color_transform = pixel_transformations.ColorTransform()

    random.shuffle(imagenet_class_index_keys)
    for key in tqdm(imagenet_class_index_keys):
        class_dir_name = os.path.join(output_path, partition, imagenet_class_index_dict[key][0])
        if os.path.isdir(class_dir_name):
            continue
        os.makedirs(class_dir_name, exist_ok=True)
        idx = int(key)
        z_dict = dict()
        print('Generating images for class {}'.format(idx))
        class_vectors = one_hot_from_int(idx, batch_size=nimg)
        class_vectors = torch.from_numpy(class_vectors).to('cuda')
    
        seed = start_seed + idx
        noise_vectors = truncated_noise_sample(truncation=truncation, batch_size=nimg, seed=seed)
        noise_vectors = torch.from_numpy(noise_vectors).to('cuda')
            
        for batch_start in range(0, nimg, batch_size):
            s = slice(batch_start, min(nimg, batch_start + batch_size))
            ys = class_vectors[s]
            zs = noise_vectors[s]
            tbs = zs.shape[0]
            
            # get anchors
            with torch.no_grad():
                out_anchors = model(zs, ys, truncation)
        
            out_anchors = out_anchors.cpu()
            ims_anchors = convert_to_images(out_anchors)

            # get neighbors
            for ii in range(opt.num_neighbors):
                # alphas for transforms
                # 3D
                _, alphas_rot3d_graph = rot3d_transform.get_alphas(tbs)
                alphas_rot3d_graph = torch.tensor(alphas_rot3d_graph, device='cuda', dtype=torch.float32)
                # 2D
                _, alphas_rot2d_graph = rot2d_transform.get_alphas(tbs)
                alphas_rot2d_graph = torch.tensor(alphas_rot2d_graph, device='cuda', dtype=torch.float32)
                # Zoom, shiftx, shifty
                alphas_zxy = zxy_transform.get_alphas(tbs)
                alphas_zoom_graph = torch.tensor(alphas_zxy[1], device='cuda', dtype=torch.float32)
                alphas_shiftx_graph = torch.tensor(alphas_zxy[3], device='cuda', dtype=torch.float32)
                alphas_shifty_graph = torch.tensor(alphas_zxy[5], device='cuda', dtype=torch.float32)
                # Color
                _, alphas_color_graph = color_transform.get_alphas(tbs)
                alphas_color_graph = torch.tensor(alphas_color_graph, device='cuda', dtype=torch.float32)

                # generate neighbors
                z_new = zs + 5*alphas_rot3d_graph * walk_rot3d + alphas_rot2d_graph * walk_rot2d + \
                        alphas_zoom_graph * walk_zoom + alphas_shiftx_graph * walk_shiftx + alphas_shifty_graph * walk_shifty
                for c in range(color_transform.num_channels):
                    z_new = z_new + alphas_color_graph[:,c].unsqueeze(1) * walk_color[:,:,c]

                with torch.no_grad():
                    out_neighbors = model(z_new, ys, truncation)
 
                out_neighbors = out_neighbors.cpu()
                ims_neighbors = convert_to_images(out_neighbors)
                # save anchor and its neighbors
                # save anchors
                for b in range(tbs):
                    if ii == 0: 
                        im = ims_anchors[b]
                        im_name = 'seed%04d_sample%05d_anchor.%s' % (seed, batch_start+b, imformat)
                        im.save(os.path.join(class_dir_name, im_name))
                        z_dict[im_name] = [zs[b].cpu().numpy(), idx]

                    im = ims_neighbors[b]
                    im_name = 'seed%04d_sample%05d_neighbor_%d.%s' % (seed, batch_start+b, ii, imformat)
                    im.save(os.path.join(class_dir_name, im_name))
                    z_dict[im_name] = [z_new[b].detach().cpu().numpy(), idx]

        with open(os.path.join(class_dir_name, 'z_dataset.pkl'), 'wb') as fid:
            pickle.dump(z_dict,fid)
    
    pix_transforms_alphas_dict = {'rot3d_alpha_max': rot3d_transform.alpha_max, 
                                  'rot2d_alpha_max': rot2d_transform.alpha_max, 
                                  'zoom_alpha_max': zxy_transform.alpha_max_zoom, 
                                  'shiftxy_alpha_max': zxy_transform.alpha_max_shift, 
                                  'color_alpha_max': color_transform.alpha_max}
    return pix_transforms_alphas_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Sample from biggan")
    parser.add_argument('--out_dir', default='/data/vision/phillipi/ganclr/datasets', type=str)
    parser.add_argument('--partition', default='train', type=str)
    parser.add_argument('--truncation', default=1.0, type=float)
    parser.add_argument('--size', default=256, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--imformat', default='png', type=str)
    parser.add_argument('--num_imgs', default=1300, type=int, help='num imgs per class')
    parser.add_argument('--start_seed', default=0, type=int)
    parser.add_argument('--num_neighbors', default=20, type=int, help='num samples per anchor')
    parser.add_argument('--desc', default='steer_pth_imagenet100', type=str, help='this will be the tag of this specfic dataset, added to the end of the dataset name')
    
    opt = parser.parse_args()
    print(opt)
    model_name = 'biggan-deep-%s' % opt.size
    output_path = (os.path.join(opt.out_dir, '{}_tr{}_{}_N{}'.format(model_name, 
                                                                     opt.truncation, 
                                                                     opt.desc, 
                                                                     opt.num_neighbors)))
    
    parser.add_argument('--model_name', default=model_name)
    parser.add_argument('--output_path', default=output_path)
    opt = parser.parse_args()
    print(opt)
    pix_transforms_alphas_dict = sample(opt)
    with open(os.path.join(opt.output_path, opt.partition, 'steer_alphas_config.yml'), 'w') as fid:
        yaml.dump(pix_transforms_alphas_dict, fid, default_flow_style=False)
    
    
    
