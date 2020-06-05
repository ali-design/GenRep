''' 
code adapted from lucy's code
you will need to pip install pytorch_pretrained_biggan, see https://github.com/huggingface/pytorch-pretrained-BigGAN
'''

import torch
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
    output_path = (os.path.join(opt.out_dir, 'biggan%dtr%d-%s_100classes' %
                   (opt.size, int(opt.truncation), opt.imformat)))
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

    list100 = os.listdir('/data/scratch-oc40/jahanian/ganclr_results/ImageNet100/train')

    model = BigGAN.from_pretrained(model_name).cuda()
    imagenet_class_index_keys = imagenet_class_index_dict.keys()
    for key in imagenet_class_index_keys:
        if imagenet_class_index_dict[key][0] not in list100:
            continue

        class_dir_name = os.path.join(output_path, partition, imagenet_class_index_dict[key][0])
        os.makedirs(class_dir_name, exist_ok=True)
        idx = int(key)
        print('Generating images for class {}'.format(idx))
        class_vector = one_hot_from_int(idx, batch_size=nimg)
        seed = start_seed + idx
        noise_vector = truncated_noise_sample(truncation=truncation,
                                              batch_size=nimg,
                                              seed=seed)
        class_vector = torch.from_numpy(class_vector).cuda()
        noise_vector = torch.from_numpy(noise_vector).cuda()
        for batch_start in range(0, nimg, batch_size):
            s = slice(batch_start, min(nimg, batch_start + batch_size))

            with torch.no_grad():
                output = model(noise_vector[s], class_vector[s], truncation)
            output = output.cpu()
            ims = convert_to_images(output)
            for i, im in enumerate(ims):
                im.save(os.path.join(class_dir_name, 'seed%04d_sample%05d.%s' % (seed, batch_start+i, imformat)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Sample from biggan")
    parser.add_argument('--out_dir', default='/data/scratch-oc40/jahanian/ganclr_results/', type=str)
    parser.add_argument('--partition', default='train', type=str)
    parser.add_argument('--truncation', default=1.0, type=float)
    parser.add_argument('--size', default=256, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--imformat', default='png', type=str)
    parser.add_argument('--num_imgs', default=1300, type=int, help='num imgs per class')
    parser.add_argument('--start_seed', default=0, type=int)
    opt = parser.parse_args()
    sample(opt)