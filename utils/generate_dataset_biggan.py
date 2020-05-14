''' 
code provided by lucy
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
    output_path = (os.path.join(opt.out_dir, 'biggan%dtr%d-%s' %
                   (opt.size, int(opt.truncation*10), opt.imformat)))
    partition = opt.partition
    # start_seed, nimg = constants.get_seed_nimg(partition)
    start_seed = opt.start_seed
    nimg = opt.num_imgs
    model_name = 'biggan-deep-%s' % opt.size
    truncation = opt.truncation
    imformat = opt.imformat
    batch_size = opt.batch_size

    model = BigGAN.from_pretrained(model_name).cuda()
    for idx in tqdm(range(1000)):
        os.makedirs(os.path.join(output_path, partition, '%04d' % idx),
                    exist_ok=True)
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
                im.save(os.path.join(
                    output_path, partition, '%04d' % idx,
                    'seed%04d_sample%05d.%s' % (seed, batch_start+i, imformat)
                ))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Sample from biggan")
    parser.add_argument('--out_dir', default='/data/scratch-oc40/jahanian/ganclr_results/', type=str)
    parser.add_argument('--partition', default='val', type=str)
    parser.add_argument('--truncation', default=1.0, type=float)
    parser.add_argument('--size', default=256, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--imformat', default='png', type=str)
    parser.add_argument('--num_imgs', help='num imgs per class', default=10, type=int)
    parser.add_argument('--start_seed', default=5000, type=int)
    opt = parser.parse_args()
    sample(opt)