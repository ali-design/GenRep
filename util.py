from __future__ import print_function

import ipdb
import math
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from pytorch_pretrained_biggan import (
    BigGAN,
    truncated_noise_sample,
    one_hot_from_int
)
from torchvision import datasets
import random
import glob
import os 
from PIL import Image
import json
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
        img.append(Image.fromarray(out_array))
    return img

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


class OnlineGansetDataset(Dataset):
    """The idea is to load the anchor image and its neighbor"""

    def __init__(self, root_dir, neighbor_std=1.0, transform=None, truncation=1.0, dim_z=128,
                 seed=None, walktype='gaussian', uniformb=None, num_samples=130000, size_biggan=256, device_id=0):
        """
        Args:
            neighbor_std: std in the z-space
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            walktype: whether we are moving in a gaussian ball or a uniform ball
        """
        super(OnlineGansetDataset, self).__init__()
        self.neighbor_std = neighbor_std
        self.uniformb = uniformb
        self.root_dir = root_dir
        self.transform = transform
        self.walktype = walktype
        self.classes, self.class_to_idx = self.find_classes(self.root_dir)
        self.dim_z = dim_z
        self.truncation = truncation
        self.random_state = None if seed is None else np.random.RandomState(seed) 

        model_name = 'biggan-deep-%s' % size_biggan
        self.device_id = device_id
        self.model = BigGAN.from_pretrained(model_name).cuda(self.device_id)
        self.num_samples = num_samples
        with open('utils/imagenet_class_index.json', 'rb') as fid:
            self.imagenet_class_index_dict = json.load(fid)
        list100 = os.listdir('/data/scratch-oc40/jahanian/ganclr_results/ImageNet100/train')
        self.class_indices_interest = []

        imagenet_class_index_keys = self.imagenet_class_index_dict.keys()
        for key in imagenet_class_index_keys:
            if self.imagenet_class_index_dict[key][0] in list100:
                self.class_indices_interest.append(key)


        # get list of anchor images

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def gen_images(self, z, class_ind, use_grad=False):
        if class_ind.ndim == 3:
            class_ind = class_ind.squeeze(1)
            z = z.squeeze(1)

        class_vector = class_ind.cuda(self.device_id)
        noise_vector = z.cuda(self.device_id)
        if not use_grad:
            with torch.no_grad():
                output = self.model(noise_vector, class_vector, self.truncation)
        else:
            output = self.model(noise_vector, class_vector, self.truncation)
        output = output.cpu()
        output = convert_to_images(output)
        # TODO: clipping and other ops to convert to image. Unsure if differentiable
        return output

    def gen_images_transform(self, z, class_ind, use_grad=False):
        images = self.gen_images(z, class_ind, use_grad)
        images_orig = images
        if self.transform:
            images = [self.transform(image) for image in images]
        images = torch.cat([im.unsqueeze(0) for im in images], 0)
        return images, images_orig

    def sample_class(self):
        indices = random.choices(self.class_indices_interest, k=1)
        return indices

    def sample_noise(self):
        values = truncnorm.rvs(-2, 2, size=(1, self.dim_z), random_state=self.random_state).astype(np.float32)
        return values

    def sample_neighbors(self, values):
        if self.walktype == 'gaussian':
            neighbors = np.random.normal(0, self.neighbor_std, size=(1, self.dim_z)).astype(np.float32)
        else:
            raise NotImplementedError
        return values + neighbors
            


        


    def __len__(self):
        
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = int(self.sample_class()[0])

        one_hot_index = one_hot_from_int(label)
        w = self.sample_noise()
        dw = self.sample_neighbors(w)



        return w, dw, one_hot_index, label


class GansetDataset(datasets.ImageFolder):
    """The idea is to load the anchor image and its neighbor"""

    def __init__(self, root_dir, neighbor_std=1.0, transform=None, walktype='gaussian', uniformb=None):
        """
        Args:
            neighbor_std: std in the z-space
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            walktype: whether we are moving in a gaussian ball or a uniform ball
        """
        super(GansetDataset, self).__init__(root_dir, transform, target_transform=None)
        self.neighbor_std = neighbor_std
        self.uniformb = uniformb
        self.root_dir = root_dir
        self.transform = transform
        self.walktype = walktype
        self.classes, self.class_to_idx = self._find_classes(self.root_dir)

        # get list of anchor images
        self.imglist = glob.glob(os.path.join(self.root_dir, '*/*_anchor.png'))
        self.dir_size = len(self.imglist)
        print('Length: {}'.format(self.dir_size))

    def __len__(self):
        
        return self.dir_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.imglist[idx]
        image = Image.open(img_name)
        img_name_neighbor = self.imglist[idx].replace('anchor',str(self.neighbor_std))
        image_neighbor = Image.open(img_name_neighbor)
        label = self.imglist[idx].split('/')[-2]
        # with open('./utils/imagenet_class_index.json', 'rb') as fid:
        #     imagenet_class_index_dict = json.load(fid)
        # for key, value in imagenet_class_index_dict.items():
        #     if value[0] == label:
        #         label = key
        #         break
        label = self.class_to_idx[label]
        if self.transform:
            image = self.transform(image)
            image_neighbor = self.transform(image_neighbor)

        return image, image_neighbor, label


class GansteerDataset(datasets.ImageFolder):
    """The idea is to load the negative-alpha image and its neighbor (positive-alpha)"""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(GansteerDataset, self).__init__(root_dir, transform, target_transform=None)
        self.root_dir = root_dir
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(self.root_dir)

        # get list of nalpha images
        self.imglist = glob.glob(os.path.join(self.root_dir, '*/*_anchor.png'))
        self.dir_size = len(self.imglist)

    def __len__(self):
        
        return self.dir_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.imglist[idx]
        # print('anchor image:', img_name)
        image = Image.open(img_name)
        if (int(os.path.basename(self.imglist[idx]).split('_')[4].replace('sample', '')) % 2 ==0):
            img_name_neighbor = self.imglist[idx].replace('anchor','palpha')
        else:
            img_name_neighbor = self.imglist[idx].replace('anchor','nalpha')

        # lets randomly switch to a steered color neighbor:
        coin = np.random.rand()
        # if coin < 0.66 and coin >= 0.33:
        #     img_name_neighbor.replace('biggan256tr1-png_steer_rot3d_100', 'biggan256tr1-png_steer_color_100')
        # elif coin < 0.33:
        #     img_name_neighbor.replace('biggan256tr1-png_steer_rot3d_100', 'biggan256tr1-png_steer_zoom_100')
        if coin < 0.5:
            color_list = ['W_sample', 'R_sample', 'G_sample', 'B_sample']
            color_choice = np.random.choice(color_list)
            img_name_neighbor = img_name_neighbor.replace('biggan256tr1-png_steer_zoom_100', 'biggan256tr1-png_steer_color_100')
            img_name_neighbor = img_name_neighbor.replace('sample', color_choice)

            if color_choice in ['R_sample', 'G_sample', 'B_sample'] and 'nalpha' in img_name_neighbor:
                img_name_neighbor = img_name_neighbor.replace('nalpha', 'palpha')

        # print('neighbor: ', img_name_neighbor)
        # print(os.path.exists(img_name_neighbor))
        image_neighbor = Image.open(img_name_neighbor)
        label = self.imglist[idx].split('/')[-2]
        # with open('./utils/imagenet_class_index.json', 'rb') as fid:
        #     imagenet_class_index_dict = json.load(fid)
        # for key, value in imagenet_class_index_dict.items():
        #     if value[0] == label:
        #         label = key
        #         break
        label = self.class_to_idx[label]
        if self.transform:
            image = self.transform(image)
            image_neighbor = self.transform(image_neighbor)

        return image, image_neighbor, label
