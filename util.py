from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets
import glob
import os 
from PIL import Image
import json

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
