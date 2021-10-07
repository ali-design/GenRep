from __future__ import print_function

import os
import sys
import argparse
import ipdb
import time
import math
from tqdm import tqdm
import numpy as np
sys.path.append('../')

import torch
from torch import nn
import torch.backends.cudnn as cudnn
# from sklearn import svm
from torchvision import transforms, datasets
from torchvision.models import resnet50
# from util import AverageMeter
# from util import adjust_learning_rate, warmup_learning_rate, accuracy
from torchnet.meter import mAPMeter
# from util import set_optimizer
from util import VOCDetectionDataset, Caltech101
from networks.resnet_big import SupConResNet, LinearClassifier

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=60,
                        help='number of training epochs')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='biggan',
                choices=['biggan', 'cifar10', 'cifar100', 'imagenet100', 'imagenet100K', 'imagenet', 
                         'voc2007', 'caltech101', 'birdsnap', 'sun', 'cars', 'aircraft', 'dtd', 'pets', 'flowers'], help='dataset')

    # other setting
    parser.add_argument('--model_weight', type=str, default='ckpt',
                        choices=['ckpt', 'scratch', 'pretrained'])
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    parser.add_argument('--out', type=str, default='',
                        help='path to pre-trained model')

    # specifying folders
    parser.add_argument('-d', '--data_folder', type=str,
                        default='',
                        help='the data folder')
    parser.add_argument('--expname', type=str)

    opt = parser.parse_args()



    # warm-up for large-batch training,
    
    opt.img_size = 256
    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'biggan' or opt.dataset == 'imagenet100' or opt.dataset == 'imagenet100K' or opt.dataset == 'imagenet':
        opt.n_cls = 1000
    elif opt.dataset == 'voc2007':
        opt.n_cls = 20
    elif opt.dataset == 'caltech101':
        opt.n_cls = 101
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_loader(opt):
    # construct data loader
    transfer_dataset = ['voc2007', 'caltech101']
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'biggan' or opt.dataset == 'imagenet100' or opt.dataset == 'imagenet100K' or opt.dataset == 'imagenet' or opt.dataset in transfer_dataset:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    # NOTE: Following SimCLLR we resize to 224x224 or similar and do center crop of 224
    if opt.dataset == 'biggan' or opt.dataset == 'imagenet100' or opt.dataset == 'imagenet100K' or opt.dataset == 'imagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(int(opt.img_size*0.875), scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        # Todo: arg this 256
        val_transform = transforms.Compose([
            transforms.Resize(int(opt.img_size*0.875)),
            transforms.CenterCrop(int(opt.img_size*0.875)),
            transforms.ToTensor(),
            normalize,
        ])
    elif opt.dataset in transfer_dataset:

        train_transform = transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.RandomResizedCrop(int(opt.img_size*0.875), scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        # Todo: arg this 256
        val_transform = transforms.Compose([
            transforms.Resize(int(opt.img_size*0.875)),
            transforms.CenterCrop(int(opt.img_size*0.875)),
            transforms.ToTensor(),
            normalize,
        ])

    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.img_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    if opt.dataset == 'biggan' or opt.dataset == 'imagenet100' or opt.dataset == 'imagenet100K' or opt.dataset == 'imagenet':
        train_dataset = datasets.ImageFolder(root=os.path.join(opt.data_folder, 'train'),
                                             transform=train_transform)
        val_dataset = datasets.ImageFolder(root=os.path.join(opt.data_folder, 'val'),
                                           transform=val_transform)
    if opt.dataset == 'voc2007':
        train_dataset = VOCDetectionDataset(root=opt.data_folder,
                                              year='2007',
                                              image_set='train',
                                              transform=val_transform)

        val_dataset = VOCDetectionDataset(root=opt.data_folder,
                                              year='2007',
                                              image_set='val',
                                              transform=val_transform)
    elif opt.dataset == 'caltech101':
        train_dataset = Caltech101(root=opt.data_folder,
                                   split='train',
                                   transform=val_transform)


        val_dataset = Caltech101(root=opt.data_folder,
                                   split='val',
                                   transform=val_transform)

    else:
        raise ValueError(opt.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True, sampler=None)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

    return train_loader, val_loader


def set_model(opt):
    if opt.model_weight != 'pretrained':
        model = SupConResNet(name=opt.model, img_size=opt.img_size)
    else:
        if opt.model_weight == 'pretrained':
            model = resnet50(pretrained=True).cuda()
            modules = list(model.children())[:-1]
            model = nn.Sequential(*modules)

        if opt.model_weight == 'scratch':
            pass

    criterion = None
    classifier = None

    
    if opt.model_weight == 'ckpt':
        ckpt = torch.load(opt.ckpt, map_location='cpu')
        state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            if opt.model_weight == 'ckpt':
                new_state_dict = {}
                for k, v in state_dict.items():
                    k = k.replace("module.", "")
                    new_state_dict[k] = v
                state_dict = new_state_dict
        model = model.cuda()
        cudnn.benchmark = True
        if opt.model_weight == 'ckpt':
            model.load_state_dict(state_dict, strict=False)
    #ipdb.set_trace()
    return model, classifier, criterion



def extract(val_loader, model, opt):
    """validation"""
    model.eval()
    labels_all = []
    fts_all = []

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(tqdm(val_loader)):
            images = images.float().cuda()
            labels = labels.cuda()

            if opt.model_weight == 'pretrained':
                output = model(images).squeeze(-1).squeeze(-1)
            else:
                output = model.encoder(images)
            fts_all.append(output.cpu().numpy())
            labels_all.append(labels.cpu().numpy())

    fts_all = np.concatenate(fts_all, 0)
    labels_all = np.concatenate(labels_all, 0)
    return fts_all, labels_all

def main():
    print("Extracting...")
    best_acc = 0
    opt = parse_option()
    mode = opt.expname
    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # training routine
    dataset = opt.dataset
    features_folder = opt.out
    f1 = '{}/train.npz'.format(features_folder)
    f2 = '{}/val.npz'.format(features_folder)
    print(f1, f2)
    fts_train, labels_train = extract(train_loader, model, opt)
    fts_val, labels_val = extract(val_loader, model, opt)

    dir_name = os.path.dirname(f1)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    np.savez(f1, features=fts_train, labels=labels_train)
    np.savez(f2, features=fts_val, labels=labels_val)


if __name__ == '__main__':
    main()
