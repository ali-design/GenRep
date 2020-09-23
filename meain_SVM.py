from __future__ import print_function

import os
import sys
import argparse
import ipdb
import time
import math

import torch
import torch.backends.cudnn as cudnn

from torchvision import transforms, datasets
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from torchnet.meter import mAPMeter
from util import set_optimizer
from util import VOCDetectionDataset
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
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=60,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.3,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,40,50',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='biggan',
                        choices=['biggan', 'cifar10', 'cifar100', 'imagenet100', 'imagenet100K', 'imagenet', 'voc2007'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    # specifying folders
    parser.add_argument('-d', '--data_folder', type=str,
                        default='/data/scratch-oc40/jahanian/ganclr_results/ImageNet100',
                        help='the data folder')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.dataset == 'cifar10':
        opt.img_size = 32
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.img_size = 32
        opt.n_cls = 100
    elif opt.dataset == 'biggan' or opt.dataset == 'imagenet100' or opt.dataset == 'imagenet100K' or opt.dataset == 'imagenet':
        opt.img_size = 128
        opt.n_cls = 1000
    elif opt.dataset == 'voc2007':
        opt.img_size = 128
        opt.n_cls = 20
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'biggan' or opt.dataset == 'imagenet100' or opt.dataset == 'imagenet100K' or opt.dataset == 'imagenet' or opt.dataset == 'voc2007':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    if opt.dataset == 'biggan' or opt.dataset == 'imagenet100' or opt.dataset == 'imagenet100K' or opt.dataset == 'imagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(int(opt.img_size*0.875), scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        # Todo: arg this 256
        val_transform = transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.CenterCrop(int(opt.img_size*0.875)),
            transforms.ToTensor(),
            normalize,
        ])
    elif opt.dataset == "voc2007":

        train_transform = transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.RandomResizedCrop(int(opt.img_size*0.875), scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        # Todo: arg this 256
        val_transform = transforms.Compose([
            transforms.Resize(opt.img_size),
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

    if opt.dataset == 'voc2007':
        train_dataset = VOCDetectionDataset(root=opt.data_folder,
                                              year='2007',
                                              image_set='train',
                                              transform=val_transform)

        val_dataset = VOCDetectionDataset(root=opt.data_folder,
                                              year='2007',
                                              image_set='val',
                                              transform=val_transform)

    else:
        raise ValueError(opt.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

    return train_loader, val_loader


def set_model(opt):
    model = SupConResNet(name=opt.model, img_size=opt.img_size)
    if opt.dataset == 'voc2007':
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion



def extract(val_loader, model, opt, path_file_out):
    """validation"""
    model.eval()
    labels_all = []
    fts_all = []

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            
            output = model.encoder(images)
            ipdb.set_trace()
            fts_all.append(output)
            labels_all.append(labels)

def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # training routine

    extract(train_loader, model, opt, '../scratch/features/{}/train.npz'.format(features_folder))
    extract(val_loader, model, opt, '../scratch/features/{}/val.npz'.format(features_folder))
    



if __name__ == '__main__':
    main()
