"""
Linear Classifier for SAE
"""
from __future__ import print_function


import os
import sys
import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import argparse
import socket

import tensorboard_logger as tb_logger
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from util import adjust_learning_rate, accuracy, AverageMeter
from dataset import RGB2Lab, RGB2RGB, RGB2YDbDr, RandomTranslateWithReflect
from models.generator import RevNetGenerator
from models.tinyalexnet import alexnet, LinearClassifier, alexnet_cifar


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for Classifier')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    # For Adam, layer 5: learning rate 0.0002; layer 7: learning rate 0.001
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='where to decay lr, can be a list')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--layer', type=int, default=7, help='which layer to do linear probe')

    parser.add_argument('--model_prefix', type=str,
                        default='/data/vision/billf/scratch/yltian/Pedesis/LearningView/STL10_models/')
    parser.add_argument('--model_path', type=str, default='', help='specify model name')

    # setting
    parser.add_argument('--view', type=str, default='raw', choices=['raw', 'learn'])
    parser.add_argument('--color', type=str, default='RGB', choices=['RGB', 'Lab', 'YDbDr'])
    parser.add_argument('--gen_block', type=int, default=4, help='the num of blocks in generator')

    opt = parser.parse_args()

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.data_folder = '/data/vision/billf/scratch/yltian/datasets'
        opt.tb_path = '/data/vision/billf/scratch/yltian/Pedesis/LearningView/STL10_linear_tensorboard'
        opt.model_prefix = '/data/vision/billf/scratch/yltian/Pedesis/LearningView/STL10_models/'
    elif hostname.startswith('instance'):
        opt.data_folder = '/mnt/globalssd/datasets'
        opt.tb_path = '/mnt/globalssd/save/LearningView/STL10_linear_tensorboard'
        opt.model_prefix = '/mnt/globalssd/save/LearningView/STL10_models/'
    else:
        raise NotImplementedError('server invalid: {}'.format(hostname))

    opt.model_path = os.path.join(opt.model_prefix, opt.model_path)

    for m in ['A', 'B', 'C']:
        mode = 'layer_{}'.format(m)
        if mode in opt.model_path:
            opt.layer_mode = m

    for c in ['RGB', 'Lab', 'YDbDr']:
        if c in opt.model_path:
            opt.color = c
            break

    for t in range(1, 20):
        if 'btype_{}_'.format(t) in opt.model_path:
            opt.block_type = t
            break

    for n in range(1, 20):
        if 'bnum_{}_'.format(n) in opt.model_path:
            opt.gen_block = n
            break

    opt.dataset = 'STL10'
    for d in ['CIFAR100', 'CIFAR10']:
        if '{}_'.format(d) in opt.model_path:
            opt.dataset = d
            break

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = opt.model_path.split('/')[-2]

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    return opt


def main():

    args = parse_option()

    args.gpu = 0

    # # data and loader
    if args.color in ['RGB']:
        mean = (0.43, 0.42, 0.39)
        std = (0.27, 0.26, 0.27)
        color_transfer = RGB2RGB()
    elif args.color in ['Lab']:
        mean = (50.0, 6.025, -6.6895)
        std = (50.0, 92.208, 101.1675)
        color_transfer = RGB2Lab()
    elif args.color in ['YDbDr']:
        mean = (0.434, -0.049, -0.028)
        std = (0.500, 1.169, 1.321)
        color_transfer = RGB2YDbDr()
    else:
        raise NotImplementedError('color not supported: {}'.format(args.color))

    normalize = transforms.Normalize(mean=mean, std=std)

    if args.dataset == 'STL10':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.3, 1.0), ratio=(0.7, 1.4)),
            transforms.RandomHorizontalFlip(),
            color_transfer,
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize(70),
            transforms.CenterCrop(64),
            color_transfer,
            transforms.ToTensor(),
            normalize
        ])
    elif 'CIFAR' in args.dataset:
        train_transform = transforms.Compose([
            transforms.RandomApply([RandomTranslateWithReflect(4)], p=0.8),
            transforms.RandomHorizontalFlip(),
            color_transfer,
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            color_transfer,
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise NotImplementedError('dataset not supported: {}'.format(args.dataset))

    if args.dataset == 'STL10':
        train_set = datasets.STL10(root=args.data_folder,
                                   download=True,
                                   split='train',
                                   transform=train_transform)
        test_set = datasets.STL10(root=args.data_folder,
                                  download=True,
                                  split='test',
                                  transform=test_transform)
    elif args.dataset == 'CIFAR10':
        train_set = datasets.CIFAR10(root=args.data_folder,
                                     download=True,
                                     train=True,
                                     transform=train_transform)
        test_set = datasets.CIFAR10(root=args.data_folder,
                                    download=True,
                                    train=False,
                                    transform=test_transform)
    elif args.dataset == 'CIFAR100':
        train_set = datasets.CIFAR100(root=args.data_folder,
                                      download=True,
                                      train=True,
                                      transform=train_transform)
        test_set = datasets.CIFAR100(root=args.data_folder,
                                     download=True,
                                     train=False,
                                     transform=test_transform)
    else:
        raise NotImplementedError('dataset not supported: {}'.format(args.dataset))

    train_batch_size = 256 if 'CIFAR' in args.dataset else 128
    test_batch_size = 128 if 'CIFAR' in args.dataset else 64
    train_loader = DataLoader(train_set,
                              batch_size=train_batch_size,
                              shuffle=True,
                              num_workers=8)
    test_loader = DataLoader(test_set,
                             batch_size=test_batch_size,
                             shuffle=False,
                             num_workers=8)
    if args.dataset == 'STL10' or args.dataset == 'CIFAR10':
        n_class = 10
    elif args.dataset == 'CIFAR100':
        n_class = 100
    else:
        raise NotImplementedError('dataset not supported: {}'.format(args.dataset))

    # checkpoint
    ckpt = torch.load(args.model_path)
    state_dict = ckpt['model']
    has_module = False
    for k, v in state_dict.items():
        if k.startswith('module'):
            has_module = True
    if has_module:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        state_dict = new_state_dict
    ckpt['model'] = state_dict
    # encoder
    if 'CIFAR' in args.dataset:
        model = alexnet_cifar()
    else:
        model = alexnet()
    model.load_state_dict(ckpt['model'])
    # generator
    if args.view == 'raw':
        generator = torch.nn.Sequential()
    elif args.view == 'learn':
        generator = RevNetGenerator(args.gen_block, args.block_type, args.layer_mode, args.color == 'RGB')
        # generator = RevNetGenerator(args.gen_block, args.block_type, args.layer_mode, True)
        generator.load_state_dict(ckpt['generator'])
    else:
        raise NotImplementedError('view not implemented: {}'.format(args.view))

    model = model.cuda().eval()
    generator = generator.cuda().eval()

    if 'CIFAR' in args.dataset:
        test_data = torch.zeros(2, 3, 32, 32).cuda()
    else:
        test_data = torch.zeros(2, 3, 64, 64).cuda()
    test_feat = model.compute_feat(test_data, args.layer)
    print(test_feat.shape)
    feat_dim = test_feat.view(2, -1).size(1)
    print('feature dimension: ', feat_dim)

    classifier = LinearClassifier(feat_dim, n_class)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(),
                           lr=args.learning_rate,
                           betas=(args.beta1, args.beta2),
                           weight_decay=args.weight_decay)

    classifier = classifier.cuda()
    criterion = criterion.cuda()
    cudnn.benchmark = True

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    for epoch in range(1, args.epochs + 1):

        # train the model
        print('============training============')
        generator.eval()
        model.eval()
        classifier.train()

        top1 = AverageMeter()
        adjust_learning_rate(epoch, args, optimizer)
        end = time.time()
        for idx, data in enumerate(train_loader):
            data_time = time.time() - end

            img, label = data[0], data[1]
            img = img.float()
            bsz = img.size(0)
            img = img.cuda()
            label = label.cuda()

            # forward
            with torch.no_grad():
                img = generator(img)
                feat = model.compute_feat(img, args.layer)
                feat = feat.detach()
                feat = feat.view(bsz, -1)
            pred = classifier(feat)
            loss = criterion(pred, label)

            acc1 = accuracy(pred, label, topk=(1,))
            top1.update(acc1[0].item(), bsz)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time = time.time() - end
            end = time.time()

            if idx % args.print_freq == 0:
                print('epoch {} batch {}/{}, data_time{:.4f}, batch_time{:.4f}, loss:{:.5f}, '
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'
                      .format(epoch, idx+1, len(train_loader), data_time,
                              batch_time, loss.item(), top1=top1))
                sys.stdout.flush()

        print('average: ', top1.avg)
        logger.log_value('train_acc', top1.avg, epoch)
        del top1

    # test the model
    print('============testing============')
    model.eval()
    classifier.eval()

    top1 = AverageMeter()
    for idx, data in enumerate(test_loader):

        img, label = data[0], data[1]
        img = img.float()
        bsz = img.size(0)
        img = img.cuda()
        label = label.cuda()

        # forward
        with torch.no_grad():
            img = generator(img)
            feat = model.compute_feat(img, args.layer)
            feat = feat.detach()
            feat = feat.view(bsz, -1)
            pred = classifier(feat)

        acc1 = accuracy(pred, label, topk=(1,))
        top1.update(acc1[0].item(), bsz)

    print('testing: average: ', top1.avg)
    logger.log_value('test_acc', top1.avg, 1)
    logger.log_value('test_acc', top1.avg, 2)


if __name__ == '__main__':
    main()
