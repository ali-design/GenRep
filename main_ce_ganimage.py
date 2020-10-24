from __future__ import print_function

import os
import sys
import argparse
import time
import math
import ipdb

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import AverageMeter
from util import TwoCropTransform, AverageMeter, GansetDataset, GansteerDataset
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from networks.resnet_big import SupCEResNet

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
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.2,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
            choices=['cifar10', 'cifar100', 'imagenet100', 'biggan'], help='dataset')

    # other setting
    parser.add_argument('--ganrndwalk', action='store_true', help='augment by walking in the gan latent space')
    parser.add_argument('--walktype', type=str, default="gaussian", help='how should we random walk on latent space',
                        choices=['gaussian', 'uniform'])
    parser.add_argument('--zstd', type=float, default=1.0, help='augment std away from z')
    parser.add_argument('--ganzoomwalk', action='store_true', help='augment by steerability walk for rot3d')

    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('-d', '--data_folder', type=str,
                        default='/data/scratch-oc40/jahanian/ganclr_results/ImageNet100',
                        help='the data folder')

    parser.add_argument('-s', '--cache_folder', type=str,
                        default='/data/scratch-oc40/jahanian/ganclr_results/',
                        help='the saving folder')


    opt = parser.parse_args()

    # set the path according to the environment
    # opt.data_folder = './datasets/'
    opt.model_path = '{}/SupCon/{}_models'.format(opt.cache_folder, opt.dataset)
    opt.tb_path = '{}/SupCon/{}_tensorboard'.format(opt.cache_folder, opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.method = 'SupCE'
    if opt.ganrndwalk:
        if opt.walktype == 'gaussian':
            walk_type = 'zstd_{}'.format(opt.zstd)
        elif opt.walktype == 'uniform':
            walk_type = 'uniform_{}'.format(opt.uniformb)

        opt.model_name = '{}_{}_ganrndwalk_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'.\
            format(opt.method, opt.dataset, opt.walktype, opt.model, opt.learning_rate, 
                opt.weight_decay, opt.batch_size, opt.trial)
    elif opt.ganzoomwalk:
        opt.model_name = '{}_{}_ganzoomwalk_{}_lr_{}_decay_{}_bsz_{}_trial_{}'.\
            format(opt.method, opt.dataset, opt.model, opt.learning_rate, 
                opt.weight_decay, opt.batch_size, opt.trial)

    else: 
        opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'.\
            format(opt.method, opt.dataset, opt.model, opt.learning_rate,
                    opt.weight_decay, opt.batch_size, opt.trial)

    opt.model_name = '{}_{}'.format(opt.model_name, os.path.basename(opt.data_folder))

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
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

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'imagenet100' or opt.dataset == 'biggan':
        opt.n_cls = 100
        opt.img_size = 128
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
    elif opt.dataset == 'biggan' or opt.dataset == 'imagenet100' or opt.dataset == 'imagenet100K' or opt.dataset == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    #train_transform = transforms.Compose([
    #    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
    #    transforms.RandomHorizontalFlip(),
    #    transforms.ToTensor(),
    #    normalize,
    #])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=int(opt.img_size*0.875), scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.CenterCrop(int(opt.img_size*0.875)),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)
    elif opt.dataset == 'imagenet100' or opt.dataset == 'imagenet100K' or opt.dataset == 'imagenet':
        train_dataset = datasets.ImageFolder(root=os.path.join(opt.data_folder, 'train'),
                                    transform=train_transform)
        val_dataset = datasets.ImageFolder(root=os.path.join(opt.data_folder, 'val'),
                                    transform=val_transform)
    elif opt.dataset == 'biggan':
        if opt.ganrndwalk:
            train_dataset = GansetDataset(root_dir=os.path.join(opt.data_folder, 'train'), 
                                        neighbor_std=opt.zstd, transform=train_transform)        

            try:
                val_dataset = GansetDataset(root_dir=os.path.join(opt.data_folder, 'val'), 
                                            neighbor_std=opt.zstd, transform=train_transform)        
            except:
                val_dataset = None

        elif opt.ganzoomwalk:
            train_dataset = GansteerDataset(root_dir=os.path.join(opt.data_folder, 'train'), 
                                        transform=train_transform)        
            try:
                val_dataset = GansteerDataset(root_dir=os.path.join(opt.data_folder, 'val'), 
                                            transform=val_transform)        
            except:
                val_dataset = None
        else:
            train_dataset = datasets.ImageFolder(root=os.path.join(opt.data_folder, 'train'),
                                        transform=TwoCropTransform(train_transform))
            try:
                val_dataset = datasets.ImageFolder(root=os.path.join(opt.data_folder, 'val'),
                                                     transform=TwoCropTransform(val_transform))
            except:
                val_dataset = None
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    if val_dataset is None:
        val_loader = None
    else:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=32, shuffle=False,
            num_workers=8, pin_memory=True)

    return train_loader, val_loader


def set_model(opt):
    model = SupCEResNet(name=opt.model, num_classes=opt.n_cls, img_size=int(opt.img_size*0.875))
    criterion = torch.nn.CrossEntropyLoss()

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, content in enumerate(train_loader):
        if len(content) == 2:
            images = content[0]
            labels = content[1]
        elif len(content) == 3:
            images = [content[0], content[1]]
            labels = content[2]
        data_time.update(time.time() - end)
        images = images[1]
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = model(images)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images[1]
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(images)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('train_loss', loss, epoch)
        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # evaluation
        if val_loader is not None:
            loss, val_acc = validate(val_loader, model, criterion, opt)
            logger.log_value('val_loss', loss, epoch)
            logger.log_value('val_acc', val_acc, epoch)
        else:
            val_acc = 0.

        if val_acc > best_acc:
            best_acc = val_acc

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
