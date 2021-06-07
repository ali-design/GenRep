from __future__ import print_function

import os
import sys
import argparse
import time
import math
import ipdb
import torch
import torch.backends.cudnn as cudnn

from torchvision import transforms, datasets
import tensorboard_logger as tb_logger
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
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=64,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1.0,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
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
    parser.add_argument('-s', '--cache_folder', type=str,
                        default='/data/vision/torralba/scratch/xavierpuig/ganclr/',
                        help='the saving folder')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    # specifying folders
    parser.add_argument('-d', '--data_folder', type=str,
                        default='/data/vision/torralba/datasets/imagenet_pytorch_new',
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

    opt.tb_path = os.path.join(opt.cache_folder, 'linear_eval_tensorboard')
    ckpt_red = opt.ckpt.replace('/data/vision/', '').replace('/', '_').replace('last.pth', '').replace('torralba_', '').replace('phillipi_', '').replace('_data_scratch_xavierpuig_ganclr', '')
    opt.fn = ckpt_red+'_lr.{}'.format(opt.learning_rate)
    opt.tb_folder = os.path.join(opt.tb_path, ckpt_red+'_lr.{}'.format(opt.learning_rate)).strip()
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

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
    elif opt.dataset == 'biggan' or opt.dataset == 'imagenet100' or opt.dataset == 'imagenet100K' or opt.dataset == 'imagenet':
        train_dataset = datasets.ImageFolder(root=os.path.join(opt.data_folder, 'train'),
                                             transform=train_transform)
        val_dataset = datasets.ImageFolder(root=os.path.join(opt.data_folder, 'val'),
                                           transform=val_transform)
    elif opt.dataset == 'voc2007':
        train_dataset = VOCDetectionDataset(root=opt.data_folder,
                                              year='2007',
                                              image_set='train',
                                              transform=train_transform)

        val_dataset = VOCDetectionDataset(root=opt.data_folder,
                                              year='2007',
                                              image_set='val',
                                              transform=val_transform)

    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
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
        # ipdb.set_trace()
        model.load_state_dict(state_dict,  strict=False)

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    meanAPmetric = mAPMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        import ipdb
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        if opt.dataset == 'voc2007':
            meanAPmetric.add(output.detach(), labels)
            import ipdb
            ipdb.set_trace()

        else:
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
            if opt.dataset == 'voc2007':
                meanAPmetricV = meanAPmetric.value().item()
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'mAP {meanAP:.3f}'.format(
                       epoch, idx + 1, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, meanAP=meanAPmetricV))

            else:
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       epoch, idx + 1, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()
    if opt.dataset == 'voc2007':
        return losses.avg, meanAPmetric.value().item()
    else:
        return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    meanAPmetric = mAPMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            if opt.dataset == 'voc2007':
                meanAPmetric.add(output.detach(), labels)

            else:
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                if opt.dataset == 'voc2007':
                    meanAPmetricV = meanAPmetric.value().item()
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'mAP {meanAP:.3f}'.format(
                           idx, len(val_loader), batch_time=batch_time,
                           loss=losses, meanAP=meanAPmetricV))
                else:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                           idx, len(val_loader), batch_time=batch_time,
                           loss=losses, top1=top1))

    if opt.dataset == 'voc2007':
        print(' * mAP {mAP:.3f}'.format(mAP=meanAPmetric.value().item()))
        return losses.avg, meanAPmetric.value().item()
    else:
        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()
    print(opt)

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    print("Logging in {}".format(opt.tb_folder))
    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))
        logger.log_value('loss_train', loss, epoch)
        logger.log_value('acc_train', acc, epoch)

        # eval for one epoch
        loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        logger.log_value('loss_eval', loss, epoch)
        logger.log_value('acc_eva;', val_acc, epoch)
        if val_acc > best_acc:
            best_acc = val_acc

    print('best accuracy: {:.2f}'.format(best_acc))
    fn = opt.fn
    with open(fn+'.txt', 'w+') as f:
        f.write('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
