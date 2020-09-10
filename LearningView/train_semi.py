"""
semi-supervised view generator training
"""

from __future__ import print_function

import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
import socket

import tensorboard_logger as tb_logger
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader, _MultiProcessingDataLoaderIter

from dataset import STL10MINE, STL10Supervised
from dataset import RGB2RGB, RGB2Lab, RGB2YDbDr

from models.estimator import ResNetMINEV1
from models.generator import RevNetGenerator
from models.classifier import AlexNetClassifier, ResNetClassifier

from memory.NCEAverage import CMCMem

from criterion import dv_bound, infonce_bound, cmc_bound

from util import adjust_learning_rate_mine, AverageMeter, accuracy


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=2, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

    # model definition
    parser.add_argument('--estimator', type=str, default='resnet50v1', choices=['resnet50v1', 'resnet50v2'])
    parser.add_argument('--classifier', type=str, default='resnet18', choices=['alexnet', 'resnet18', 'resnet50'])
    parser.add_argument('--cls_w', type=float, default=1.0, help='classification weight')
    parser.add_argument('--info_w', type=float, default=1.0, help='infomin weight')

    # setting
    parser.add_argument('--color', type=str, default='RGB', choices=['RGB', 'Lab', 'YDbDr'])
    parser.add_argument('--n_block', type=int, default=4, help='num of blocks in generator')
    parser.add_argument('--block_type', type=int, default=2, help='type of block')
    parser.add_argument('--layer_mode', type=str, default='A', choices=['A', 'B', 'C'])
    parser.add_argument('--lr_factor', type=float, default=5, help='multiplication factor for lr of generator')
    parser.add_argument('--exp_id', type=str, default='exp0', help='set experimental id')

    parser.add_argument('--bd', type=str, default='dv', choices=['dv', 'infonce', 'cmc'])

    opt = parser.parse_args()

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.data_folder = '/data/vision/billf/scratch/yltian/datasets'
        opt.model_path = '/data/vision/billf/scratch/yltian/Pedesis/LearningView/STL10_MINE_models'
        opt.tb_path = '/data/vision/billf/scratch/yltian/Pedesis/LearningView/STL10_MINE_tensorboard'
    elif hostname.startswith('instance'):
        opt.data_folder = '/mnt/globalssd/datasets'
        opt.model_path = '/mnt/globalssd/save/LearningView/STL10_MINE_models'
        opt.tb_path = '/mnt/globalssd/save/LearningView/STL10_MINE_tensorboard'
    else:
        raise NotImplementedError('server invalid: {}'.format(hostname))

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.info_w == 1.0:
        opt.model_name = 'semi_{}_{}_{}_btype_{}_bnum_{}_layer_{}_factor_{}'.format(
            opt.classifier, opt.cls_w, opt.color, opt.block_type, opt.n_block, opt.layer_mode, opt.lr_factor)
    elif opt.info_w == 0:
        opt.model_name = 'sup_{}_{}_{}_btype_{}_bnum_{}_layer_{}_factor_{}'.format(
            opt.classifier, opt.cls_w, opt.color, opt.block_type, opt.n_block, opt.layer_mode, opt.lr_factor)
    else:
        raise NotImplementedError('not supporting info_w besides {0, 1.0}')

    opt.model_name = '{}_{}'.format(opt.model_name, opt.bd)

    if opt.exp_id:
        opt.model_name = '{}_{}'.format(opt.model_name, opt.exp_id)

    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    return opt


def main():

    args = parse_option()

    args.gpu = 0

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

    # # ==== set the data loader ====
    # train_transform = transforms.Compose([
    #     transforms.RandomCrop(64),
    #     color_transfer,
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=mean, std=std),
    # ])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.2, 1)),
        transforms.RandomHorizontalFlip(),
        color_transfer,
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    train_dataset = STL10MINE(root=args.data_folder,
                              download=True,
                              split='train+unlabeled',
                              transform=train_transform,
                              two_crop=True)
    print(len(train_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, sampler=None)

    supervised_transform = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.2, 1)),
        transforms.RandomHorizontalFlip(),
        color_transfer,
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    supervised_dataset = STL10Supervised(root=args.data_folder,
                                         transform=supervised_transform)
    print(len(supervised_dataset))
    supervised_loader = torch.utils.data.DataLoader(
        supervised_dataset, batch_size=int(args.batch_size/2), shuffle=True,
        num_workers=int(args.num_workers/2), pin_memory=True, sampler=None)

    supervised_loader = _MultiProcessingDataLoaderIter(supervised_loader)

    # # ==== create model and optimizer ====
    generator = RevNetGenerator(args.n_block, args.block_type, args.layer_mode, args.color == 'RGB')
    # generator = RevNetGenerator(args.n_block, args.block_type, args.layer_mode, True)
    estimator = ResNetMINEV1(name=args.estimator[:-2], ch1=1, ch2=2, split=[1, 2],
                             is_norm=(args.bd == 'infonce' or args.bd == 'cmc'))
    if args.classifier == 'alexnet':
        classifier = AlexNetClassifier()
    else:
        classifier = ResNetClassifier(args.classifier)

    generator = generator.cuda()
    estimator = estimator.cuda()
    classifier = classifier.cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    #Ali: don't need genertor
    # w = torch.tensor(size_z).cuda()

    if args.bd == 'cmc':
        contrast = CMCMem(128, len(train_dataset), 4096, use_softmax=True).cuda()
    else:
        contrast = None

    optimizer_e = torch.optim.Adam(estimator.parameters(),
                                   lr=args.learning_rate,
                                   betas=(args.beta1, args.beta2),
                                   weight_decay=args.weight_decay)
    optimizer_g = torch.optim.Adam(generator.parameters(),
                                   lr=args.learning_rate * args.lr_factor,
                                   betas=(args.beta1, args.beta2),
                                   weight_decay=args.weight_decay)
    optimizer_c = torch.optim.Adam(classifier.parameters(),
                                   lr=args.learning_rate * args.lr_factor,
                                   betas=(args.beta1, args.beta2),
                                   weight_decay=args.weight_decay)
    #Ali
    # optimizer_w = torch.optim.Adam(w,
    # lr=0.0002,
    # betas=(args.beta1, args.beta2),
    # weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # save random initialization
    save_model(generator, args, 0)
    #Ali: save w too

    # routine
    for epoch in range(1, args.epochs + 1):

        adjust_learning_rate_mine(epoch, args.learning_rate, args.lr_decay_epochs, args.lr_decay_rate, optimizer_e)
        adjust_learning_rate_mine(epoch, args.learning_rate * args.lr_factor,
                                  args.lr_decay_epochs, args.lr_decay_rate, optimizer_g)
        adjust_learning_rate_mine(epoch, args.learning_rate * args.lr_factor,
                                  args.lr_decay_epochs, args.lr_decay_rate, optimizer_c)
        #Ali: probably for w too

        print("==> training...")

        time1 = time.time()
        loss, mine, acc_l, acc_r = train(epoch, train_loader, supervised_loader,
                                         generator, estimator, classifier,
                                         optimizer_g, optimizer_e, optimizer_c,
                                         contrast, criterion, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('mine', mine, epoch)
        logger.log_value('acc_l', acc_l, epoch)
        logger.log_value('acc_r', acc_r, epoch)

        # save model
        if epoch % args.save_freq == 0:
            save_model(generator, args, epoch)
            #Ali: save w too

        pass


def save_model(generator, args, epoch):
    print('==> Saving...')
    state = {
        'opt': args,
        'generator': generator.state_dict(),
        'epoch': epoch,
    }
    save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
    torch.save(state, save_file)


def train(epoch, train_loader, super_loader,
          generator, estimator, classifier,
          optimizer_g, optimizer_e, optimizer_c,
          contrast, criterion, opt):
    """
    one epoch training
    """

    generator.train()
    estimator.train()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mines = AverageMeter()
    top1_left_meter = AverageMeter()
    top1_right_meter = AverageMeter()

    end = time.time()
    for idx, (inputs, _, index) in enumerate(train_loader):
        #Ali
        # for idx in range(train_loader_size):
        # out_1, out_2 = generate bsz number of G:
        # z_labels = hot(bsz, num_class)
        # z_vects = np.random.normal(bsz,140)
        # out_1 = G(z_vects, z_labels)
        # out_2 = G(z_vects + w, z_labels) # this is the substitue for the generator: want to learn single variable w instead of a model
        # index = z_labels.cuda()

        data_time.update(time.time() - end)

        bsz = inputs.size(0)
        inputs = inputs.float().cuda()
        index = index.cuda()

        # ===================forward=====================
        inputs_1, inputs_2 = torch.split(inputs, [3, 3], dim=1)
        out_1 = generator(inputs_1)
        out_2 = generator(inputs_2)

        # ===================backward=====================
        # # train estimator
        optimizer_e.zero_grad()
        #Ali:
        feat_1, feat_2 = estimator(out_1.detach(), out_2.detach())
        #Ali: remove this

        out_1_a, out_1_b = torch.split(out_1, [1, 2], dim=1)
        out_2_a, out_2_b = torch.split(out_2, [1, 2], dim=1)
        input_view = torch.cat([out_1_a, out_2_b], dim=1)
        feat_1, feat_2 = estimator(input_view.detach())
        # feat_1, feat_2 = estimator(out_1.detach())

        #Ali need to change to batch-wise for contrastive loss
        # out_1 = contrast(feat_2)
        if opt.bd == 'infonce':
            loss, mine, score_pos, score_neg = infonce_bound(feat_1, feat_2)
        elif opt.bd == 'cmc':
            out_1, out_2 = contrast(feat_1, feat_2, index)
            loss, mine, score_pos, score_neg = cmc_bound(out_1, out_2)
        else:
            loss, mine, score_pos, score_neg = dv_bound(feat_1, feat_2)
        loss.backward()
        optimizer_e.step()

        # # train generator
        optimizer_g.zero_grad()
        optimizer_c.zero_grad()
        #Ali
        # optimizer_w.zero_grad()

        feat_1, feat_2 = estimator(input_view)
        # feat_1, feat_2 = estimator(out_1)
        if opt.bd == 'infonce':
            loss, mine, score_pos, score_neg = infonce_bound(feat_1, feat_2)
        elif opt.bd == 'cmc':
            out_1, out_2 = contrast(feat_1, feat_2, index)
            loss, mine, score_pos, score_neg = cmc_bound(out_1, out_2)
        else:
            loss, mine, score_pos, score_neg = dv_bound(feat_1, feat_2)
        # mine.backward()

        (super_x, super_y) = next(super_loader)
        bsz_cls = super_x.shape[0]
        super_x = super_x.float().cuda()
        super_y = super_y.cuda()
        super_view = generator(super_x)
        logit_1, logit_2 = classifier(super_view)
        cls_loss_1 = criterion(logit_1, super_y)
        cls_loss_2 = criterion(logit_2, super_y)

        acc1_left = accuracy(logit_1, super_y)
        acc1_right = accuracy(logit_2, super_y)

        total_loss = opt.info_w * mine + opt.cls_w * (cls_loss_1 + cls_loss_2)
        total_loss.backward()

        optimizer_g.step()
        optimizer_c.step()

        # ===================meters=====================
        losses.update(loss.item(), bsz)
        mines.update(mine.item(), bsz)
        top1_left_meter.update(acc1_left[0].item(), bsz_cls)
        top1_right_meter.update(acc1_right[0].item(), bsz_cls)

        batch_time.update(time.time() - end)
        end = time.time()

        if idx == 0:
            print('starting score')
            print('pos: ', score_pos.item())
            print('neg: ', score_neg.item())

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('pos:', score_pos.item())
            print('neg:', score_neg.item())
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'mine {mine.val:.3f} ({mine.avg:.3f})\t'
                  'acc1 {acc_l.val:.3f} ({acc_l.avg:.3f})\t'
                  'acc2 {acc_r.val:.3f} ({acc_r.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, mine=mines,
                   acc_l=top1_left_meter, acc_r=top1_right_meter))
            sys.stdout.flush()

    return losses.avg, mines.avg, top1_left_meter.avg, top1_right_meter.avg


if __name__ == '__main__':
    main()
