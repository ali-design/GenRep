from __future__ import print_function
import math
from torch.multiprocessing import Pool
import functools

import numpy as np
import pdb
import os
import sys
import argparse
import time
import math

import torchvision.utils as vutils
from torch.utils.data import *
import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torchvision.transforms import functional

from util import TwoCropTransform, AverageMeter, GansetDataset, GansteerDataset
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss
import oyaml as yaml
import pbar as pbar

import io
import IPython.display
import PIL.Image
from pprint import pformat
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_hub as hub
from scipy.stats import truncnorm
import utils_bigbigan as ubigbi
from tqdm import tqdm
import json
import pickle
from tensorflow.python.client import device_lib

from pytorch_pretrained_biggan import (
    BigGAN,
    truncated_noise_sample,
    one_hot_from_int
)


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--encoding_type', type=str, default='contrastive',
                        choices=['contrastive', 'crossentropy', 'autoencoding'])
    parser.add_argument('--print_freq', type=int, default=1,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')

    parser.add_argument('--batch_size_gen', type=int, default=86,
                        help='batch_size')

    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')
    parser.add_argument('--showimg', action='store_true', help='display image in tensorboard')

    parser.add_argument('--resume', default='', type=str, help='whether to resume training')
    parser.add_argument('--niter', type=int, default=256, help='number of iter for online sampling')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.03,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='biggan',
                        choices=['biggan', 'cifar10', 'cifar100', 'imagenet100', 'imagenet100K', 'imagenet'], help='dataset')

    ## Ali: todo: this should be based on opt.encoding type and remove the default (revisit every default) and name of the model for saving
    # method
    parser.add_argument('--numcontrast', type=int, default=20,
                        help='num of workers to use')
    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument('--walk_method', type=str, choices=['none', 'random', 'steer', 'pca'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    # specifying folders
    parser.add_argument('-d', '--data_folder', type=str,
                        default='/data/scratch-oc40/jahanian/ganclr_results/ImageNet100',
                        help='the data folder')
    parser.add_argument('-s', '--cache_folder', type=str,
                        default='/data/scratch-oc40/jahanian/ganclr_results/',
                        help='the saving folder')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = opt.data_folder
    opt.model_path = os.path.join(opt.cache_folder, '{}_online/{}_models'.format(opt.method, opt.dataset))
    opt.tb_path = os.path.join(opt.cache_folder, '{}_online/{}_tensorboard'.format(opt.method, opt.dataset))

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}onlineMP_{}_{}_ncontrast.{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
            format(opt.method, opt.dataset, opt.walk_method, opt.model, opt.numcontrast, opt.learning_rate, 
            opt.weight_decay, opt.batch_size, opt.temp, opt.trial)


    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    opt.model_name = '{}_{}'.format(opt.model_name, os.path.basename(opt.data_folder))
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

    if opt.dataset == 'biggan' or opt.dataset == 'imagenet100' or opt.dataset == 'imagenet100K' or opt.dataset == 'imagenet':
        if opt.method == 'SimCLR':
            opt.img_size = 128
        else:
            opt.img_size = 128
    elif opt.dataset == 'cifar10' or opt.dataset == 'cifar100':
        opt.img_size = 32

    return opt


def worker_func(x):
    print("Wrker", x)
    torch.cuda.set_device(x+1)
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
    opt.mean = mean
    opt.std = std

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
#     gan_model_name='biggan-deep-256'
    gan_model_name = 'https://tfhub.dev/deepmind/bigbigan-resnet50/1'  # ResNet-50
    dataset = OnlineGanDataset(train_transform, gan_model_name, opt=opt)
    dataset.offset_start = 85 * opt.niter 
    all_epochs_sampler = BatchSampler(SequentialSampler(dataset), batch_size=opt.batch_size_gen, drop_last=False)
    data_loader = DataLoader(dataset, batch_size=None, sampler=all_epochs_sampler, 
                             num_workers=opt.num_workers, worker_init_fn=worker_func, multiprocessing_context='spawn')

    return data_loader

def trans_func(single_image, transform):
    pil_image = functional.to_pil_image(single_image[0])
    return transform(pil_image)

class OnlineGanDataset(Dataset):
    def __init__(self, transform, gan_model_name, opt):

        self.transform = transform
        self.gan_model_name = gan_model_name
        self.offset_start = 0
        self.opt = opt
        self.gan_model = None
        self.func = functools.partial(trans_func, transform=transform)
#         with open('./utils/imagenet100_class_index.json', 'rb') as fid: 
        with open('./utils/imagenet_class_index.json', 'rb') as fid: 
            imagenet_class_index_dict = json.load(fid)
        self.idx_imagenet100 = list(map(int, list(imagenet_class_index_dict.keys())))


    def get_available_gpus(self):
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

#     def lazy_init_gan(self):
#         if self.gan_model is None:
#             print("initializing GAN", torch.cuda.device_count(), torch.cuda.current_device())
#             module = hub.Module(self.gan_model_name)  # inference
#             self.gpu_idx_current = torch.cuda.current_device()
#             self.gan_model = ubigbi.BigBiGAN(module)
#             self.gen_ph = self.gan_model.make_generator_ph()
#             # Compute samples G(z) from encoder input z (`gen_ph`).
#             self.gen_samples = self.gan_model.generate(self.gen_ph)
#             ## Create a TensorFlow session and initialize variables
#             init = tf.global_variables_initializer()
#             self.sess = tf.Session()
#             self.sess.run(init)
#             print('lazy_init: get_available_gpus()', self.get_available_gpus())

    def lazy_init_gan(self):
        start_time = time.time()
        if self.gan_model is None:
            print("initializing GAN on {} GPUs, currently on GPU:{}".format(torch.cuda.device_count(),
                                                                            torch.cuda.current_device()))
            self.gpu_idx_current = torch.cuda.current_device()

            with tf.device('/gpu:{}'.format(self.gpu_idx_current)):
                module = hub.Module(self.gan_model_name)  # inference
                self.gan_model = ubigbi.BigBiGAN(module)
                self.gen_ph = self.gan_model.make_generator_ph()
                self.gen_samples = self.gan_model.generate(self.gen_ph)
            
#                 self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
                self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

                init = tf.global_variables_initializer()
                self.sess.run(init)
            print('spent time to init device GPU:{} is {}'.format(torch.cuda.current_device(), time.time() - start_time))
#             print('lazy_init: get_available_gpus()', self.get_available_gpus())

    def apply_im_transform(self, anchor_out):
        anchor_out = 255 * ((anchor_out + 1.0)/2.0)
        anchor_out = anchor_out.detach().cpu().numpy()
        anchor_out = anchor_out.astype(np.uint8)
        anchor_out = np.transpose(anchor_out, [0, 2, 3, 1])
        anchor_out =  np.split(anchor_out, anchor_out.shape[0])
        images_anchor =  map(self.func, anchor_out)
        images_anchor = np.concatenate([x[None,:] for x in images_anchor])
        images_anchor = torch.from_numpy(images_anchor)
        return images_anchor
    
    def __len__(self):
        #  Since we are skipping samples on every iteration
        # On every iteration we skip batch % batch_gen
        # niter / batch * batch mod batch_gen
        batch_size = self.opt.batch_size
        skipped = (self.opt.niter // batch_size) * (self.opt.batch_size_gen % batch_size)
        return (self.opt.niter + skipped) * self.opt.epochs
    
    def __getitem__(self, indices):
        start_time = time.time()
        self.lazy_init_gan()
        truncation = 1.0
        std_scale = 0.2
        batch_size = len(indices)
        
        start_seed = 0
        idx = indices[0] + self.offset_start
        seed = start_seed + 2 * idx
        state = None if seed is None else np.random.RandomState(seed)
        zs = truncation * truncnorm.rvs(-2, 2, size=(batch_size, 120), random_state=state).astype(np.float32)
        feed_dict = {self.gen_ph: zs}
#         print('__getitem__: get_available_gpus()', self.get_available_gpus())

#         with tf.device('/device:GPU:{}'.format(self.gpu_idx_current)):
        anchor_out = self.sess.run(self.gen_samples, feed_dict=feed_dict)
        
        anchor_out = np.transpose(anchor_out, (0, 3, 1, 2))
        anchor_out = torch.from_numpy(anchor_out).cuda()
        
#         anchor_out = anchor_out[:,:,0:112, 0:112]
#         ims = torch.tensor(anchor_out, dtype=torch.float32, device='cuda') #<== might need to manage between TF and those used for encoder
#         images.append(ims)

        
#         zs = torch.from_numpy(zs)
        zsold = zs
        idx_cls = np.random.choice(self.idx_imagenet100, batch_size)
#         class_vector = one_hot_from_int(idx_cls, batch_size=batch_size)
#         class_vector = torch.from_numpy(class_vector)
#         zs = zs.cuda()
#         class_vector = class_vector.cuda()
        #model_biggan.to(f'cuda:{model_biggan.device_ids[0]}')
#         with torch.no_grad():
#             anchor_out = self.gan_model(zs, class_vector, truncation)

        seed = start_seed + 2 * idx + 1
        state = None if seed is None else np.random.RandomState(seed)
        ws = truncation * truncnorm.rvs(-2, 2, size=(batch_size, 120), scale=std_scale, random_state=state).astype(np.float32)
        zs = zs + ws
        feed_dict = {self.gen_ph: zs}
#         with tf.device('/device:GPU:{}'.format(self.gpu_idx_current)):
        anchor_out2 = self.sess.run(self.gen_samples, feed_dict=feed_dict)
            
        anchor_out2 = np.transpose(anchor_out2, (0, 3, 1, 2))
        anchor_out2 = torch.from_numpy(anchor_out2).cuda()
        
        images_anchor = self.apply_im_transform(anchor_out)
        images_anchor2 = self.apply_im_transform(anchor_out2)
#         print('loader spent time', time.time() - start_time)
        return images_anchor, images_anchor2, idx_cls

def set_model(opt):
    if opt.encoding_type == 'contrastive':
        model = SupConResNet(name=opt.model, img_size=int(opt.img_size*0.875))
        criterion = SupConLoss(temperature=opt.temp)

    elif opt.encoding_type == 'crossentropy':
        model = SupCEResNet(name=opt.model, num_classes=opt.n_cls, img_size=int(opt.img_size*0.875))
        criterion = torch.nn.CrossEntropyLoss()

    elif opt.encoding_type == 'autoencoding':
        print("TODO(ali): Implement here")
        raise NotImplementedError

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder, device_ids=[0])
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(data_loader_iterator, model, criterion, optimizer, epoch, opt, start_seed):
    """one epoch training"""
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    top1 = AverageMeter()
    end = time.time()

    print("Start train")
    count = opt.niter // opt.batch_size
    ratio_gen_to_consumer = math.ceil(opt.batch_size / opt.batch_size_gen)
    iter_num = 0
    while iter_num < count:
        idx = iter_num
        iter_num += 1
        data_batch = []
        for it in range(ratio_gen_to_consumer):
            data = next(data_loader_iterator)
            data_batch.append(data)


        data = [torch.cat(tensor_val) for tensor_val in zip(*data_batch)]
#         print(data[0].shape)
        data = [tensor_val[:opt.batch_size] for tensor_val in data]
        if len(data) == 2:
            images = data[0]
            labels = data[1]
        elif len(data) == 3:
            images = data[:2]
            labels = data[2]
        else:
            raise NotImplementedError
        data_time.update(time.time() - end)
        if opt.encoding_type != 'contrastive':
            # We only pick one of images
            images = images[1]
        else:
            ims = images[0]
            anchors = images[1]
            images = torch.cat([images[0].unsqueeze(1), images[1].unsqueeze(1)],
                               dim=1)
            # print('2) images shape', images.shape)

            images = images.view(-1, 3, int(opt.img_size*0.875), int(opt.img_size*0.875)).cuda(non_blocking=True)
            # print('3) images shape', images.shape)

        # labels = labels.cuda(non_blocking=True) <== do we need non_blocking for idx_cls?
        bsz = labels.shape[0]
        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, opt.niter, optimizer)
        # compute loss


        if opt.encoding_type == 'contrastive':
            features = model(images)
            features = features.view(bsz, 2, -1)
            if opt.method == 'SupCon':
                loss = criterion(features, labels)
            elif opt.method == 'SimCLR':
                loss = criterion(features)
            else:
                raise ValueError('contrastive method not supported: {}'.
                                 format(opt.method))
        elif opt.encoding_type == 'crossentropy':
            output = model(images)
            loss = criterion(output, labels)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)
        else:
            raise NotImplementedError


        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        print('time spent per batch:', batch_time.avg)
        # print info
        if (idx + 1) % opt.print_freq == 0:
            if opt.encoding_type == 'crossentropy':
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          epoch, idx + 1, count, batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1))
            else:
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                          epoch, idx + 1, count, batch_time=batch_time,
                       data_time=data_time, loss=losses))
            sys.stdout.flush()
    other_metrics = {}

    if opt.encoding_type == 'crossentropy':
        other_metrics['top1_acc'] = top1.avg

    if opt.showimg:
        other_metrics['image'] = [ims[:8], anchors[:8]]

    return losses.avg, other_metrics


def trans_func(single_image, transform):
    pil_image = functional.to_pil_image(single_image[0])
    return transform(pil_image)

def main():
    opt = parse_option()
    
    print('train config:', opt)

    with open(os.path.join(opt.save_folder, 'train_config.yml'), 'w') as f:
        yaml.dump(vars(opt), f, default_flow_style=False)
    
    # One GPU is used for consuming, the rest for generating
    num_gpus = torch.cuda.device_count() - 1
    if opt.batch_size_gen == -1:
        opt.batch_size_gen = math.ceil(opt.batch_size / (num_gpus))

    # build data loader
    # opt.encoding_type tells us how to get training data
    opt.niter = 130000
    opt.num_workers = min(opt.num_workers, torch.cuda.device_count() - 1)
#     opt.num_workers = 1
    train_loader = set_loader(opt)

    # build model and criterion
    # opt.encoding_type tells us what to put as the head; choices are:
    # contrastive -> mlp or linear
    # crossentropy -> one linear for pred_y
    # autoencoding -> one linear for pred_z and one linear for pred_y
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # Start the data loader

    # training routine
    train_loader_iterator = iter(train_loader)
    init_epoch = 1

    if len(opt.resume) > 0:
        model_ckp = torch.load(opt.resume)
        init_epoch = model_ckp['epoch'] + 1
        model.load_state_dict(model_ckp['model'])
        optimizer.load_state_dict(model_ckp['optimizer'])

    for epoch in range(init_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, other_metrics = train(train_loader_iterator, model, criterion, optimizer, epoch, opt, start_seed=0)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        for metric_name, metric_value in other_metrics.items():
            if metric_name == 'image':
                images = metric_value
                anchors = images[0]
                otherims = images[1]
                bs = anchors.shape[0]
                grid_images = vutils.make_grid(
                        torch.cat((anchors, otherims)), nrow=bs)
                grid_images *= np.array(opt.std)[:, None, None]
                grid_images += np.array(opt.mean)[:, None, None]
                grid_images = (255*grid_images.cpu().numpy()).astype(np.uint8)
                grid_images = grid_images[None, :].transpose(0,2,3,1)
                logger.log_images(metric_name, grid_images, epoch)
            else:
                logger.log_value(metric_name, metric_value, epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
