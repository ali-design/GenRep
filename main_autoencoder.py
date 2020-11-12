## Adapted for biggan based on latent-composite code  
from __future__ import print_function
import argparse
import os
import random
import itertools
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
from torch.nn.functional import cosine_similarity
from tensorboardX import SummaryWriter
import oyaml as yaml
# from utils import zdataset, customnet, pbar, util, masking
# from utils import customnet, pbar, util, masking
from utils import pbar, util, masking
import customenet_biggan as customnet
# import zdataset_biggan
from networks import biggan_networks
import numpy as np
import json

import sys
sys.path.append('resources/PerceptualSimilarity') # TODO: just use lpips import
import models

import pdb;

def train(opt):
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # tensorboard
    writer = SummaryWriter(logdir='training/runs/%s' % os.path.basename(opt.outf))

    device = torch.device("cuda:0" if opt.cuda else "cpu")
    batch_size = int(opt.batchSize)

    # load the generator
    netG = biggan_networks.load_biggan(opt.netG).to(device).eval() #for biggan, it's model_name, e.g. 'biggan-deep-256' 
    util.set_requires_grad(False, netG)
#     print(netG)
    
#     # find output shape
## Ali: to find output shape, we use biggan_networks.truncated_noise_sample_() instead of zdataset_biggan.z_sample_for_model()
#    z = zdataset_biggan.z_sample_for_model(netG, size=1).to(device)
#     # Prepare an input for netG
    truncation = 1.0
    zbs = 1
    z = biggan_networks.truncated_noise_sample_(truncation=truncation, batch_size=zbs).to(device)
    cls_vector = biggan_networks.one_hot_from_int_(77, batch_size=zbs).to(device)
    out_shape = netG(z, cls_vector, truncation).shape
    in_shape = z.shape
    nz = in_shape[1]
    # print(out_shape)
                                   
    # determine encoder input dim
    assert(not (opt.masked and opt.vae_like)), "specify 1 of masked or vae_like"
    has_masked_input = opt.masked or opt.vae_like
    input_dim = 4 if has_masked_input else 3
    modify_input = customnet.modify_layers # adds the to_z layer

    # load the encoder
    depth = int(opt.netE_type.split('-')[-1])
    nz = nz * 2 if opt.vae_like else nz
    netE = customnet.CustomResNet(size=depth, halfsize=out_shape[-1]<=150,
                                  num_classes=nz,
                                  modify_sequence=modify_input,
                                  channels_in=input_dim)
    netE.to(device)
    # print(netE)
#     import pdb;
#     pdb.set_trace()
    last_layer_z = torch.nn.Linear(2048, 128).to(device)
    last_layer_y = torch.nn.Linear(2048, opt.num_imagenet_classes).to(device)

    # losses + optimizers
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    perceptual_loss = models.PerceptualLoss(model='net-lin', net='vgg',
                                            use_gpu=opt.cuda)
    # optimizerE = optim.Adam(netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    start_ep = 0
    ## also loss_y and optim for z and y:
    ce_loss = nn.CrossEntropyLoss()
    # optimizer_z = optim.Adam(last_layer_z.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    # optimizer_y = optim.Adam(last_layer_y.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerE = optim.Adam(list(netE.parameters()) + list(last_layer_z.parameters()) + list(last_layer_y.parameters()), 
                            lr=opt.lr, betas=(opt.beta1, 0.999))
    # z datasets
    min_bs = min(16, batch_size)
    train_loader = training_loader(truncation, batch_size, opt.seed)
    test_zs = biggan_networks.truncated_noise_sample_(truncation=truncation, 
                                                batch_size=min_bs, 
                                                seed=opt.seed).to(device)
    class_name_list = ['robin', 'standard_poodle', 'African_hunting_dog', 'gibbon', 'ambulance', 'boathouse', 'cinema', 'Dutch_oven',
                        'lampshade', 'laptop', 'mixing_bowl', 'pedestal', 'rotisserie', 'slide_rule', 'tripod', 'chocolate_sauce']        
    test_class_vectors = biggan_networks.one_hot_from_names_(class_name_list[0:min_bs], batch_size=min_bs).to(device)
#     with open('./imagenet100_class_index.json', 'rb') as fid:
#         imagenet100_dict = json.load(fid)

    test_idx = [15, 267, 275, 368, 407, 449, 498, 544, 619, 620, 659, 708, 766, 798, 872, 960]
    test_idx = test_idx[0:min_bs]
    # load data from checkpoint
    # come back
    assert(not (opt.netE and opt.finetune)), "specify 1 of netE or finetune"
    if opt.finetune:
        checkpoint = torch.load(opt.finetune)
        sd = checkpoint['state_dict']
        # skip weights with dim mismatch, e.g. if you finetune from
        # an RGB encoder
        if sd['conv1.weight'].shape[1] != input_dim:
            # skip first conv if needed
            print("skipping initial conv")
            sd = {k: v for k, v in sd.items() if k != 'conv1.weight'}
        if sd['fc.bias'].shape[0] != nz:
            # skip fc if needed
            print("skipping fc layers")
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        netE.load_state_dict(sd, strict=False)
    if opt.netE:
        checkpoint = torch.load(opt.netE)
        netE.load_state_dict(checkpoint['state_dict'])
        last_layer_z.load_state_dict(checkpoint['state_dict_last_z'])
        last_layer_y.load_state_dict(checkpoint['state_dict_last_y'])
        optimizerE.load_state_dict(checkpoint['optimizer'])
        start_ep = checkpoint['epoch'] + 1

    epoch_batches = 1600 // batch_size
    for epoch, epoch_loader in enumerate(pbar(
        epoch_grouper(train_loader, epoch_batches),
        total=(opt.niter-start_ep)), start_ep):

        # stopping condition
        if epoch > opt.niter:
            break

        # run a train epoch of epoch_batches batches
        for step, (z_batch,) in enumerate(pbar(
            epoch_loader, total=epoch_batches), 1):
            z_batch = z_batch.to(device)
            netE.zero_grad()
            last_layer_z.zero_grad()
            last_layer_y.zero_grad()
            
            # fake_im = netG(z_batch).detach()
            idx = np.random.choice(opt.num_imagenet_classes, z_batch.shape[0]).tolist()
            class_vector = biggan_networks.one_hot_from_int_(idx, batch_size=z_batch.shape[0]).to(device)
            fake_im = netG(z_batch, class_vector, truncation).detach() 
            
            if has_masked_input:
                ## come back
                hints_fake, mask_fake = masking.mask_upsample(fake_im)
                encoded = netE(torch.cat([hints_fake, mask_fake], dim=1)).view(z_batch.shape)
                if opt.masked:
                    regenerated = netG(encoded, class_vector, truncation)
                elif opt.vae_like:
                    sample = torch.randn_like(encoded[:, nz//2:, :, :])
                    encoded_mean  = encoded[:, nz//2:, :, :]
                    encoded_sigma = torch.exp(encoded[:, :nz//2, :, :])
                    reparam = encoded_mean + encoded_sigma * sample
                    regenerated = netG(reparam, class_vector, truncation)
                    encoded = encoded_mean # just use mean in z loss
            else:
                # standard RGB encoding
                encoded = netE(fake_im)
                z_pred = last_layer_z(encoded)
                y_pred = last_layer_y(encoded)
            
                regenerated = netG(z_pred, class_vector, truncation)

            # compute loss
            loss_y = ce_loss(y_pred, torch.tensor(idx, dtype=torch.int64).to(device))
            loss_z = cor_square_error_loss(z_pred, z_batch)
            loss_mse = mse_loss(regenerated, fake_im)
            loss_perceptual = perceptual_loss.forward(
                regenerated, fake_im).mean()
            loss = (opt.lambda_z * loss_y + opt.lambda_z * loss_z + opt.lambda_mse * loss_mse
                    + opt.lambda_lpips * loss_perceptual)
            loss = (opt.lambda_z * loss_y + opt.lambda_z * loss_z + opt.lambda_mse * loss_mse
                    + opt.lambda_lpips * loss_perceptual)
            # optimize
            loss.backward()
            optimizerE.step()
            # optimizer_z.step()
            # optimizer_y.step()

            # send losses to tensorboard
            if step % 20 == 0:
                total_batches = epoch * epoch_batches + step
                writer.add_scalar('loss/train_y', loss_y, total_batches)
                writer.add_scalar('loss/train_z', loss_z, total_batches)
                writer.add_scalar('loss/train_mse', loss_mse, total_batches)
                writer.add_scalar('loss/train_lpips', loss_perceptual,
                                  total_batches)
                writer.add_scalar('loss/train_total', loss, total_batches)
        
        # import pdb;
        # pdb.set_trace()
        # run the fixed test zs for visualization
        netE.eval()
        last_layer_z.eval()
        last_layer_y.eval()
        with torch.no_grad():
            fake_im = netG(test_zs, test_class_vectors, truncation)
            if has_masked_input:
                ## come back
                hints_fake, mask_fake = masking.mask_upsample(fake_im)
                encoded = netE(torch.cat([hints_fake, mask_fake], dim=1)).view(test_zs.shape)
                if opt.masked:
                    regenerated = netG(encoded, test_class_vectors, truncation)
                elif opt.vae_like:
                    sample = torch.randn_like(encoded[:, nz//2:, :, :])
                    encoded_mean  = encoded[:, nz//2:, :, :]
                    encoded_sigma = torch.exp(encoded[:, :nz//2, :, :])
                    reparam = encoded_mean + encoded_sigma * sample
                    regenerated = netG(reparam, test_class_vectors, truncation)
                    encoded = encoded_mean # just use mean in z loss
            else:
                encoded = netE(fake_im)
                pred_z = last_layer_z(encoded)
                pred_y = last_layer_y(encoded)
                regenerated = netG(pred_z, test_class_vectors, truncation)

            # compute loss
            loss_y = ce_loss(y_pred, torch.tensor(test_idx, dtype=torch.int64).to(device))

            loss_z = cor_square_error_loss(pred_z, test_zs)
            loss_mse = mse_loss(regenerated, fake_im)
            loss_perceptual = perceptual_loss.forward(
                regenerated, fake_im).mean()
            loss = (opt.lambda_z * loss_y + opt.lambda_z * loss_z + opt.lambda_mse * loss_mse
                    + opt.lambda_lpips * loss_perceptual)
            loss = (opt.lambda_z * loss_y + opt.lambda_z * loss_z)
            # send to tensorboard
            writer.add_scalar('loss/test_y', loss_y, epoch)
            writer.add_scalar('loss/test_z', loss_z, epoch)
            writer.add_scalar('loss/test_mse', loss_mse, epoch)
            writer.add_scalar('loss/test_lpips', loss_perceptual,
                              epoch)
            writer.add_scalar('loss/test_total', loss, epoch)
            if has_masked_input:
                grid = vutils.make_grid(
                    torch.cat((fake_im, hints_fake, regenerated)), nrow=8,
                    normalize=True, scale_each=(-1, 1))
            else:
                grid = vutils.make_grid(
                    torch.cat((fake_im, regenerated)), nrow=8,
                    normalize=True, scale_each=(-1, 1))
            writer.add_image('Image', grid, epoch)
        netE.train()

        # do checkpointing
        if epoch % 1000 == 0 or epoch == opt.niter:
            sd = {
                'state_dict': netE.state_dict(),
                'state_dict_last_z': last_layer_z.state_dict(),
                'state_dict_last_y': last_layer_y.state_dict(),
                'optimizer': optimizerE.state_dict(),
                'epoch': epoch
            }
            torch.save(sd, '%s/netE_epoch_%d.pth' % (opt.outf, epoch))


def cor_square_error_loss(x, y, eps=1e-8):
    # Analogous to MSE, but in terms of Pearson's correlation
    return (1.0 - cosine_similarity(x, y, eps=eps)).mean()

def training_loader(truncation, batch_size, global_seed=0):
    '''
    Returns an infinite generator that runs through randomized z
    batches, forever.
    '''
    g_epoch = 1
    while True:
        z_data = biggan_networks.truncated_noise_dataset(truncation=truncation,
                                                         batch_size=10000, 
                                                         seed=g_epoch + global_seed)
        dataloader = torch.utils.data.DataLoader(
                z_data,
                shuffle=False,
                batch_size=batch_size,
                num_workers=10,
                pin_memory=True)
        for batch in dataloader:
            yield batch
        g_epoch += 1

def epoch_grouper(loader, epoch_size, num_epochs=None):
    '''
    To use with the infinite training loader: groups the training data
    batches into epochs of the given size.
    '''
    it = iter(loader)
    epoch = 0
    while True:
        chunk_it = itertools.islice(it, epoch_size)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return
        yield itertools.chain((first_el,), chunk_it)
        epoch += 1
        if num_epochs is not None and epoch >= num_epochs:
            return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_imagenet_classes', type=int, default=1000,
                        help='e.g., 100 or 1000')
    parser.add_argument('--netE_type', type=str, default='resnet-50',
                        help='type of encoder architecture')
    parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
    parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--netG', default='', help="generator to load")
    parser.add_argument('--netE', default='', help="path to netE (to continue training)")
    parser.add_argument('--outf', default='./resnet50_zy_pix', help='folder to output model checkpoints')
    parser.add_argument('--seed', default=0, type=int, help='manual seed')
    parser.add_argument('--lambda_z', default=1.0, type=float, help='loss weighting')
    parser.add_argument('--lambda_mse', default=1.0, type=float, help='loss weighting')
    parser.add_argument('--lambda_lpips', default=1.0, type=float, help='loss weighting')
    parser.add_argument('--finetune', type=str, default='',
                        help="finetune from these weights")
    parser.add_argument('--masked', action='store_true', help="train with masking")
    parser.add_argument('--vae_like', action='store_true',
                        help='train with masking, predict mean and sigma')

    opt = parser.parse_args()
    opt.outf = '{}_{}'.format(opt.outf, opt.num_imagenet_classes)
    print(opt)
    
    assert opt.netE_type == 'resnet-50'

    opt.outf = opt.outf.format(**vars(opt))

    os.makedirs(opt.outf, exist_ok=True)
    # save options
    with open(os.path.join(opt.outf, 'optE.yml'), 'w') as f:
        yaml.dump(vars(opt), f, default_flow_style=False)

    train(opt)