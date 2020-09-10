import torch
from torch.autograd import Function
from torch import nn
from .alias_multinomial import AliasMethod
import math


# =========================
# InsDis and MoCo
# =========================

class MemoryInsDis(nn.Module):
    """Memory bank with instance discrimination"""
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=False):
        super(MemoryInsDis, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

        # # pool
        # import numpy as np
        # self.pool = torch.from_numpy(np.random.choice(outputSize, K, replace=False)).cuda()
        # self.cur = 0

    def forward(self, x, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z = self.params[2].item()
        momentum = self.params[3].item()

        batchSize = x.size(0)
        outputSize = self.memory.size(0)
        inputSize = self.memory.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)

            # neg_idx = self.pool.view(1, -1).expand(batchSize, -1)
            # pos_idx = y.view(-1, 1)
            # idx = torch.cat([pos_idx, neg_idx], dim=1)

        # sample
        weight = torch.index_select(self.memory, 0, idx.view(-1))
        weight = weight.view(batchSize, K + 1, inputSize)
        out = torch.bmm(weight, x.view(batchSize, inputSize, 1))

        if self.use_softmax:
            out = torch.div(out, T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, T))
            if Z < 0:
                self.params[2] = out.mean() * outputSize
                Z = self.params[2].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            weight_pos = torch.index_select(self.memory, 0, y.view(-1))
            weight_pos.mul_(momentum)
            weight_pos.add_(torch.mul(x, 1 - momentum))
            weight_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(weight_norm)
            self.memory.index_copy_(0, y, updated_weight)

        # # update pool
        # with torch.no_grad():
        #     out_ids = torch.arange(batchSize).cuda()
        #     out_ids += self.cur
        #     out_ids = torch.fmod(out_ids, self.K)
        #     out_ids = out_ids.long()
        #     self.pool.index_copy_(0, out_ids, y)
        #     self.cur = (self.cur + batchSize) % self.K

        return out


class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""
    def __init__(self, inputSize, outputSize, K, T=0.07, use_softmax=False):
        super(MemoryMoCo, self).__init__()
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.queueSize = K
        self.T = T
        self.index = 0
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))
        print('using queue shape: ({},{})'.format(self.queueSize, inputSize))

    def forward(self, q, k):
        batchSize = q.shape[0]
        k = k.detach()

        Z = self.params[0].item()

        # pos logit
        l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)
        # neg logit
        queue = self.memory.clone()
        l_neg = torch.mm(queue.detach(), q.transpose(1, 0))
        l_neg = l_neg.transpose(0, 1)

        out = torch.cat((l_pos, l_neg), dim=1)

        if self.use_softmax:
            out = torch.div(out, self.T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, self.T))
            if Z < 0:
                self.params[0] = out.mean() * self.outputSize
                Z = self.params[0].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, k)
            self.index = (self.index + batchSize) % self.queueSize

        return out


class CMCMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""
    def __init__(self, inputSize, outputSize, K, T=0.07, use_softmax=False):
        super(CMCMoCo, self).__init__()
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.queueSize = K
        self.T = T
        self.index = 0
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([-1, -1]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_x', torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_y', torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))
        print('using queue shape: ({},{})'.format(self.queueSize, inputSize))

    def forward(self, q_x, k_x, q_y, k_y):
        batchSize = q_x.shape[0]
        k_x = k_x.detach()
        k_y = k_y.detach()

        Z_x = self.params[0].item()
        Z_y = self.params[1].item()

        # pos logit
        l_pos_x = torch.bmm(q_x.view(batchSize, 1, -1), k_y.view(batchSize, -1, 1))
        l_pos_x = l_pos_x.view(batchSize, 1)
        l_pos_y = torch.bmm(q_y.view(batchSize, 1, -1), k_x.view(batchSize, -1, 1))
        l_pos_y = l_pos_y.view(batchSize, 1)

        # neg logit
        queue_y = self.memory_y.clone()
        l_neg_x = torch.mm(queue_y.detach(), q_x.transpose(1, 0))
        l_neg_x = l_neg_x.transpose(0, 1)
        queue_x = self.memory_x.clone()
        l_neg_y = torch.mm(queue_x.detach(), q_y.transpose(1, 0))
        l_neg_y = l_neg_y.transpose(0, 1)

        out_x = torch.cat((l_pos_x, l_neg_x), dim=1)
        out_y = torch.cat((l_pos_y, l_neg_y), dim=1)

        if self.use_softmax:
            out_x = torch.div(out_x, self.T)
            out_x = out_x.squeeze().contiguous()
            out_y = torch.div(out_y, self.T)
            out_y = out_y.squeeze().contiguous()
        else:
            out_x = torch.exp(torch.div(out_x, self.T))
            out_y = torch.exp(torch.div(out_y, self.T))
            if Z_x < 0:
                self.params[0] = out_x.mean() * self.outputSize
                Z_x = self.params[0].clone().detach().item()
                print("normalization constant Z_x is set to {:.1f}".format(Z_x))
            if Z_y < 0:
                self.params[1] = out_y.mean() * self.outputSize
                Z_y = self.params[1].clone().detach().item()
                print("normalization constant Z_y is set to {:.1f}".format(Z_y))
            # compute the out
            out_x = torch.div(out_x, Z_x).squeeze().contiguous()
            out_y = torch.div(out_y, Z_y).squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory_x.index_copy_(0, out_ids, k_x)
            self.memory_y.index_copy_(0, out_ids, k_y)
            self.index = (self.index + batchSize) % self.queueSize

        return out_x, out_y


class CMCMem(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=False):
        super(CMCMem, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.T = T
        self.m = momentum
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([-1, -1]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_l', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_ab', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, l, ab, y, idx=None):
        Z_l = self.params[0].item()
        Z_ab = self.params[1].item()

        batchSize = l.size(0)
        outputSize = self.memory_l.size(0)
        inputSize = self.memory_l.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        # sample
        weight_l = torch.index_select(self.memory_l, 0, idx.view(-1)).detach()
        weight_l = weight_l.view(batchSize, self.K + 1, inputSize)
        out_ab = torch.bmm(weight_l, ab.view(batchSize, inputSize, 1))
        # sample
        weight_ab = torch.index_select(self.memory_ab, 0, idx.view(-1)).detach()
        weight_ab = weight_ab.view(batchSize, self.K + 1, inputSize)
        out_l = torch.bmm(weight_ab, l.view(batchSize, inputSize, 1))

        if self.use_softmax:
            out_l = torch.div(out_l, self.T)
            out_l = out_l.squeeze().contiguous()
            out_ab = torch.div(out_ab, self.T)
            out_ab = out_ab.squeeze().contiguous()
        else:
            out_l = torch.exp(torch.div(out_l, self.T))
            out_ab = torch.exp(torch.div(out_ab, self.T))
            # set Z_0 if haven't been set yet,
            # Z_0 is used as a constant approximation of Z, to scale the probs
            if Z_l < 0:
                self.params[0] = out_l.mean() * outputSize
                Z_l = self.params[0].clone().detach().item()
                print("normalization constant Z_l is set to {:.1f}".format(Z_l))
            if Z_ab < 0:
                self.params[1] = out_ab.mean() * outputSize
                Z_ab = self.params[1].clone().detach().item()
                print("normalization constant Z_ab is set to {:.1f}".format(Z_ab))
            # compute out_l, out_ab
            out_l = torch.div(out_l, Z_l).squeeze().contiguous()
            out_ab = torch.div(out_ab, Z_ab).squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(self.m)
            l_pos.add_(torch.mul(l, 1 - self.m))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)
            self.memory_l.index_copy_(0, y, updated_l)

            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
            ab_pos.mul_(self.m)
            ab_pos.add_(torch.mul(ab, 1 - self.m))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
            self.memory_ab.index_copy_(0, y, updated_ab)

        return out_l, out_ab
