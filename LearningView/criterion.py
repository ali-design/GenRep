from __future__ import print_function

import torch
import math
import torch.nn.functional as F


def dv_bound(feat_1, feat_2):
    N, units = feat_1.size()
    u = torch.mm(feat_1, feat_2.t())

    mask = torch.eye(N).to(feat_1.device)
    n_mask = 1 - mask

    E_pos = (u * mask).sum() / mask.sum()

    score_pos = (u * mask).sum() / mask.sum()
    score_neg = (u * n_mask).sum() / n_mask.sum()

    u -= 100 * (1 - n_mask)
    u_max = torch.max(u)
    E_neg = torch.log((n_mask * torch.exp(u - u_max)).sum() + 1e-6) + u_max - math.log(n_mask.sum())
    loss = E_neg - E_pos
    mine = - loss

    return loss, mine, score_pos, score_neg


def infonce_bound(feat_l, feat_ab, t=0.07):

    bsz = feat_l.size(0)

    l_score = torch.mm(feat_l, feat_ab.t()) / t
    l_loss = - torch.diag(F.log_softmax(l_score, dim=1)).mean()
    l_prob = torch.diag(F.softmax(l_score, dim=1)).mean()

    ab_score = torch.mm(feat_ab, feat_l.t()) / t
    ab_loss = - torch.diag(F.log_softmax(ab_score, dim=1)).mean()
    ab_prob = torch.diag(F.softmax(ab_score, dim=1)).mean()

    loss = l_loss + ab_loss
    mine = math.log(bsz) - loss

    return loss, mine, l_prob, ab_prob


def cmc_bound(out_l, out_ab):
    bsz = out_l.shape[0]

    out_l = out_l.squeeze()
    out_ab = out_ab.squeeze()

    l_prob = out_l[:, 0].mean()
    ab_prob = out_ab[:, 0].mean()

    out_l = F.log_softmax(out_l, dim=1)
    out_ab = F.log_softmax(out_ab, dim=1)
    l_loss = - out_l[:, 0].mean()
    ab_loss = -out_ab[:, 0].mean()

    loss = l_loss + ab_loss
    mine = math.log(4096) - loss

    return loss, mine, l_prob, ab_prob

