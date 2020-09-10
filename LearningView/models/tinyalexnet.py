from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn


class alexnet(nn.Module):
    def __init__(self, feat_dim=128):
        super(alexnet, self).__init__()

        self.l_to_ab = alexnet_half(in_channel=1, feat_dim=feat_dim)
        self.ab_to_l = alexnet_half(in_channel=2, feat_dim=feat_dim)

    def forward(self, x):
        l, ab = torch.split(x, [1, 2], dim=1)
        feat_l = self.l_to_ab(l)
        feat_ab = self.ab_to_l(ab)
        return feat_l, feat_ab

    def compute_feat(self, x, layer):
        l, ab = torch.split(x, [1, 2], dim=1)
        feat_l = self.l_to_ab.compute_feat(l, layer)
        feat_ab = self.ab_to_l.compute_feat(ab, layer)
        feat = torch.cat((feat_l, feat_ab), dim=1)
        return feat


class alexnetExtended(nn.Module):
    def __init__(self, w=0.5, branch=1, feat_dim=128, split=None):
        super(alexnetExtended, self).__init__()

        self.w = w
        self.branch = branch,
        self.feat_dim = feat_dim
        if split is None:
            self.split = [1, 2]
        else:
            self.split = split

        if len(self.split) == 2:
            self.l_to_ab = alexnet_half(self.split[0], w, branch=1, feat_dim=feat_dim)
            self.ab_to_l = alexnet_half(self.split[1], w, branch=1, feat_dim=feat_dim)
        else:
            encoders = {}
            for i in range(len(self.split)):
                encoders['encoder_{}'.format(i)] = alexnet_half(self.split[i],
                                                                w=w,
                                                                branch=branch,
                                                                feat_dim=feat_dim)
            self.encoders = nn.ModuleDict(encoders)

    def forward(self, x):
        data = torch.split(x, self.split, dim=1)
        if len(self.split) == 2:
            feat_l = self.l_to_ab(data[0])
            feat_ab = self.ab_to_l(data[1])
            return feat_l, feat_ab
        else:
            output = []
            for i in range(len(self.split)):
                output.append(self.encoders['encoder_{}'.format(i)](data[i]))
            return output

    def compute_feat(self, x, layer, view_id=0):
        data = torch.split(x, self.split, dim=1)
        if len(self.split) == 2:
            feat_l = self.l_to_ab.compute_feat(data[0], layer)
            feat_ab = self.ab_to_l.compute_feat(data[1], layer)
            if view_id == 0:
                feat = torch.cat((feat_l, feat_ab), dim=1)
                return feat
            elif view_id == 1:
                return feat_l
            elif view_id == 2:
                return feat_ab
            else:
                raise NotImplementedError('view not supported in computing features: {}'.format(view_id))
        else:
            output = []
            for i in range(len(self.split)):
                output.append(self.encoders['encoder_{}'.format(i)].compute_feat(data[i], layer))
            if view_id == 0:
                output = torch.cat(output, dim=1)
                return output
            elif view_id <= len(output):
                return output[view_id - 1]
            else:
                raise NotImplementedError('view not supported in computing features: {}'.format(view_id))



class alexnet_half(nn.Module):
    """alexnet feature network"""
    def __init__(self, in_channel=1, w=0.5, branch=1, feat_dim=128):
        super(alexnet_half, self).__init__()

        self.w = w
        self.branch = branch
        self.feat_dim = feat_dim

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channel, int(96 * w), 3, 1, 1, bias=False),
            nn.BatchNorm2d(int(96 * w)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(int(96 * w), int(192 * w), 3, 1, 1, bias=False),
            nn.BatchNorm2d(int(192 * w)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(int(192 * w), int(384 * w), 3, 1, 1, bias=False),
            nn.BatchNorm2d(int(384 * w)),
            nn.ReLU(inplace=True),
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(int(384 * w), int(384 * w), 3, 1, 1, bias=False),
            nn.BatchNorm2d(int(384 * w)),
            nn.ReLU(inplace=True),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(int(384 * w), int(192 * w), 3, 1, 1, bias=False),
            nn.BatchNorm2d(int(192 * w)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )

        self.fc6 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(int(192 * 7 * 7 * w), int(4096 * w)),
            nn.BatchNorm1d(int(4096 * w)),
            nn.ReLU(inplace=True),
        )
        self.fc7 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(int(4096 * w), int(4096 * w)),
            nn.BatchNorm1d(int(4096 * w)),
            nn.ReLU(inplace=True),
        )
        self.fc8 = nn.Sequential(
            nn.Linear(int(4096 * w), feat_dim)
        )
        if self.branch >= 2:
            self.fc8_2 = nn.Sequential(nn.Linear(int(4096 * w), feat_dim))
        if self.branch >= 3:
            self.fc8_3 = nn.Sequential(nn.Linear(int(4096 * w), feat_dim))
        if self.branch >= 4:
            raise NotImplementedError('# of branches not supported yet: {}'.format(self.branch))
        self.l2norm = Normalize(2)

    def forward(self, x):
        x1 = self.conv_block_1(x)
        x2 = self.conv_block_2(x1)
        x3 = self.conv_block_3(x2)
        x4 = self.conv_block_4(x3)
        x5 = self.conv_block_5(x4)
        x5 = x5.view(x.size(0), -1)
        x6 = self.fc6(x5)
        x7 = self.fc7(x6)
        x8 = self.fc8(x7)
        x8 = self.l2norm(x8)
        if self.branch == 1:
            return x8
        if self.branch == 2:
            x8_2 = self.l2norm(self.fc8_2(x7))
            return x8, x8_2
        if self.branch == 3:
            x8_2 = self.l2norm(self.fc8_2(x7))
            x8_3 = self.l2norm(self.fc8_3(x7))
            return x8, x8_2, x8_3

    def compute_feat(self, x, layer):
        if layer <= 0:
            return x
        x = self.conv_block_1(x)
        if layer == 1:
            return x
        x = self.conv_block_2(x)
        if layer == 2:
            return x
        x = self.conv_block_3(x)
        if layer == 3:
            return x
        x = self.conv_block_4(x)
        if layer == 4:
            return x
        x = self.conv_block_5(x)
        if layer == 5:
            return x
        x = x.view(x.shape[0], -1)
        x = self.fc6(x)
        if layer == 6:
            return x
        x = self.fc7(x)
        if layer == 7:
            return x
        x8 = self.l2norm(self.fc8(x))
        if self.branch == 1:
            return x8
        if self.branch == 2:
            x8_2 = self.l2norm(self.fc8_2(x))
            return x8, x8_2
        if self.branch == 3:
            x8_2 = self.l2norm(self.fc8_2(x))
            x8_3 = self.l2norm(self.fc8_3(x))
            return x8, x8_2, x8_3


class Normalize(nn.Module):
    """Normalizing the feature"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


#########################################
# ===== alexnet mix ===== #
#########################################
class alexnet_mix(nn.Module):
    def __init__(self, branch=2, feat_dim=128):
        super(alexnet_mix, self).__init__()

        self.branch = branch
        self.l_to_ab = alexnet_half_mix(in_channel=1, branch=branch, feat_dim=feat_dim)
        self.ab_to_l = alexnet_half_mix(in_channel=2, branch=branch, feat_dim=feat_dim)

    def forward(self, x):
        l, ab = torch.split(x, [1, 2], dim=1)

        if self.branch == 2:
            feat_l, self_l = self.l_to_ab(l)
            feat_ab, self_ab = self.ab_to_l(ab)
            return feat_l, feat_ab, self_l, self_ab
        else:
            feat_l, mu_l, logvar_l = self.l_to_ab(l)
            feat_ab, mu_ab, logvar_ab = self.ab_to_l(ab)
            return feat_l, mu_l, logvar_l, feat_ab, mu_ab, logvar_ab

    def compute_feat(self, x, layer):
        l, ab = torch.split(x, [1, 2], dim=1)
        feat_l = self.l_to_ab.compute_feat(l, layer)
        feat_ab = self.ab_to_l.compute_feat(ab, layer)
        feat = torch.cat((feat_l, feat_ab), dim=1)
        return feat

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class alexnet_half_mix(nn.Module):
    """alexnet feature network"""
    def __init__(self, in_channel=1, branch=2, feat_dim=128):
        super(alexnet_half_mix, self).__init__()

        assert branch == 2 or branch == 3, 'branch is not 2 or 3, is {}'.format(branch)
        self.branch = branch

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channel, 96 // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(96 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(96 // 2, 192 // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(192 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(192 // 2, 384 // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(384 // 2),
            nn.ReLU(inplace=True),
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(384 // 2, 384 // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(384 // 2),
            nn.ReLU(inplace=True),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(384 // 2, 192 // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(192 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )

        self.fc6 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(192 * 7 * 7 // 2, 4096 // 2),
            nn.BatchNorm1d(4096 // 2),
            nn.ReLU(inplace=True),
        )
        self.fc7 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(4096 // 2, 4096 // 2),
            nn.BatchNorm1d(4096 // 2),
            nn.ReLU(inplace=True),
        )
        self.fc8_1 = nn.Sequential(
            nn.Linear(4096 // 2, feat_dim)
        )
        self.fc8_2 = nn.Sequential(
            nn.Linear(4096 // 2, feat_dim)
        )
        self.l2norm_1 = Normalize(2)
        self.l2norm_2 = Normalize(2)

        if self.branch == 3:
            self.fc8_3 = nn.Sequential(
                nn.Linear(4096 // 2, feat_dim)
            )
            self.l2norm_3 = Normalize(2)

    def forward(self, x):
        x1 = self.conv_block_1(x)
        x2 = self.conv_block_2(x1)
        x3 = self.conv_block_3(x2)
        x4 = self.conv_block_4(x3)
        x5 = self.conv_block_5(x4)
        x5 = x5.view(x.size(0), -1)
        x6 = self.fc6(x5)
        x7 = self.fc7(x6)

        if self.branch == 3:
            x8_1 = self.fc8_1(x7)
            x8_1 = self.l2norm_1(x8_1)
            x8_2 = self.fc8_2(x7)
            x8_3 = self.fc8_3(x7)
            return x8_1, x8_2, x8_3
        else:
            x8_1 = self.fc8_1(x7)
            x8_1 = self.l2norm_1(x8_1)
            x8_2 = self.fc8_2(x7)
            x8_2 = self.l2norm_2(x8_2)
            return x8_1, x8_2

    def compute_feat(self, x, layer=7):
        if layer <= 0:
            return x
        x = self.conv_block_1(x)
        if layer == 1:
            return x
        x = self.conv_block_2(x)
        if layer == 2:
            return x
        x = self.conv_block_3(x)
        if layer == 3:
            return x
        x = self.conv_block_4(x)
        if layer == 4:
            return x
        x = self.conv_block_5(x)
        if layer == 5:
            return x
        x = x.view(x.shape[0], -1)
        x = self.fc6(x)
        if layer == 6:
            return x
        x = self.fc7(x)
        if layer == 7:
            return x

        if self.branch == 3:
            x8_1 = self.fc8_1(x)
            x8_1 = self.l2norm_1(x8_1)
            x8_2 = self.fc8_2(x)
            x8_3 = self.fc8_3(x)
            x = torch.cat((x8_1, x8_2, x8_3), dim=1)
        else:
            x8_1 = self.fc8_1(x)
            x8_1 = self.l2norm_1(x8_1)
            x8_2 = self.fc8_2(x)
            x8_2 = self.l2norm_2(x8_2)
            x = torch.cat((x8_1, x8_2), dim=1)
        return x


#########################################
# ===== Classifiers ===== #
#########################################

class LinearClassifier(nn.Module):

    def __init__(self, dim_in, n_label=10):
        super(LinearClassifier, self).__init__()

        self.net = nn.Linear(dim_in, n_label)

    def forward(self, x):
        return self.net(x)


class NonLinearClassifier(nn.Module):

    def __init__(self, dim_in, n_label=10, p=0.1):
        super(NonLinearClassifier, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_in, 200),
            nn.Dropout(p=p),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),
            nn.Linear(200, n_label),
        )

    def forward(self, x):
        return self.net(x)


#########################################
# ===== alexnet cifar ===== #
#########################################
class alexnet_cifar(nn.Module):
    def __init__(self, feat_dim=128):
        super(alexnet_cifar, self).__init__()

        self.l_to_ab = alexnet_half_cifar(in_channel=1, feat_dim=feat_dim)
        self.ab_to_l = alexnet_half_cifar(in_channel=2, feat_dim=feat_dim)

    def forward(self, x):
        l, ab = torch.split(x, [1, 2], dim=1)
        feat_l = self.l_to_ab(l)
        feat_ab = self.ab_to_l(ab)
        return feat_l, feat_ab

    def compute_feat(self, x, layer):
        l, ab = torch.split(x, [1, 2], dim=1)
        feat_l = self.l_to_ab.compute_feat(l, layer)
        feat_ab = self.ab_to_l.compute_feat(ab, layer)
        feat = torch.cat((feat_l, feat_ab), dim=1)
        return feat


class alexnet_half_cifar(nn.Module):
    """alexnet for cifar"""
    def __init__(self, in_channel=1, w=0.5, branch=1, feat_dim=128):
        super(alexnet_half_cifar, self).__init__()

        self.w = w
        self.branch = branch
        self.feat_dim = feat_dim

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channel, int(96 * w), 3, 1, 1, bias=False),
            nn.BatchNorm2d(int(96 * w)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(int(96 * w), int(192 * w), 3, 1, 1, bias=False),
            nn.BatchNorm2d(int(192 * w)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(int(192 * w), int(384 * w), 3, 1, 1, bias=False),
            nn.BatchNorm2d(int(384 * w)),
            nn.ReLU(inplace=True),
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(int(384 * w), int(384 * w), 3, 1, 1, bias=False),
            nn.BatchNorm2d(int(384 * w)),
            nn.ReLU(inplace=True),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(int(384 * w), int(192 * w), 3, 1, 1, bias=False),
            nn.BatchNorm2d(int(192 * w)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )

        self.fc6 = nn.Sequential(
            nn.Linear(int(192 * 3 * 3 * w), int(1024 * w)),
            nn.BatchNorm1d(int(1024 * w)),
            nn.ReLU(inplace=True),
        )
        self.fc7 = nn.Sequential(
            nn.Linear(int(1024 * w), int(1024 * w)),
            nn.BatchNorm1d(int(1024 * w)),
            nn.ReLU(inplace=True),
        )
        self.fc8 = nn.Sequential(
            nn.Linear(int(1024 * w), feat_dim)
        )
        if self.branch >= 2:
            self.fc8_2 = nn.Sequential(nn.Linear(int(4096 * w), feat_dim))
        if self.branch >= 3:
            self.fc8_3 = nn.Sequential(nn.Linear(int(4096 * w), feat_dim))
        if self.branch >= 4:
            raise NotImplementedError('# of branches not supported yet: {}'.format(self.branch))
        self.l2norm = Normalize(2)

    def forward(self, x):
        x1 = self.conv_block_1(x)
        x2 = self.conv_block_2(x1)
        x3 = self.conv_block_3(x2)
        x4 = self.conv_block_4(x3)
        x5 = self.conv_block_5(x4)
        x5 = x5.view(x.size(0), -1)
        x6 = self.fc6(x5)
        x7 = self.fc7(x6)
        x8 = self.fc8(x7)
        x8 = self.l2norm(x8)
        if self.branch == 1:
            return x8
        if self.branch == 2:
            x8_2 = self.l2norm(self.fc8_2(x7))
            return x8, x8_2
        if self.branch == 3:
            x8_2 = self.l2norm(self.fc8_2(x7))
            x8_3 = self.l2norm(self.fc8_3(x7))
            return x8, x8_2, x8_3

    def compute_feat(self, x, layer):
        if layer <= 0:
            return x
        x = self.conv_block_1(x)
        if layer == 1:
            return x
        x = self.conv_block_2(x)
        if layer == 2:
            return x
        x = self.conv_block_3(x)
        if layer == 3:
            return x
        x = self.conv_block_4(x)
        if layer == 4:
            return x
        x = self.conv_block_5(x)
        if layer == 5:
            return x
        x = x.view(x.shape[0], -1)
        x = self.fc6(x)
        if layer == 6:
            return x
        x = self.fc7(x)
        if layer == 7:
            return x
        x8 = self.l2norm(self.fc8(x))
        if self.branch == 1:
            return x8
        if self.branch == 2:
            x8_2 = self.l2norm(self.fc8_2(x))
            return x8, x8_2
        if self.branch == 3:
            x8_2 = self.l2norm(self.fc8_2(x))
            x8_3 = self.l2norm(self.fc8_3(x))
            return x8, x8_2, x8_3


if __name__ == '__main__':
    model = alexnet_cifar()

    data = torch.randn(2, 3, 32, 32)

    out1, out2 = model(data)

    print(out1.shape)
    print(out2.shape)
