from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ==============
# STL dataset
# ==============
class ResNetSTL(nn.Module):

    def __init__(self, block, layers, low_dim=128, in_channel=3, width=0.5, is_norm=True):
        super(ResNetSTL, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channel, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.base = int(width * 64)

        self.layer1 = self._make_layer(block, self.base, layers[0])
        self.layer2 = self._make_layer(block, 2 * self.base, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * self.base, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * self.base, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8 * self.base * block.expansion, low_dim)

        if is_norm:
            self.norm = Normalize(2)
        else:
            self.norm = nn.Sequential()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        zero_init_residual = True
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list([])
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, layer=7):
        if layer <= 0:
            return x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if layer == 1:
            return x

        x = self.layer1(x)
        if layer == 2:
            return x

        x = self.layer2(x)
        if layer == 3:
            return x

        x = self.layer3(x)
        if layer == 4:
            return x

        x = self.layer4(x)
        if layer == 5:
            return x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if layer == 6:
            return x

        x = self.fc(x)
        x = self.norm(x)

        return x


def resnet18STL(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetSTL(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet50STL(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetSTL(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101STL(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetSTL(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


class ResNetSB(nn.Module):
    """feature learning network"""
    def __init__(self, name='resnet50'):
        super(ResNetSB, self).__init__()
        if name == 'resnet18':
            self.l_to_ab = resnet18STL(in_channel=1)
            self.ab_to_l = resnet18STL(in_channel=2)
        elif name == 'resnet50':
            self.l_to_ab = resnet50STL(in_channel=1)
            self.ab_to_l = resnet50STL(in_channel=2)
        elif name == 'resnet101':
            self.l_to_ab = resnet101STL(in_channel=1)
            self.ab_to_l = resnet101STL(in_channel=2)
        else:
            raise NotImplementedError('model {} is not implemented'.format(name))

    def forward(self, x, layer=7):
        l, ab = torch.split(x, [1, 2], dim=1)
        feat_l = self.l_to_ab(l, layer)
        feat_ab = self.ab_to_l(ab, layer)
        return feat_l, feat_ab

    def compute_feat(self, x, layer=6):
        l, ab = torch.split(x, [1, 2], dim=1)
        feat_l = self.l_to_ab(l, layer)
        feat_ab = self.ab_to_l(ab, layer)
        feat = torch.cat((feat_l, feat_ab), dim=1)
        return feat


class MIFCNet(nn.Module):
    """Simple custom network for computing MI.
    """
    def __init__(self, n_input, n_units):
        """
        Args:
            n_input: Number of input units.
            n_units: Number of output units.
        """
        super(self, MIFCNet).__init__()

        assert(n_units >= n_input)

        self.linear_shortcut = nn.Linear(n_input, n_units)
        self.block_nonlinear = nn.Sequential(
            nn.Linear(n_input, n_units),
            nn.BatchNorm1d(n_units),
            nn.ReLU(),
            nn.Linear(n_units, n_units)
        )

        # initialize the initial projection to a sort of noisy copy
        eye_mask = np.zeros((n_units, n_input), dtype=np.uint8)
        for i in range(n_input):
            eye_mask[i, i] = 1

        self.linear_shortcut.weight.data.uniform_(-0.01, 0.01)
        self.linear_shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.)

    def forward(self, x):
        """
        Args:
            x: Input tensor.
        Returns:
            torch.Tensor: network output.
        """
        h = self.block_nonlinear(x) + self.linear_shortcut(x)
        return h


class ResNetMINEV1(nn.Module):
    """Mutual Information Neural Estimator"""
    def __init__(self, name='resnet50', ch1=2, ch2=2, split=[2, 2], is_norm=False):
        super(ResNetMINEV1, self).__init__()
        if is_norm:
            low_dim = 128
        else:
            low_dim = 256
        w = 0.5
        self.split = split

        # low_dim = 256
        # w = 0.5
        # is_norm = False
        # self.split = split

        # low_dim = 128
        # w = 0.5
        # is_norm = False
        if name == 'resnet18':
            self.l_to_ab = resnet18STL(in_channel=ch1, low_dim=low_dim, width=w, is_norm=is_norm)
            self.ab_to_l = resnet18STL(in_channel=ch2, low_dim=low_dim, width=w, is_norm=is_norm)
        elif name == 'resnet50':
            self.l_to_ab = resnet50STL(in_channel=ch1, low_dim=low_dim, width=w, is_norm=is_norm)
            self.ab_to_l = resnet50STL(in_channel=ch2, low_dim=low_dim, width=w, is_norm=is_norm)
        elif name == 'resnet101':
            self.l_to_ab = resnet101STL(in_channel=ch1, low_dim=low_dim, width=w, is_norm=is_norm)
            self.ab_to_l = resnet101STL(in_channel=ch2, low_dim=low_dim, width=w, is_norm=is_norm)
        else:
            raise NotImplementedError('model {} is not implemented'.format(name))

        # self.MI_view1 = MIFCNet(low_dim, 1024)
        # self.MI_view2 = MIFCNet(low_dim, 1024)

        # self.MI_view1 = nn.Sequential(
        #     nn.Linear(low_dim, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 1024)
        # )
        # self.MI_view2 = nn.Sequential(
        #     nn.Linear(low_dim, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 1024)
        # )

    def forward(self, x):
        l, ab = torch.split(x, self.split, dim=1)
        feat_l = self.l_to_ab(l)
        feat_ab = self.ab_to_l(ab)

        # feat_l = self.MI_view1(feat_l)
        # feat_ab = self.MI_view2(feat_ab)

        return feat_l, feat_ab


class ResNetMINEV2(nn.Module):
    """Mutual Information Neural Estimator"""
    def __init__(self, name='resnet50', ch=3):
        super(ResNetMINEV2, self).__init__()
        low_dim = 256
        w = 0.5
        is_norm = False
        if name == 'resnet18':
            self.encoder = resnet18STL(in_channel=ch, low_dim=low_dim * 2, width=w * 2, is_norm=is_norm)
        elif name == 'resnet50':
            self.encoder = resnet50STL(in_channel=ch, low_dim=low_dim * 2, width=w * 2, is_norm=is_norm)
        elif name == 'resnet101':
            self.encoder = resnet101STL(in_channel=ch, low_dim=low_dim * 2, width=w * 2, is_norm=is_norm)
        else:
            raise NotImplementedError('model {} is not implemented'.format(name))

        self.estimator = nn.Sequential(
            nn.Linear(2 * low_dim, 2 * low_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * low_dim, 1)
        )

    def forward(self, x):
        feat = self.encoder(x)
        out = self.estimator(feat)
        return out


if __name__ == '__main__':

    estimator = ResNetMINEV1()

    data = torch.randn(10, 4, 64, 64)
    f1, f2 = estimator(data)
    print(f1.shape)
    print(f2.shape)
