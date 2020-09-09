from __future__ import print_function

import torch
import torch.nn as nn


# ===================
# AlexNet Classifier
# ===================
class AlexNetClassifier(nn.Module):
    def __init__(self, num_class=10):
        super(AlexNetClassifier, self).__init__()

        self.l_to_ab = AlexNetHalf(in_channel=1, num_class=num_class)
        self.ab_to_l = AlexNetHalf(in_channel=2, num_class=num_class)

    def forward(self, x):
        l, ab = torch.split(x, [1, 2], dim=1)
        logit_l = self.l_to_ab(l)
        logit_ab = self.ab_to_l(ab)
        return logit_l, logit_ab


class AlexNetHalf(nn.Module):
    """alexnet classification network"""
    def __init__(self, in_channel=1, w=0.5, num_class=10):
        super(AlexNetHalf, self).__init__()

        self.w = w
        self.num_class = num_class

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
            nn.Dropout(),
            nn.Linear(int(192 * 7 * 7 * w), int(4096 * w)),
            # nn.BatchNorm1d(int(4096 * w)),
            nn.ReLU(inplace=True),
        )
        self.fc7 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(int(4096 * w), int(4096 * w)),
            # nn.BatchNorm1d(int(4096 * w)),
            nn.ReLU(inplace=True),
        )
        self.fc8 = nn.Sequential(
            nn.Linear(int(4096 * w), num_class)
        )

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
        return x8


# ===================
# ResNet Classifier
# ===================
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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


class ResNetSTL(nn.Module):
    """resnet classifier"""
    def __init__(self, block, layers, in_channel=3, width=0.5, num_class=10):
        super(ResNetSTL, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channel, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.base = int(width * 64)

        self.layer1 = self._make_layer(block, self.base, layers[0])
        self.layer2 = self._make_layer(block, 2 * self.base, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * self.base, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * self.base, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8 * self.base * block.expansion, num_class)

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18STL(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetSTL(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet50STL(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetSTL(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101STL(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetSTL(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


class ResNetClassifier(nn.Module):
    def __init__(self, name='resnet50', num_class=10):
        super(ResNetClassifier, self).__init__()

        ch1 = 1
        ch2 = 2
        if name == 'resnet18':
            self.l_to_ab = resnet18STL(in_channel=ch1, num_class=num_class)
            self.ab_to_l = resnet18STL(in_channel=ch2, num_class=num_class)
        elif name == 'resnet50':
            self.l_to_ab = resnet50STL(in_channel=ch1, num_class=num_class)
            self.ab_to_l = resnet50STL(in_channel=ch2, num_class=num_class)
        elif name == 'resnet101':
            self.l_to_ab = resnet101STL(in_channel=ch1, num_class=num_class)
            self.ab_to_l = resnet101STL(in_channel=ch2, num_class=num_class)
        else:
            raise NotImplementedError('model {} is not implemented'.format(name))

    def forward(self, x):
        l, ab = torch.split(x, [1, 2], dim=1)
        logit_l = self.l_to_ab(l)
        logit_ab = self.ab_to_l(ab)
        return logit_l, logit_ab


if __name__ == '__main__':

    # classifier = AlexNetClassifier(num_class=10)
    classifier = ResNetClassifier(name='resnet18', num_class=10)

    data = torch.randn(4, 3, 64, 64)
    out1, out2 = classifier(data)

    print(out1.shape)
    print(out2.shape)
