from __future__ import print_function

import torch
import torch.nn as nn
from .coupling import CouplingLayer, SqueezeLayer, MaskType


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def convkxk(in_planes, out_planes, kernel, stride=1, bias=True):
    """kxk convolution with padding"""
    assert kernel % 2 == 1, 'kernel size is even number!'
    padding = (kernel - 1) // 2
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride,
                     padding=padding, bias=bias)


class RevBlock(nn.Module):
    """one reversible block"""
    def __init__(self, in_channel, no_activation=False, is_even=False):
        super(RevBlock, self).__init__()
        self.in_channel = in_channel
        if is_even:
            self.ch1 = in_channel // 2
            self.ch2 = in_channel - self.ch1
        else:
            self.ch2 = in_channel // 2
            self.ch1 = in_channel - self.ch2
        self.no_activation = no_activation

        self.f_bn1 = nn.BatchNorm2d(self.ch2)
        self.f_conv1 = conv3x3(self.ch2, self.ch2)
        self.f_bn2 = nn.BatchNorm2d(self.ch2)
        self.f_conv2 = conv3x3(self.ch2, self.ch1)

        self.g_bn1 = nn.BatchNorm2d(self.ch1)
        self.g_conv1 = conv3x3(self.ch1, self.ch1)
        self.g_bn2 = nn.BatchNorm2d(self.ch1)
        self.g_conv2 = conv3x3(self.ch1, self.ch2)

        if no_activation:
            self.f_func = nn.Sequential(
                self.f_conv1,
                self.f_bn2,
                nn.ReLU(),
                self.f_conv2,
            )
        else:
            self.f_func = nn.Sequential(
                self.f_bn1,
                nn.ReLU(),
                self.f_conv1,
                self.f_bn2,
                nn.ReLU(),
                self.f_conv2,
            )

        self.g_func = nn.Sequential(
            self.g_bn1,
            nn.ReLU(),
            self.g_conv1,
            self.g_bn2,
            nn.ReLU(),
            self.g_conv2,
        )

    def forward(self, x):
        x1, x2 = torch.split(x, [self.ch1, self.ch2], dim=1)

        y1 = x1 + self.f_func(x2)
        y2 = x2 + self.g_func(y1)

        return torch.cat([y1, y2], dim=1)

    def reverse(self, y):
        y1, y2 = torch.split(y, [self.ch1, self.ch2], dim=1)

        x2 = y2 - self.g_func(y1)
        x1 = y1 - self.f_func(x2)

        return torch.cat([x1, x2], dim=1)


class RevBlockV2(nn.Module):
    """Reversible Residual Block V2"""
    def __init__(self, in_channel=3, kernel=1, no_activation=False, is_even=False):
        super(RevBlockV2, self).__init__()
        self.in_channel = in_channel
        if is_even:
            self.ch1 = in_channel // 2
            self.ch2 = in_channel - self.ch1
        else:
            self.ch2 = in_channel // 2
            self.ch1 = in_channel - self.ch2
        self.no_activation = no_activation

        self.f_conv1 = convkxk(self.ch2, self.ch2, kernel)
        self.f_conv2 = convkxk(self.ch2, self.ch1, kernel)

        self.g_conv1 = convkxk(self.ch1, self.ch1, kernel)
        self.g_conv2 = convkxk(self.ch1, self.ch2, kernel)

        if no_activation:
            self.f_func = nn.Sequential(
                self.f_conv1,
                nn.ReLU(),
                self.f_conv2,
            )
        else:
            self.f_func = nn.Sequential(
                nn.ReLU(),
                self.f_conv1,
                nn.ReLU(),
                self.f_conv2,
            )

        self.g_func = nn.Sequential(
            nn.ReLU(),
            self.g_conv1,
            nn.ReLU(),
            self.g_conv2,
        )

    def forward(self, x):
        x1, x2 = torch.split(x, [self.ch1, self.ch2], dim=1)

        y1 = x1 + self.f_func(x2)
        y2 = x2 + self.g_func(y1)

        return torch.cat([y1, y2], dim=1)

    def reverse(self, y):
        y1, y2 = torch.split(y, [self.ch1, self.ch2], dim=1)

        x2 = y2 - self.g_func(y1)
        x1 = y1 - self.f_func(x2)

        return torch.cat([x1, x2], dim=1)


class RevBlockV3(nn.Module):
    """change the dimension order earlier"""
    def __init__(self, in_channel=3, kernel=1, no_activation=False, is_even=False, half_flow=False):
        super(RevBlockV3, self).__init__()
        self.half_flow = half_flow
        if self.half_flow:
            in_channel = in_channel - 1

        self.in_channel = in_channel

        if is_even:
            self.ch1 = in_channel // 2
            self.ch2 = in_channel - self.ch1
        else:
            self.ch2 = in_channel // 2
            self.ch1 = in_channel - self.ch2
        self.no_activation = no_activation

        self.f_conv1 = convkxk(self.ch2, min(self.ch1, self.ch2), kernel)
        self.f_conv2 = convkxk(min(self.ch1, self.ch2), self.ch1, kernel)

        self.g_conv1 = convkxk(self.ch1, min(self.ch1, self.ch2), kernel)
        self.g_conv2 = convkxk(min(self.ch1, self.ch2), self.ch2, kernel)

        if no_activation:
            self.f_func = nn.Sequential(
                self.f_conv1,
                nn.ReLU(),
                self.f_conv2,
            )
        else:
            self.f_func = nn.Sequential(
                nn.ReLU(),
                self.f_conv1,
                nn.ReLU(),
                self.f_conv2,
            )

        self.g_func = nn.Sequential(
            nn.ReLU(),
            self.g_conv1,
            nn.ReLU(),
            self.g_conv2,
        )

    def forward(self, x):
        if self.half_flow:
            x0, x1, x2 = torch.split(x, [1, self.ch1, self.ch2], dim=1)
        else:
            x1, x2 = torch.split(x, [self.ch1, self.ch2], dim=1)

        y1 = x1 + self.f_func(x2)
        y2 = x2 + self.g_func(y1)

        if self.half_flow:
            return torch.cat([x0, y1, y2], dim=1)
        else:
            return torch.cat([y1, y2], dim=1)

    def reverse(self, y):
        if self.half_flow:
            x0, y1, y2 = torch.split(y, [1, self.ch1, self.ch2], dim=1)
        else:
            y1, y2 = torch.split(y, [self.ch1, self.ch2], dim=1)

        x2 = y2 - self.g_func(y1)
        x1 = y1 - self.f_func(x2)

        if self.half_flow:
            return torch.cat([x0, x1, x2], dim=1)
        else:
            return torch.cat([x1, x2], dim=1)


class RevBlockV4(nn.Module):
    """reduce f and g to single layer function"""
    def __init__(self, in_channel=3, kernel=1, no_activation=False, is_even=False, half_flow=False):
        super(RevBlockV4, self).__init__()
        self.half_flow = half_flow
        if self.half_flow:
            in_channel = in_channel - 1

        self.in_channel = in_channel

        if is_even:
            self.ch1 = in_channel // 2
            self.ch2 = in_channel - self.ch1
        else:
            self.ch2 = in_channel // 2
            self.ch1 = in_channel - self.ch2
        self.no_activation = no_activation

        self.f_conv1 = convkxk(self.ch2, self.ch1, kernel)

        self.g_conv1 = convkxk(self.ch1, self.ch2, kernel)

        if no_activation:
            self.f_func = nn.Sequential(
                self.f_conv1,
            )
        else:
            self.f_func = nn.Sequential(
                nn.ReLU(),
                self.f_conv1,
            )

        self.g_func = nn.Sequential(
            nn.ReLU(),
            self.g_conv1,
        )

    def forward(self, x):
        if self.half_flow:
            x0, x1, x2 = torch.split(x, [1, self.ch1, self.ch2], dim=1)
        else:
            x1, x2 = torch.split(x, [self.ch1, self.ch2], dim=1)

        y1 = x1 + self.f_func(x2)
        y2 = x2 + self.g_func(y1)

        if self.half_flow:
            return torch.cat([x0, y1, y2], dim=1)
        else:
            return torch.cat([y1, y2], dim=1)

    def reverse(self, y):
        if self.half_flow:
            x0, y1, y2 = torch.split(y, [1, self.ch1, self.ch2], dim=1)
        else:
            y1, y2 = torch.split(y, [self.ch1, self.ch2], dim=1)

        x2 = y2 - self.g_func(y1)
        x1 = y1 - self.f_func(x2)

        if self.half_flow:
            return torch.cat([x0, x1, x2], dim=1)
        else:
            return torch.cat([x1, x2], dim=1)


class RevBlockV5(nn.Module):
    """Only contains Linear function"""
    def __init__(self, in_channel=3, kernel=1, no_activation=False, is_even=False, half_flow=False):
        super(RevBlockV5, self).__init__()
        self.half_flow = half_flow
        if self.half_flow:
            in_channel = in_channel - 1

        self.in_channel = in_channel

        if is_even:
            self.ch1 = in_channel // 2
            self.ch2 = in_channel - self.ch1
        else:
            self.ch2 = in_channel // 2
            self.ch1 = in_channel - self.ch2
        self.no_activation = no_activation

        self.f_conv1 = convkxk(self.ch2, self.ch1, kernel)

        self.g_conv1 = convkxk(self.ch1, self.ch2, kernel)

        self.f_func = nn.Sequential(
            self.f_conv1,
        )

        self.g_func = nn.Sequential(
            self.g_conv1,
        )

    def forward(self, x):
        if self.half_flow:
            x0, x1, x2 = torch.split(x, [1, self.ch1, self.ch2], dim=1)
        else:
            x1, x2 = torch.split(x, [self.ch1, self.ch2], dim=1)

        y1 = x1 + self.f_func(x2)
        y2 = x2 + self.g_func(y1)

        if self.half_flow:
            return torch.cat([x0, y1, y2], dim=1)
        else:
            return torch.cat([y1, y2], dim=1)

    def reverse(self, y):
        if self.half_flow:
            x0, y1, y2 = torch.split(y, [1, self.ch1, self.ch2], dim=1)
        else:
            y1, y2 = torch.split(y, [self.ch1, self.ch2], dim=1)

        x2 = y2 - self.g_func(y1)
        x1 = y1 - self.f_func(x2)

        if self.half_flow:
            return torch.cat([x0, x1, x2], dim=1)
        else:
            return torch.cat([x1, x2], dim=1)


class RevBlockV6(nn.Module):
    """Volume preserving functions"""
    def __init__(self, in_channel=3, kernel=1, no_activation=False, split_channel=0, half_flow=False):
        super(RevBlockV6, self).__init__()
        self.in_channel = in_channel
        self.split_channel = split_channel
        self.no_activation = no_activation

        print(split_channel)

        self.ch1 = 1
        self.ch2 = 2

        self.half_flow = half_flow
        if self.half_flow:
            self.ch1 = 1
            self.ch2 = 1

        if self.split_channel <= 2:
            self.f_conv1 = convkxk(self.ch2, self.ch2, kernel)
            self.f_conv2 = convkxk(self.ch2, self.ch1, kernel)
        else:
            self.f_conv1 = convkxk(self.ch1, self.ch2, kernel)
            self.f_conv2 = convkxk(self.ch2, self.ch2, kernel)

        if no_activation:
            self.f_func = nn.Sequential(
                self.f_conv1,
                nn.ReLU(),
                self.f_conv2,
            )
        else:
            self.f_func = nn.Sequential(
                nn.ReLU(),
                self.f_conv1,
                nn.ReLU(),
                self.f_conv2,
            )

    def forward(self, x):
        if self.half_flow:
            a, b, c = torch.split(x, [1, 1, 1], dim=1)
            if self.split_channel == 1:
                return torch.cat([a, b + self.f_func(c), c], dim=1)
            elif self.split_channel == 2:
                return torch.cat([a, b, c + self.f_func(b)], dim=1)
            else:
                raise NotImplementedError('split_channel wrong!')
        else:
            if self.split_channel == 0:
                x1, x2 = torch.split(x, [1, 2], dim=1)
                x1 = x1 + self.f_func(x2)
                y = torch.cat([x1, x2], dim=1)
            elif self.split_channel == 1:
                a, b, c = torch.split(x, [1, 1, 1], dim=1)
                x1 = b
                x2 = torch.cat([a, c], dim=1)
                x1 = x1 + self.f_func(x2)
                y = torch.cat([a, x1, c], dim=1)
            elif self.split_channel == 2:
                x2, x1 = torch.split(x, [2, 1], dim=1)
                x1 = x1 + self.f_func(x2)
                y = torch.cat([x2, x1], dim=1)
            elif self.split_channel == 3:
                x1, x2 = torch.split(x, [1, 2], dim=1)
                x2 = x2 + self.f_func(x1)
                y = torch.cat([x1, x2], dim=1)
            else:
                raise NotImplementedError('{}'.format(self.split_channel))
            return y

    def reverse(self, y):
        if self.half_flow:
            a, b, c = torch.split(y, [1, 1, 1], dim=1)
            if self.split_channel == 1:
                return torch.cat([a, b - self.f_func(c), c], dim=1)
            elif self.split_channel == 2:
                return torch.cat([a, b, c - self.f_func(b)], dim=1)
            else:
                raise NotImplementedError('split_channel wrong!')
        else:
            if self.split_channel == 0:
                y1, y2 = torch.split(y, [1, 2], dim=1)
                y1 = y1 - self.f_func(y2)
                x = torch.cat([y1, y2], dim=1)
            elif self.split_channel == 1:
                a, b, c = torch.split(y, [1, 1, 1], dim=1)
                y1 = b
                y2 = torch.cat([a, c], dim=1)
                y1 = y1 - self.f_func(y2)
                x = torch.cat([a, y1, c], dim=1)
            elif self.split_channel == 2:
                y2, y1 = torch.split(y, [2, 1], dim=1)
                y1 = y1 - self.f_func(y2)
                x = torch.cat([y2, y1], dim=1)
            elif self.split_channel == 3:
                y1, y2 = torch.split(y, [1, 2], dim=1)
                y2 = y2 - self.f_func(y1)
                x = torch.cat([y1, y2], dim=1)
            else:
                raise NotImplementedError('{}'.format(self.split_channel))
            return x


class RevBlockV7(nn.Module):
    """Volume preserving functions"""
    def __init__(self, in_channel=3, kernel=1, no_activation=False):
        super(RevBlockV7, self).__init__()
        self.in_channel = in_channel
        self.no_activation = no_activation

        if no_activation:
            self.func = nn.Sequential(
                convkxk(in_channel, 8, kernel),
                nn.ReLU(),
                convkxk(8, in_channel, kernel),
            )
        else:
            self.func = nn.Sequential(
                nn.ReLU(),
                convkxk(in_channel, 8, kernel),
                nn.ReLU(),
                convkxk(8, in_channel, kernel),
            )

    def forward(self, x):
            y = self.func(x)
            return y

    def reverse(self, y):
        return y


class RevBlockV9(nn.Module):
    """Volume preserving functions with more powerful functions"""
    def __init__(self, in_channel=3, kernel=1, no_activation=False, split_channel=0, half_flow=False):
        super(RevBlockV9, self).__init__()
        self.in_channel = in_channel
        self.split_channel = split_channel
        self.no_activation = no_activation

        print(split_channel)

        self.ch1 = 1
        self.ch2 = 2

        self.half_flow = half_flow
        if self.half_flow:
            self.ch1 = 1
            self.ch2 = 1

        if self.split_channel <= 2:
            self.f_conv1 = convkxk(self.ch2, 8, kernel)
            self.f_conv2 = convkxk(8, self.ch1, kernel)
        else:
            self.f_conv1 = convkxk(self.ch1, 8, kernel)
            self.f_conv2 = convkxk(8, self.ch2, kernel)

        if no_activation:
            self.f_func = nn.Sequential(
                self.f_conv1,
                nn.ReLU(),
                self.f_conv2,
            )
        else:
            self.f_func = nn.Sequential(
                nn.ReLU(),
                self.f_conv1,
                nn.ReLU(),
                self.f_conv2,
            )

    def forward(self, x):
        if self.half_flow:
            a, b, c = torch.split(x, [1, 1, 1], dim=1)
            if self.split_channel == 1:
                return torch.cat([a, b + self.f_func(c), c], dim=1)
            elif self.split_channel == 2:
                return torch.cat([a, b, c + self.f_func(b)], dim=1)
            else:
                raise NotImplementedError('split_channel wrong!')
        else:
            if self.split_channel == 0:
                x1, x2 = torch.split(x, [1, 2], dim=1)
                x1 = x1 + self.f_func(x2)
                y = torch.cat([x1, x2], dim=1)
            elif self.split_channel == 1:
                a, b, c = torch.split(x, [1, 1, 1], dim=1)
                x1 = b
                x2 = torch.cat([a, c], dim=1)
                x1 = x1 + self.f_func(x2)
                y = torch.cat([a, x1, c], dim=1)
            elif self.split_channel == 2:
                x2, x1 = torch.split(x, [2, 1], dim=1)
                x1 = x1 + self.f_func(x2)
                y = torch.cat([x2, x1], dim=1)
            elif self.split_channel == 3:
                x1, x2 = torch.split(x, [1, 2], dim=1)
                x2 = x2 + self.f_func(x1)
                y = torch.cat([x1, x2], dim=1)
            else:
                raise NotImplementedError('{}'.format(self.split_channel))
            return y

    def reverse(self, y):
        if self.half_flow:
            a, b, c = torch.split(y, [1, 1, 1], dim=1)
            if self.split_channel == 1:
                return torch.cat([a, b - self.f_func(c), c], dim=1)
            elif self.split_channel == 2:
                return torch.cat([a, b, c - self.f_func(b)], dim=1)
            else:
                raise NotImplementedError('split_channel wrong!')
        else:
            if self.split_channel == 0:
                y1, y2 = torch.split(y, [1, 2], dim=1)
                y1 = y1 - self.f_func(y2)
                x = torch.cat([y1, y2], dim=1)
            elif self.split_channel == 1:
                a, b, c = torch.split(y, [1, 1, 1], dim=1)
                y1 = b
                y2 = torch.cat([a, c], dim=1)
                y1 = y1 - self.f_func(y2)
                x = torch.cat([a, y1, c], dim=1)
            elif self.split_channel == 2:
                y2, y1 = torch.split(y, [2, 1], dim=1)
                y1 = y1 - self.f_func(y2)
                x = torch.cat([y2, y1], dim=1)
            elif self.split_channel == 3:
                y1, y2 = torch.split(y, [1, 2], dim=1)
                y2 = y2 - self.f_func(y1)
                x = torch.cat([y1, y2], dim=1)
            else:
                raise NotImplementedError('{}'.format(self.split_channel))
            return x


class RevNetGenerator(nn.Module):
    """revnet for view generation"""
    def __init__(self, n_block, block_type=2, layer_mode='A', is_rgb=True):
        """
        :param n_block:
        :param block_type: 1: RevBlock, 2: RevBlockV2
        :param layer_mode: A: kernel 1, B first 3, C last 3
        """
        super(RevNetGenerator, self).__init__()

        self.block_type = block_type
        self.layer_mode = layer_mode
        self.is_rgb = is_rgb
        self.generator = self._make_layer(n_block, in_channel=3)

    def _make_layer(self, n_block, in_channel=3):
        layers = list([])

        if self.block_type == -1: # original 7
            layers.append(CouplingLayer(3, MaskType.CHECKERBOARD, False, no_activation=True))
            layers.append(CouplingLayer(3, MaskType.CHECKERBOARD, True))
            layers.append(CouplingLayer(3, MaskType.CHECKERBOARD, False))

            layers.append(SqueezeLayer(squeeze=True))
            layers.append(CouplingLayer(3*4, MaskType.CHANNEL_WISE, False))
            layers.append(CouplingLayer(3 * 4, MaskType.CHANNEL_WISE, False))
            layers.append(CouplingLayer(3 * 4, MaskType.CHANNEL_WISE, True))
            layers.append(SqueezeLayer(squeeze=False))

            "volume preserving"
            for i in range(3):
                split_channel = i % 3 if self.is_rgb else 0
                layers.append(RevBlockV6(in_channel=in_channel, no_activation=False, split_channel=split_channel))
        else:
            for i in range(n_block):
                if self.block_type == 1:
                    if i == 0:
                        layers.append(RevBlock(in_channel=in_channel, no_activation=True, is_even=(i%2==0)))
                    else:
                        layers.append(RevBlock(in_channel=in_channel, is_even=(i%2==0)))
                elif self.block_type == 2:
                    if i == 0:
                        if self.layer_mode == 'B':
                            layers.append(RevBlockV2(in_channel=in_channel, kernel=3, no_activation=True, is_even=(i % 2 == 0)))
                        else:
                            layers.append(RevBlockV2(in_channel=in_channel, no_activation=True, is_even=(i % 2 == 0)))
                    elif i == n_block - 1:
                        if self.layer_mode == 'C':
                            layers.append(RevBlockV2(in_channel=in_channel, kernel=3, no_activation=True, is_even=(i % 2 == 0)))
                        else:
                            layers.append(RevBlockV2(in_channel=in_channel, no_activation=True, is_even=(i % 2 == 0)))
                    else:
                        layers.append(RevBlockV2(in_channel=in_channel, no_activation=True, is_even=(i % 2 == 0)))
                elif self.block_type == 3:
                    "lower dimension earlier"
                    is_even = (i % 2 == 0) if self.is_rgb else True
                    if i == 0:
                        layers.append(RevBlockV3(in_channel=in_channel, no_activation=True, is_even=is_even))
                    else:
                        layers.append(RevBlockV3(in_channel=in_channel, no_activation=False, is_even=is_even))
                elif self.block_type == 4:
                    "only one layer"
                    is_even = (i % 2 == 0) if self.is_rgb else True
                    if i == 0:
                        layers.append(RevBlockV4(in_channel=in_channel, no_activation=True, is_even=is_even))
                    else:
                        layers.append(RevBlockV4(in_channel=in_channel, no_activation=False, is_even=is_even))
                elif self.block_type == 5:
                    "linear layer"
                    # is_even = (i % 2 == 0) if self.is_rgb else True
                    # follows rgb-style splitting
                    is_even = (i % 2 == 0)
                    if i == 0:
                        layers.append(RevBlockV5(in_channel=in_channel, no_activation=True, is_even=is_even))
                    else:
                        layers.append(RevBlockV5(in_channel=in_channel, no_activation=False, is_even=is_even))
                elif self.block_type == 6:
                    "volume preserving"
                    # split_channel = i % 3 if self.is_rgb else 0
                    # split_channel = i % 3 if self.is_rgb else (i + 3) % 3
                    split_channel = i % 3 if self.is_rgb else (i % 2) * 3
                    if i == 0:
                        layers.append(RevBlockV6(in_channel=in_channel,
                                                 no_activation=True,
                                                 split_channel=split_channel))
                    else:
                        layers.append(RevBlockV6(in_channel=in_channel,
                                                 no_activation=False,
                                                 split_channel=split_channel))
                elif self.block_type == 7:
                    if i == 0:
                        layers.append(RevBlockV7(in_channel=in_channel,
                                                 no_activation=True,
                                                 kernel=1))
                    else:
                        layers.append(RevBlockV7(in_channel=in_channel,
                                                 no_activation=False,
                                                 kernel=1))
                elif self.block_type == 8:
                    if i == 0:
                        layers.append(RevBlockV7(in_channel=in_channel,
                                                 no_activation=True,
                                                 kernel=3))
                    else:
                        layers.append(RevBlockV7(in_channel=in_channel,
                                                 no_activation=False,
                                                 kernel=3))
                elif self.block_type == 9:
                    "volume preserving"
                    split_channel = i % 3 if self.is_rgb else (i % 2) * 3
                    if i == 0:
                        layers.append(RevBlockV9(in_channel=in_channel,
                                                 no_activation=True,
                                                 split_channel=split_channel))
                    else:
                        layers.append(RevBlockV9(in_channel=in_channel,
                                                 no_activation=False,
                                                 split_channel=split_channel))
                elif self.block_type == 13:
                    "lower dimension earlier"
                    is_even = (i % 2 == 0) if self.is_rgb else True
                    if i == 0:
                        layers.append(RevBlockV3(in_channel=in_channel, no_activation=True, is_even=is_even, half_flow=True))
                    else:
                        layers.append(RevBlockV3(in_channel=in_channel, no_activation=False, is_even=is_even, half_flow=True))
                elif self.block_type == 14:
                    "only one layer"
                    is_even = (i % 2 == 0) if self.is_rgb else True
                    if i == 0:
                        layers.append(RevBlockV4(in_channel=in_channel, no_activation=True, is_even=is_even, half_flow=True))
                    else:
                        layers.append(RevBlockV4(in_channel=in_channel, no_activation=False, is_even=is_even, half_flow=True))
                elif self.block_type == 15:
                    "linear layer"
                    is_even = (i % 2 == 0) if self.is_rgb else True
                    if i == 0:
                        layers.append(RevBlockV5(in_channel=in_channel, no_activation=True, is_even=is_even, half_flow=True))
                    else:
                        layers.append(RevBlockV5(in_channel=in_channel, no_activation=False, is_even=is_even, half_flow=True))
                elif self.block_type == 16:
                    "volume preserving"
                    split_channel = (i % 2) + 1
                    if i == 0:
                        layers.append(RevBlockV6(in_channel=in_channel,
                                                 no_activation=True,
                                                 split_channel=split_channel, half_flow=True))
                    else:
                        layers.append(RevBlockV6(in_channel=in_channel,
                                                 no_activation=False,
                                                 split_channel=split_channel, half_flow=True))
                else:
                    raise NotImplementedError('block type not suported: {}'.format(self.block_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.generator(x)

    def reverse(self, y):
        for i in reversed(range(len(self.generator))):
            y = self.generator[i].reverse(y)
        return y


if __name__ == '__main__':

    # model = RevBlock(3, is_even=False)
    model = RevNetGenerator(6, block_type=7, is_rgb=True)
    # model = model.eval()
    data = torch.randn(10, 3, 32, 32)

    out = model(data)
    print(data.mean())
    print(out.mean())

    input = model.reverse(out)
    print(input.mean())
