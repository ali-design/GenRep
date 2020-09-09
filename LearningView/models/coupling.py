import torch
import torch.nn as nn
from enum import IntEnum


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


class WNConv2d(nn.Module):
    """Weight-normalized 2d convolution.
    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
        kernel_size (int): Side length of each convolutional kernel.
        padding (int): Padding to add on edges of input.
        bias (bool): Use bias in the convolution operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
        super(WNConv2d, self).__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias))

    def forward(self, x):
        x = self.conv(x)

        return x


class MaskType(IntEnum):
    CHECKERBOARD = 0
    CHANNEL_WISE = 1


def checkerboard_mask(height, width, reverse=False, dtype=torch.float32,
                      device=None, requires_grad=False):
    """Get a checkerboard mask, such that no two entries adjacent entries
    have the same value. In non-reversed mask, top-left entry is 0.
    Args:
        height (int): Number of rows in the mask.
        width (int): Number of columns in the mask.
        reverse (bool): If True, reverse the mask (i.e., make top-left entry 1).
            Useful for alternating masks in RealNVP.
        dtype (torch.dtype): Data type of the tensor.
        device (torch.device): Device on which to construct the tensor.
        requires_grad (bool): Whether the tensor requires gradient.
    Returns:
        mask (torch.tensor): Checkerboard mask of shape (1, 1, height, width).
    """
    checkerboard = [[((i % 2) + j) % 2 for j in range(width)] for i in range(height)]
    mask = torch.tensor(checkerboard, dtype=dtype, device=device, requires_grad=requires_grad)

    if reverse:
        mask = 1 - mask

    # Reshape to (1, 1, height, width) for broadcasting with tensors of shape (B, C, H, W)
    mask = mask.view(1, 1, height, width)

    return mask


class CouplingLayer(nn.Module):
    """Coupling layer in RealNVP.
    Args:
        in_channels (int): Number of channels in the input.
        mask_type (MaskType): One of `MaskType.CHECKERBOARD` or `MaskType.CHANNEL_WISE`.
        reverse_mask (bool): Whether to reverse the mask. Useful for alternating masks.
    """
    def __init__(self, in_channels, mask_type, reverse_mask, no_activation=False):
        super(CouplingLayer, self).__init__()

        # Save mask info
        self.mask_type = mask_type
        self.reverse_mask = reverse_mask

        # Build scale and translate network
        if self.mask_type == MaskType.CHANNEL_WISE:
            in_channels //= 2

        self.f_conv1 = convkxk(in_channels, in_channels, kernel=3)
        self.f_conv2 = convkxk(in_channels, in_channels, kernel=3)

        # build module
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
        if self.mask_type == MaskType.CHECKERBOARD:
            # Checkerboard mask
            b = checkerboard_mask(x.size(2), x.size(3), self.reverse_mask, device=x.device)
            x_id = x * b
            x2 = self.f_func(x_id)
            x = x + x2 * (1 - b)
        else:
            # Channel-wise mask
            if self.reverse_mask:
                x_id, x_change = x.chunk(2, dim=1)
            else:
                x_change, x_id = x.chunk(2, dim=1)

            x2 = self.f_func(x_id)
            x_change = x_change + x2

            if self.reverse_mask:
                x = torch.cat((x_id, x_change), dim=1)
            else:
                x = torch.cat((x_change, x_id), dim=1)

        return x

    def reverse(self, x):
        if self.mask_type == MaskType.CHECKERBOARD:
            # Checkerboard mask
            b = checkerboard_mask(x.size(2), x.size(3), self.reverse_mask, device=x.device)
            x_id = x * b
            x2 = self.f_func(x_id)
            x = x - x2 * (1 - b)
        else:
            # Channel-wise mask
            if self.reverse_mask:
                x_id, x_change = x.chunk(2, dim=1)
            else:
                x_change, x_id = x.chunk(2, dim=1)

            x2 = self.f_func(x_id)
            x_change = x_change - x2

            if self.reverse_mask:
                x = torch.cat((x_id, x_change), dim=1)
            else:
                x = torch.cat((x_change, x_id), dim=1)

        return x


class SqueezeLayer(nn.Module):

    def __init__(self, squeeze=True):
        super(SqueezeLayer, self).__init__()
        self.squeeze = squeeze

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1)

        if self.squeeze:
            if h % 2 != 0 or w % 2 != 0:
                raise ValueError('Expected even spatial dims HxW, got {}x{}'.format(h, w))
            x = x.view(b, h // 2, 2, w // 2, 2, c)
            x = x.permute(0, 1, 3, 5, 2, 4)
            x = x.contiguous().view(b, h // 2, w // 2, c * 4)
        else:
            if c % 4 != 0:
                raise ValueError('Number of channels {} is not divisible by 4'.format(c))
            x = x.view(b, h, w, c // 4, 2, 2)
            x = x.permute(0, 1, 4, 2, 5, 3)
            x = x.contiguous().view(b, 2 * h, 2 * w, c // 4)

        x = x.permute(0, 3, 1, 2)

        return x

    def reverse(self, x):

        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1)

        if self.squeeze:
            if c % 4 != 0:
                raise ValueError('Number of channels {} is not divisible by 4'.format(c))
            x = x.view(b, h, w, c // 4, 2, 2)
            x = x.permute(0, 1, 4, 2, 5, 3)
            x = x.contiguous().view(b, 2 * h, 2 * w, c // 4)
        else:
            if h % 2 != 0 or w % 2 != 0:
                raise ValueError('Expected even spatial dims HxW, got {}x{}'.format(h, w))
            x = x.view(b, h // 2, 2, w // 2, 2, c)
            x = x.permute(0, 1, 3, 5, 2, 4)
            x = x.contiguous().view(b, h // 2, w // 2, c * 4)

        x = x.permute(0, 3, 1, 2)

        return x
