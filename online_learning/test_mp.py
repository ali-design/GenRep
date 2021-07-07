
from __future__ import print_function
from multiprocessing import Pool
import functools

import numpy as np
import pdb
import os
import sys
import argparse
import time
import math

import torchvision.utils as vutils
import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

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
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
#import tensorflow_hub as hub
from scipy.stats import truncnorm
#import utils_bigbigan as ubigbi
from tqdm import tqdm
import json
import pickle

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

from multiprocessing import Pool

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
import math
import numpy as np
import torch
import torch.optim as optim
import random
import glob
import os 
import xml.etree.ElementTree as ET
from PIL import Image
import json
from scipy.stats import truncnorm
import random


def f(x):
    return x*x

def func(x):
    return x

def train():
    pool = Pool(5)
    data = pool.map(func, [0,1,2])
    print(data)
            

if __name__ == '__main__':
    train()
    #with Pool(5) as p:
    #    print(p.map(f, [1, 2, 3]))

