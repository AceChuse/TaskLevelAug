#!/usr/bin/python3.6

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from metamodels.dropblock import DropBlock

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']


def densenet121(**kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)
    return model


def densenet169(**kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)
    return model


def densenet201(**kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), **kwargs)
    return model


def densenet161(**kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24), **kwargs)
    return model


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, dropblock_size):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        #self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('relu1', nn.LeakyReLU(0.1)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        #self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('relu2', nn.LeakyReLU(0.1)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.dropblock_size = dropblock_size
        self.list_modules = list(self._modules.values())
        self.DropBlock = DropBlock(block_size=dropblock_size)

    def forward(self, x):
        if self.training: self.num_batches_tracked += 1

        #new_features = super(_DenseLayer, self).forward(x)
        new_features = x
        for module in self.list_modules:
            new_features = module(new_features)

        if self.drop_rate > 0:
            if self.dropblock_size is not None:
                feat_size = new_features.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20000 * DropBlock.rate) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.dropblock_size**2 * feat_size**2 / (feat_size - self.dropblock_size + 1)**2
                new_features = self.DropBlock(new_features, gamma=gamma)
            else:
                new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, dropblock_size=None):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, dropblock_size)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        #self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('relu', nn.LeakyReLU(0.1))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0., dropblock_size=None, avg_pool=False):

        super(DenseNet, self).__init__()

        self.growth_rate = growth_rate
        self.block_config = block_config
        self.num_init_features = num_init_features
        self.bn_size = bn_size
        self.drop_rate = drop_rate
        self.dropblock_size = dropblock_size
        self.avg_pool = avg_pool

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=5, stride=1, padding=2, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            #('relu0', nn.ReLU(inplace=True)),
            ('relu0', nn.LeakyReLU(0.1)),
            ('pool0', nn.AvgPool2d(2)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, dropblock_size=dropblock_size)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.num_features = num_features

        # Final relu
        #self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('relu5', nn.LeakyReLU(0.1))

        # Final pool
        if avg_pool:
            self.features.add_module('pool5', nn.AdaptiveAvgPool2d((1, 1)))

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'growth_rate=' + str(self.growth_rate) + ', ' \
               + str(self.block_config) + ', ' \
               + 'num_init_features=' + str(self.num_init_features) + ', ' \
               + 'bn_size=' + str(self.bn_size) + ', ' \
               + 'drop_rate=' + str(self.drop_rate) + ', ' \
               + 'dropblock_size=' + str(self.dropblock_size) + ', ' \
               + 'avg_pool=' + str(self.avg_pool) + ', ' \
               + 'num_features=' + str(self.num_features) + ')'