import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli

from metamodels.ResNet12_embedding import ResNet


class BatchNorm2dForP(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm2dForP, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.collecting = True
        self.current = self.batch_size = 0
        self.register_buffer('current_mean', torch.zeros(num_features))
        self.register_buffer('current_var', torch.ones(num_features))

    def forward(self, inpt):
        self._check_input_dim(inpt)

        if self.training or not self.track_running_stats:
            if self.collecting:
                if self.current != self.batch_size:
                    raise ValueError('Collection Error!')
                # exponential_average_factor is self.momentum set to
                # (when it is available) only so that if gets updated
                # in ONNX graph when this node is exported to ONNX.
                if self.momentum is None:
                    exponential_average_factor = 0.0
                else:
                    exponential_average_factor = self.momentum

                if self.training and self.track_running_stats:
                    # TODO: if statement only here to tell the jit to skip emitting this when it is None
                    if self.num_batches_tracked is not None:
                        self.num_batches_tracked += 1
                        if self.momentum is None:  # use cumulative moving average
                            exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                        else:  # use exponential moving average
                            exponential_average_factor = self.momentum

                _inpt = inpt.transpose(0, 1).reshape(self.num_features, -1)
                self.current_mean, self.current_var = _inpt.mean(dim=1), _inpt.var(dim=1, unbiased=True)

                self.running_mean = (1. - exponential_average_factor) * self.running_mean + \
                                    exponential_average_factor * self.current_mean
                self.running_var = (1. - exponential_average_factor) * self.running_var + \
                                   exponential_average_factor * self.current_var
                self.current_var = _inpt.var(dim=1, unbiased=False)

                torch.cuda.empty_cache()

                self.batch_size = inpt.size(0)
                self.current = 0
            else:
                parall_num = inpt.size(0)
                self.current += parall_num
            running_mean, running_var = self.current_mean, self.current_var

            # return (inpt - running_mean.view(1,-1,1,1)) * \
            #        (self.weight / torch.sqrt(self.current_var + self.eps)).view(1,-1,1,1) + self.bias.view(1,-1,1,1)
        else:
            running_mean, running_var = self.running_mean, self.running_var

        return F.batch_norm(inpt, running_mean, running_var, self.weight, self.bias, False, 0., self.eps)

    def collect(self, mode=True):
        self.collecting = mode


class DropoutForP(nn.Dropout):
    def __init__(self, p=0.5, inplace=False):
        super(DropoutForP, self).__init__(p, inplace)
        self.collecting = True
        self.current = self.batch_size = 0

    def forward(self, inpt):
        if self.training:
            if self.collecting:
                if self.current != self.batch_size:
                    raise ValueError('Collection Error!')
                inpt_size = inpt.size()
                self.dind = torch.ones(inpt_size).to(device=inpt.device)
                self.dind = F.dropout(self.dind, self.p, True, True)

                self.batch_size = inpt_size[0]
                self.current = 0
                out = inpt * self.dind
                self.dind = self.dind.cpu()
                torch.cuda.empty_cache()
                return out
            else:
                parall_num = inpt.size(0)
                self.current += parall_num
                return inpt * self.dind[self.current - parall_num:self.current].to(inpt.device)
        else:
            return F.dropout(inpt, self.p, False, self.inplace)

    def collect(self, mode=True):
        self.collecting = mode


class DropBlockForP(nn.Module):
    rate = 1

    def __init__(self, block_size, drop_rate=0.5):
        super(DropBlockForP, self).__init__()
        self.block_size = block_size
        self.drop_rate = drop_rate
        self.collecting = True
        self.current = self.batch_size = 0
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        if self.training:
            if self.collecting:
                if self.current != self.batch_size:
                    raise ValueError('Collection Error!')
                self.num_batches_tracked += 1.

                batch_size, channels, height, width = x.shape
                keep_rate = max(1.0 - self.drop_rate / 20000 * self.num_batches_tracked.item(), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * width ** 2 / (width - self.block_size + 1) ** 2
                # TODO: There is something whic can be changed!

                bernoulli = Bernoulli(gamma)
                mask = bernoulli.sample(
                    (batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).to(device=x.device)
                block_mask = self._compute_block_mask(mask)
                countM = block_mask.size(0) * block_mask.size(1) * block_mask.size(2) * block_mask.size(3)
                count_ones = block_mask.sum()

                self.current = 0
                self.batch_size = batch_size
                self.dind = (countM / count_ones) * block_mask
                out = x * self.dind
                self.dind = self.dind.cpu()
                torch.cuda.empty_cache()
                return out
            else:
                parall_num = x.size(0)
                self.current += parall_num
                return x * self.dind[self.current - parall_num:self.current].to(x.device)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.shape
        # print ("mask", mask[0][0])
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1),
                # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size),  # - left_padding
            ]
        ).t().cuda()
        offsets = torch.cat((torch.zeros(self.block_size ** 2, 2).cuda().long(), offsets.long()), 1)

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            # block_idxs += left_padding
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))

        block_mask = 1 - padded_mask  # [:height, :width]
        return block_mask

    def collect(self, mode=True):
        self.collecting = mode


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlockForP(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1,
                 pool='max'):
        super(BasicBlockForP, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = BatchNorm2dForP(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2dForP(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = BatchNorm2dForP(planes)
        if pool == 'max':
            self.pool = nn.MaxPool2d(stride)
        elif pool == 'avg':
            self.pool = nn.AvgPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.block_size = block_size
        if drop_block:
            self.Drop = DropBlockForP(block_size=self.block_size, drop_rate=drop_rate)
        else:
            self.Drop = DropoutForP(p=drop_rate)

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
        out = self.pool(out)

        if self.drop_rate > 0:
            return self.Drop(out)
        else:
            return out


class ResNetForP(ResNet):

    def __init__(self, block, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5, pool='max'):
        self.inplanes = 3
        super(ResNetForP, self).__init__(block, keep_prob, avg_pool, drop_rate, dropblock_size, pool)
        self.collecting = True

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1, pool='max'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm2dForP(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size, pool))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def collect(self, mode=True):
        self.collecting = mode
        for m in self.modules():
            if isinstance(m, BatchNorm2dForP) or isinstance(m, DropoutForP) or isinstance(m, DropBlockForP):
                m.collect(mode=mode)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


def resnet_forp12(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNetForP(BasicBlockForP, keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model