#!/usr/bin/python3.6

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils.checkpoint import checkpoint_sequential


from algo.parallel_unuse import MultiBasis, MultiBasisParall


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def to_parall(m):
    if type(m) == nn.Sequential:
        return SequentialParall()
    if type(m) == nn.Linear:
        return LinearParall()
    if type(m) == nn.Conv1d:
        return Conv1dParall()
    if type(m) == nn.ConvTranspose1d:
        return Conv1dTParall()
    if type(m) == nn.Conv2d:
        return Conv2dParall()
    if type(m) == nn.ConvTranspose2d:
        return Conv2dTParall()
    if type(m) == GRU:
        return GRUParall()
    if type(m) == nn.LayerNorm:
        return LayerNormParall()
    if type(m) == nn.BatchNorm1d or \
        type(m) == nn.BatchNorm2d or \
        type(m) == nn.BatchNorm3d:
        return BatchNormParall()
    if type(m) == nn.InstanceNorm1d or \
        type(m) == nn.InstanceNorm2d or \
        type(m) == nn.InstanceNorm3d:
        return InstanceNormParall()
    if type(m) == nn.Dropout or \
        type(m) == nn.Dropout2d or \
        type(m) == nn.Dropout3d:
        return DropoutParall()
    if type(m) == nn.AvgPool1d:
        return AvgPool1dParall()
    if type(m) == nn.AvgPool2d:
        return AvgPool2dParall()
    if type(m) == nn.MaxPool2d:
        return MaxPool2dParall()
    if type(m) == MultiBasis:
        return MultiBasisParall()
    return sameparall(m)


class NonFineTune(nn.Module):
    def __init__(self):
        super(NonFineTune, self).__init__()


class NFTSequential(NonFineTune, nn.Sequential):
    def __init__(self, *args):
        nn.Sequential.__init__(self, *args)


class ModuleParall(nn.Module):
    def inner_train(self, mode=True):
        self.inner_training = mode
        for module in self.children():
            if isinstance(module, ModuleParall):
                module.inner_train(mode)
        return self
    def inner_eval(self):
        return self.inner_train(False)
    def repeat_param(self, param, wname):
        raise NotImplementedError
    def repeat_lr(self, lr, wname):
        raise NotImplementedError


def pass_f(m, num=1):
    pass
def pass_t(mode=True):
    pass
def sameparall(m):
    m.get_parameters = pass_f
    m.inner_train = pass_t
    m.inner_eval = pass_t
    return m


class SequentialParall(ModuleParall, nn.Sequential):
    def get_parameters(self, model, num=1):
        pass


class LinearParall(ModuleParall):
    def __init__(self):
        super(LinearParall, self).__init__()

    def get_parameters(self, model, num=1):
        self.num = num
        self.weight = self.repeat_param(model.weight, 'weight')
        if model.bias is None:
            self.bias = None
        else:
            self.bias = self.repeat_param(model.bias, 'bias')

    def forward(self, x):
        output = x.unsqueeze(-2).matmul(self.weight.transpose(2,3)).squeeze(-2)
        if self.bias is not None:
            output += self.bias
        return output

    def repeat_param(self, param, wname):
        if wname == 'weight':
            return param.unsqueeze(0).repeat(self.num, 1, 1).unsqueeze(0)
        elif wname == 'bias':
            return param.unsqueeze(0).repeat(self.num, 1).unsqueeze(0)

    def repeat_lr(self, lr, wname):
        if wname == 'weight':
            lr = lr.view(1, self.num, 1, 1)
        elif wname == 'bias':
            lr = lr.view(1, self.num, 1)
        return lr

'''
if __name__ == '__main__':
    x = Variable(torch.randn(24).view(4, 2, 3)).to(device)
    xt = x.transpose(0, 1).contiguous()
    lnear = nn.Linear(3, 4).to(device)
    lnears = to_parall(lnear)
    lnears.get_parameters(lnear,4)
    k = 1
    print('x=',x)
    print('y1=',lnears(xt).transpose(0,1).contiguous()[k])
    print('y2=', torch.cat([lnear(x[i]).unsqueeze(0) for i in range(4)],0)[k])
'''

class _ConvNdParall(ModuleParall):
    def __init__(self):
        super(_ConvNdParall, self).__init__()

    def get_parameters(self, model, num=1):
        if model.groups != 1:
            raise ValueError('visionmodel.groups is not equal to 1!')
        self.stride = model.stride
        self.padding = model.padding
        if model.transposed:
            self.output_padding = model.output_padding
        self.dilation = model.dilation
        self.in_channels = model.in_channels
        self.out_channels = model.out_channels
        self.num = num

        weight_size = model.weight.size()
        self.weight_repeat = [self.num] + [1] * (len(weight_size) - 1)
        self.weight = self.repeat_param(model.weight, 'weight')
        if model.bias is None:
            self.bias = None
        else:
            self.bias = self.repeat_param(model.bias, 'bias')

    def repeat_param(self, param, wname):
        if wname == 'weight':
            return param.repeat(*self.weight_repeat)
        elif wname == 'bias':
            return param.repeat(self.num)

    def repeat_lr(self, lr, wname):
        lr = lr.view(self.num, 1).repeat(1, self.in_channels)
        if wname == 'weight':
            lr = lr.view(-1, *self.weight_repeat[1:])
        elif wname == 'bias':
            lr = lr.view(-1)
        return lr


class Conv2dParall(_ConvNdParall):
    def forward(self, x):
        si = x.size()
        x = x.view(si[0], si[1]*si[2], si[3], si[4])
        output = F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.num)
        so = output.size()[-2:]
        return output.view(si[0], si[1], self.out_channels, so[0], so[1])


class Conv2dTParall(_ConvNdParall):
    def forward(self, x):
        si = x.size()
        x = x.reshape(si[0], si[1]*si[2], si[3], si[4])
        output = F.conv_transpose2d(
            x, self.weight, self.bias, self.stride, self.padding,
            self.output_padding, self.num, self.dilation)
        so = output.size()[-2:]
        return output.view(si[0], si[1], self.out_channels, so[0], so[1])

'''
if __name__ == '__main__':
    conv = nn.Conv2d(2, 3, 2).to(device)
    #conv = nn.ConvTranspose2d(2,3,1).to(device)
    #print('weight=', conv.weight)
    #print('bias=', conv.bias)
    convs = to_parall(conv)
    #convs = Conv2dTParall()
    convs.get_parameters(conv, 4)
    x = Variable(torch.randn(16 * 9).view(4, 2, 2, 3, 3)).to(device)
    xt = x.transpose(0,1).contiguous()
    k = 1
    print('x=', x)
    print('y1=', convs(xt).transpose(0,1).contiguous()[k])
    print('y2=', torch.cat([conv(x[i]).unsqueeze(0) for i in range(4)], 0)[k])
'''

class Conv1dParall(_ConvNdParall):
    def forward(self, x):
        si = x.size()
        x = x.view(si[0], si[1]*si[2], si[3])
        output = F.conv1d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.num)
        so = output.size()[-1]
        return output.view(si[0], si[1], self.out_channels, so)


class Conv1dTParall(_ConvNdParall):
    def forward(self, x):
        si = x.size()
        x = x.reshape(si[0], si[1]*si[2], si[3])
        output = F.conv_transpose1d(
            x, self.weight, self.bias, self.stride, self.padding,
            self.output_padding, self.num, self.dilation)
        so = output.size()[-1]
        return output.view(si[0], si[1], self.out_channels, so)

'''
if __name__ == '__main__':
    conv = nn.Conv1d(2, 3, 2).to(device)
    #conv = nn.ConvTranspose1d(2,3,2).to(device)
    #print('weight=', conv.weight)
    #print('bias=', conv.bias)
    convs = to_parall(conv)
    convs.get_parameters(conv, 4)
    x = Variable(torch.randn(16 * 6).view(4, 2, 2, 6)).to(device)
    xt = x.transpose(0,1).contiguous()
    k = 1
    #print('x=', x)
    print('y1=', convs(xt).transpose(0,1).contiguous()[k])
    print('y2=', torch.cat([conv(x[i]).unsqueeze(0) for i in range(4)], 0)[k])
'''

class GRU(nn.GRUCell):
    def __init__(self, input_size, hidden_size, output_num, bias=True, batch_first=True):
        super(GRU, self).__init__(input_size, hidden_size, bias)
        self.output_num = output_num
        self.batch_first = batch_first

    def forward(self, x, hx=None):
        x = x.transpose(0, 1).contiguous() if self.batch_first else x
        self.check_forward_input(x[0])
        if hx is None:
            hx = x.new_zeros(x[0].size(0), self.hidden_size, requires_grad=False)
        self.check_forward_hidden(x[0], hx)
        y = []
        for _x in x:
            hx = self._backend.GRUCell(
                _x, hx,
                self.weight_ih, self.weight_hh,
                self.bias_ih, self.bias_hh,
            )
            y.append(hx.unsqueeze(0))
        y = y if self.output_num == -1 else y[-self.output_num:]
        y = torch.cat(y, dim=0)
        return y.transpose(0, 1).contiguous() if self.batch_first else y


class GRUParall(ModuleParall):
    def __init__(self):
        super(GRUParall, self).__init__()

    def get_parameters(self, model, num=1):
        self.num = num
        self.input_size = model.input_size
        self.hidden_size = model.hidden_size
        self.output_num = model.output_num
        self.batch_first = model.batch_first

        self.bias = model.bias
        self.weight_ih = self.repeat_param(model.weight_ih, 'weight_ih')
        self.weight_hh = self.repeat_param(model.weight_hh, 'weight_hh')
        if self.bias:
            self.bias_ih = self.repeat_param(model.bias_ih, 'bias_ih')
            self.bias_hh = self.repeat_param(model.bias_hh, 'bias_hh')
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, x, hx=None):
        x = x.permute(2, 0, 1, 3).contiguous() if self.batch_first else x
        # x.size() = (sequence, batch_size, meta_batch_size, input_size)
        if hx is None:
            hx = torch.zeros(*x.size()[1:-1], self.hidden_size, requires_grad=False).to(device)
        y = []
        for _x in x:
            hx = self._GRUCell(
                _x, hx, # _x.size() = (batch_size, meta_batch_size, input_size)
                self.weight_ih, self.weight_hh,
                self.bias_ih, self.bias_hh,
            )# hx.size() = (batch_size, meta_batch_size, hidden_size)
            y.append(hx.unsqueeze(-2))
        y = y if self.output_num == -1 else y[-self.output_num:]
        y = torch.cat(y, dim=-2)
        # y.size() = (batch_size, meta_batch_size, sequence, input_size)
        return y.permute(2, 0, 1, 3).contiguous() if not self.batch_first else y

    def meta_mul(self, x, weight, bias):
        output = x.unsqueeze(-2).matmul(weight.transpose(2, 3)).squeeze(-2)
        if bias is not None:
            output += bias
        return output

    def _GRUCell(self, x, hx, weight_ih, weight_hh, bias_ih, bias_hh):
        ih = self.meta_mul(x, weight_ih, bias_ih) # hi.size() = (batch_size, meta_batch_size, 3 * hidden_size)
        hh = self.meta_mul(hx, weight_hh, bias_hh)
        rz = torch.sigmoid(ih[:, :, :2 * self.hidden_size] + hh[:, :, :2 * self.hidden_size])
        n = torch.tanh(ih[:, :, 2 * self.hidden_size:] + rz[:, :, :self.hidden_size] * hh[:, :, 2 * self.hidden_size:])
        hx = (1-rz[:, :, self.hidden_size:]) * n + rz[:, :, self.hidden_size:] * hx
        return hx

    def repeat_param(self, param, wname):
        if wname[:6] == 'weight':
            return param.unsqueeze(0).repeat(self.num, 1, 1).unsqueeze(0)
        elif wname[:4] == 'bias':
            return param.unsqueeze(0).repeat(self.num, 1).unsqueeze(0)

'''
if __name__ == '__main__':
    gru = GRU(2, 3, -1).to(device)

    # gru.weight_hh[-3:] = 0
    # gru.weight_ih[-3:] = 0
    # gru.bias_hh[-3:] = 1
    # gru.bias_ih[-3:] = 0
    
    # gru.weight_hh[3:6] = 0
    # gru.weight_ih[3:6] = 0
    # gru.bias_hh[3:6] = -1e10
    # gru.bias_ih[3:6] = -1e10
    
    grus = to_parall(gru)
    grus.get_parameters(gru, 4)
    x = Variable(torch.randn(16 * 6).view(4, 2, 6, 2)).to(device)
    # x.fill_(0.1)
    xt = x.transpose(0,1).contiguous()
    k = 0
    print('y1=', grus(xt).transpose(0,1).contiguous()[k])
    print('y2=', torch.cat([gru(x[i]).unsqueeze(0) for i in range(4)], 0)[k])
'''

class LayerNormParall(ModuleParall):
    def __init__(self):
        super(LayerNormParall, self).__init__()

    def get_parameters(self, model, num=1):
        self.normalized_shape = [num]
        self.re_num = [num]
        self.normalized_shape.extend(list(model.normalized_shape))
        self.len_norm_shape = len(model.normalized_shape)
        self.re_num.extend([1] * self.len_norm_shape)
        self.eps = model.eps
        self.elementwise_affine = model.elementwise_affine

        if self.elementwise_affine:
            if model.weight is None:
                self.weight = None
            else:
                self.weight = model.weight.clone()
                self.weight = self.repeat_param(self.weight, 'weight')

            if model.bias is None:
                self.bias = None
            else:
                self.bias = model.bias.clone()
                self.bias = self.repeat_param(self.bias, 'bias')

    def forward(self, x):
        in_size = x.size()
        re_size = list(in_size[:-self.len_norm_shape])
        re_size.append(-1)
        x = x.view(*re_size)
        x = (x - x.mean(-1, keepdim=True).detach()) / \
                torch.sqrt(x.var(-1, keepdim=True, unbiased=False).detach() + self.eps)
        x = x.view(in_size)
        normalized_shape = [1] + self.normalized_shape[:1] + [1] * (
            len(in_size) - self.len_norm_shape - 2) + self.normalized_shape[1:]

        if self.elementwise_affine:
            weight = self.weight.view(normalized_shape)
            bias = self.bias.view(normalized_shape)
            return x * weight + bias
        else:
            return x

    def repeat_param(self, param, wname):
        if wname == 'weight':
            return param.unsqueeze(0).repeat(*self.re_num)
        elif wname == 'bias':
            return param.unsqueeze(0).repeat(*self.re_num)

'''
if __name__ == '__main__':
    x = torch.randn(4, 3, 2, 2).type(FloatTensor)
    xt = x.transpose(0,1).contiguous()
    ln = nn.LayerNorm([2,2]).cuda()
    lns = LayerNormParall()
    lns.get_parameters(ln,num=4)
    k = 3
    print('x=', x)
    print('y1=', lns(xt).transpose(0,1).contiguous()[k])
    print('y2=', torch.cat([ln(x[i]).unsqueeze(0) for i in range(4)], 0)[k])
'''

class BatchNormParall(ModuleParall):
    def __init__(self):
        super(BatchNormParall, self).__init__()

    def get_parameters(self, model, num=1):
        self.num = num
        self.num_features = model.num_features
        self.eps = model.eps
        self.momentum = model.momentum
        self.affine = model.affine
        self.training = model.training
        self.track_running_stats = model.track_running_stats

        if self.affine:
            self.weight = model.weight.clone()
            self.weight = self.repeat_param(self.weight, 'weight')
            self.bias = model.bias.clone()
            self.bias = self.repeat_param(self.bias, 'bias')
        else:
            self.weight = None
            self.bias = None
        if self.track_running_stats:
            raise ValueError('track_running_stats is True!')
            self.running_mean = model.running_mean
            self.running_mean = self.running_mean.repeat(num)
            self.running_var = model.running_var
            self.running_var = self.running_var.repeat(num)
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, x):
        in_size = list(x.size())
        re_size = in_size[0:1] + [in_size[1] * in_size[2]] + in_size[3:]
        x = x.view(*re_size)
        return F.batch_norm(
            x, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps).view(in_size)

    def repeat_param(self, param, wname):
        if wname == 'weight':
            return param.repeat(self.num)
        elif wname == 'bias':
            return param.repeat(self.num)

    def repeat_lr(self, lr, wname):
        lr = lr.view(self.num, 1).repeat(1, self.num_features)
        lr = lr.view(-1)
        return lr


class InstanceNormParall(BatchNormParall):
    def __init__(self):
        super(InstanceNormParall, self).__init__()

    def forward(self, x):
        in_size = list(x.size())
        re_size = in_size[0:1] + [in_size[1] * in_size[2]] + in_size[3:]
        x = x.view(*re_size)
        return F.instance_norm(
            x, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps).view(in_size)

'''
if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(4, 3, 2, 2, 2).to(device)
    #x = torch.randn(4, 3, 2, 2).to(device)
    xt = x.transpose(0,1).contiguous()
    #bn = nn.BatchNorm2d(2).to(device)
    #bn = nn.BatchNorm1d(2).to(device)
    bn = nn.InstanceNorm2d(2, affine=True).to(device)
    #bn = nn.InstanceNorm1d(2, affine=True).to(device)
    #bn.eval()
    #bns = BatchNormParall()
    bns = InstanceNormParall()
    bns.get_parameters(bn,num=4)
    k = 2
    print('x=', x)
    print('y1=', bns(xt).transpose(0,1).contiguous()[k])
    print('y2=', torch.cat([bn(x[i]).unsqueeze(0) for i in range(4)], 0)[k])
'''

class AvgPool2dParall(ModuleParall):
    def __init__(self):
        super(AvgPool2dParall, self).__init__()

    def get_parameters(self, model, num=1):
        if isinstance(model.kernel_size, int):
            self.kernel_size = (1, model.kernel_size, model.kernel_size)
        else:
            self.kernel_size = (1, model.kernel_size[0], model.kernel_size[1])
        if isinstance(model.stride, int):
            self.stride = (1, model.stride, model.stride)
        else:
            self.stride = (1, model.stride[0], model.stride[1])
        self.padding = model.padding
        self.ceil_mode = model.ceil_mode
        self.count_include_pad = model.count_include_pad

    def forward(self, x):
        return F.avg_pool3d(x, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad)


class MaxPool2dParall(ModuleParall):
    def __init__(self):
        super(MaxPool2dParall, self).__init__()

    def get_parameters(self, model, num=1):
        if isinstance(model.kernel_size, int):
            self.kernel_size = (1, model.kernel_size, model.kernel_size)
        else:
            self.kernel_size = (1, model.kernel_size[0], model.kernel_size[1])
        if isinstance(model.stride, int):
            self.stride = (1, model.stride, model.stride)
        else:
            self.stride = (1, model.stride[0], model.stride[1])
        self.padding = model.padding
        self.dilation = model.dilation
        self.return_indices = model.return_indices
        self.ceil_mode = model.ceil_mode

    def forward(self, x):
        return F.max_pool3d(x, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)

'''
if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(4, 3, 2, 4, 4).type(FloatTensor)
    xt = x.transpose(0,1).contiguous()
    #ap = nn.AvgPool2d(2).cuda()
    ap = nn.MaxPool2d(2).cuda()
    #aps = AvgPool2dParall()
    aps = MaxPool2dParall()
    aps.get_parameters(ap,num=4)
    k = 2
    print('x=', x)
    print('y1=', aps(xt).transpose(0,1).contiguous()[k])
    print('y2=', torch.cat([ap(x[i]).unsqueeze(0) for i in range(4)], 0)[k])
'''

class AvgPool1dParall(ModuleParall):
    def __init__(self):
        super(AvgPool1dParall, self).__init__()

    def get_parameters(self, model, num=1):
        if len(model.kernel_size) == 1:
            self.kernel_size = (1, model.kernel_size[0])
        else:
            raise ValueError('kernel_size should be 1 dim.')
        if len(model.stride) == 1:
            self.stride = (1, model.stride[0])
        else:
            raise ValueError('stride should be 1 dim.')
        self.padding = model.padding
        self.ceil_mode = model.ceil_mode
        self.count_include_pad = model.count_include_pad

    def forward(self, x):
        return F.avg_pool2d(x, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad)

'''
if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(4, 3, 2, 4).type(FloatTensor)
    xt = x.transpose(0,1).contiguous()
    ap = nn.AvgPool1d(2).cuda()
    #ap = nn.MaxPool2d(2).cuda()
    aps = AvgPool1dParall()
    #aps = MaxPool2dParall()
    aps.get_parameters(ap,num=4)
    k = 2
    print('x=', x)
    print('y1=', aps(xt).transpose(0,1).contiguous()[k])
    print('y2=', torch.cat([ap(x[i]).unsqueeze(0) for i in range(4)], 0)[k])
'''

class DropoutParall(ModuleParall):
    def __init__(self):
        super(DropoutParall, self).__init__()

    def get_parameters(self, model, num=1):
        self.call = (model,)

    def forward(self, x):
        in_size = list(x.size())
        re_size = [in_size[0]*in_size[1]] + in_size[2:]
        x = x.view(*re_size)
        return self.call[0](x).view(*in_size)

    def inner_train(self, mode=True):
        if self.training:
            self.call[0].train(not mode)
        else:
            self.call[0].eval()
        self.inner_training = mode
        for module in self.children():
            module.inner_train(mode)
        return self

'''
if __name__ == '__main__':
    torch.manual_seed(0)
    x = Variable(torch.arange(0, 48*3).view(4, 2, 3, 2, 3)).type(FloatTensor)
    xt = x.transpose(0, 1).contiguous()
    dp = nn.Dropout2d(p=0.2)
    dps = DropoutParall()
    dps.get_parameters(dp,4)
    k = 1
    print('x=',x)
    print('y1=',dps(xt).transpose(0,1).contiguous()[k])
    print('y2=', torch.cat([dp(x[i]).unsqueeze(0) for i in range(4)],0))
'''

class MultiBasis(nn.Module):
    def __init__(self, module, basis_num, outdims=[]):
        super(MultiBasis, self).__init__()
        self.basis_num = basis_num
        self.outdims = outdims
        if self.outdims:
            p = iter(module.parameters()).__next__()
            self.outsize = p.size(self.outdims[0])

        self.p_names = []
        self.p_dims = []
        for name, p in module.named_parameters():
            nonft_name = 'nonft_' + name
            setattr(self, nonft_name, Parameter(torch.Tensor(self.basis_num, *p.size())))
            nonft_p = getattr(self, nonft_name)
            nonft_p.data[0] = p.data
            self.p_names.append(name)
            self.p_dims.append([1] * len(p.size()))

        if self.outdims:
            for p_dim, outdim in zip(self.p_dims, self.outdims):
                p_dim[outdim] = self.outsize

        if self.p_names:
            for i in range(1, self.basis_num):
                module.reset_parameters()
                for name, p in module.named_parameters():
                    nonft_name = 'nonft_' + name
                    nonft_p = getattr(self, nonft_name)
                    nonft_p.data[i] = p.data

        for name in self.p_names:
            delattr(module, name)
            setattr(module, name, 0.)

        self.copy_module = module
        if self.p_names:
            self.basis_weight = Parameter(torch.Tensor(self.basis_num, self.outsize)) \
                if self.outdims else Parameter(torch.Tensor(self.basis_num))
        else:
            self.register_parameter('basis_weight', None)
        self.extra_repr_str = module.extra_repr()
        self.reset_parameters()

    def reset_parameters(self):
        if self.basis_weight is not None:
            self.basis_weight.data.fill_(1. / self.basis_num)

    def merge_basis(self, weight):
        for name, dim in zip(self.p_names, self.p_dims):
            p = getattr(self, 'nonft_' + name)
            setattr(self.copy_module, name, p.mul(weight.view(self.basis_num, *dim)).sum(0))

    def forward(self, inpt):
        self.merge_basis(self.basis_weight)
        return self.copy_module(inpt)

    def extra_repr(self):
        s = self.copy_module._get_name() + ', basis_num=' + str(self.basis_num) + ', '
        if self.outdims:
            s += 'outdims=' + str(self.outdims) + ', '
        return s + self.copy_module.extra_repr()

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str


class MultiBasisParall(ModuleParall):
    def __init__(self):
        super(MultiBasisParall, self).__init__()

    def get_parameters(self, model, num=1):
        self.num = num
        self.model = model
        self.basis_weight = self.repeat_param(model.basis_weight, 'basis_weight')
        if self.copy_module.get_parameters is not pass_f:
            self.copy_module_get_parameters = self.copy_module.get_parameters
            self.copy_module.get_parameters = pass_f

    def forward(self, *inpt):
        self.model.merge_basis(self.basis_weight)
        self.copy_module_get_parameters(self.model.copy_module, self.num)
        return self.copy_module.forward(*inpt)

    def repeat_param(self, param, wname):
        if wname == 'basis_weight':
            return param.clone()

'''
def module2parall(module):
    if isinstance(module, NonFineTune):
        return module
    module_parall = to_parall(module)
    for name, _module in module.named_children():
        module_parall.add_module(name, module2parall(_module))
    return module_parall


if __name__ == '__main__':
    x = Variable(torch.randn(24).view(4, 2, 3)).to(device)
    xt = x.transpose(0, 1).contiguous()
    lnear = MultiBasis(nn.Linear(3, 4), 3, outdims=[0,0]).to(device)
    print(lnear)
    lnears = module2parall(lnear)
    lnears.get_parameters(lnear,4)
    k = 1
    print('x=',x)
    print('y1=',lnears(xt).transpose(0,1).contiguous()[k])
    print('y2=', torch.cat([lnear(x[i]).unsqueeze(0) for i in range(4)],0)[k])


if __name__ == '__main__':
    conv = MultiBasis(nn.Conv2d(2, 3, 2), 3, outdims=[0,0]).to(device)
    #conv = MultiBasis(nn.ConvTranspose2d(2,3,1), 3, outdims=[0,0]).to(device)
    convs = module2parall(conv)
    convs.get_parameters(conv, 4)
    x = Variable(torch.randn(16 * 9).view(4, 2, 2, 3, 3)).to(device)
    xt = x.transpose(0,1).contiguous()
    k = 1
    print('x=', x)
    print('y1=', convs(xt).transpose(0,1).contiguous()[k])
    print('y2=', torch.cat([conv(x[i]).unsqueeze(0) for i in range(4)], 0)[k])
'''