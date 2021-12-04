from algo.parallel import *


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
        super(ModuleParall, self).__init__()

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